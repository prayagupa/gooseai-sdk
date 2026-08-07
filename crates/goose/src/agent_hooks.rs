//! Bridge that makes goose a host for the [agent-hooks] control contract.
//!
//! [agent-hooks] is a framework-neutral *control* contract: a fixed set of
//! agent lifecycle interception points, the `AgentContext` a host supplies at
//! each, and the `Verdict` an interceptor returns (`allow` / `deny` /
//! `transform`). This module exposes an [`AgentHooksInspector`] that implements
//! goose's [`ToolInspector`] trait, so any agent-hooks interceptor (a policy
//! engine, content filter, egress guard, ...) can govern goose tool calls
//! through the existing tool-inspection pipeline.
//!
//! # Scope (v1)
//!
//! The bridge governs the **`pre_tool_call`** interception point only — the
//! seam where goose's inspectors already run. The seven other points
//! (`agent_startup`, `input`, model calls, `output`, `agent_shutdown`) are not
//! yet wired; extending coverage is future work.
//!
//! Because goose's tool-inspection seam decides only allow / deny /
//! require-approval and then executes the *original* request, a `transform`
//! verdict at `pre_tool_call` cannot be applied here. It is therefore treated
//! as a **fail-closed deny** rather than silently running the un-transformed
//! call.
//!
//! # Verdict mapping
//!
//! | agent-hooks verdict            | goose [`InspectionAction`]        |
//! | ------------------------------ | --------------------------------- |
//! | `allow`                        | `Allow` (no finding emitted)      |
//! | `deny`                         | `Deny`                            |
//! | `escalate` (deny + approval)   | `RequireApproval` (goose asks)    |
//! | `transform`                    | `Deny` (fail closed; see above)   |
//!
//! In `evaluate_only` mode no finding is ever emitted: decisions are recorded
//! but never enforced.
//!
//! # Trust model
//!
//! agent-hooks is a *cooperative* control contract, **not** a security
//! boundary. Interceptors run in-process with full data access and the
//! interception points do not guarantee complete mediation. goose remains the
//! enforcement authority. See the agent-hooks `SECURITY.md` before relying on
//! it for isolation.
//!
//! # Usage
//!
//! ```ignore
//! use goose::agent_hooks::{install, AgentHooksInspector};
//!
//! // Once, at process startup, before any Agent is created:
//! install(|| {
//!     AgentHooksInspector::builder()
//!         .register(Box::new(MyPolicyInterceptor::new()))
//!         .build()
//! })
//! .expect("install agent-hooks factory once");
//! ```
//!
//! [agent-hooks]: https://github.com/responsibleai/agent-hooks

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use agent_hooks::{
    AgentContext, AgentContextBuilder, CompositionConfig, Decision, EnforcementMode,
    InterceptionBlocked, InterceptionEmitter, InterceptionRecord, Interceptor, Verdict,
};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;

use crate::config::GooseMode;
use crate::conversation::message::{Message, ToolRequest};
use crate::tool_inspection::{InspectionAction, InspectionResult, ToolInspector};

/// `framework` identifier stamped on every emitted `AgentContext`.
const FRAMEWORK: &str = "goose";

/// [`ToolInspector::name`] for this bridge.
const INSPECTOR_NAME: &str = "agent_hooks";

/// Default per-interceptor timeout (agent-hooks §7 RECOMMENDED default).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Default bound on the per-session context-builder cache.
const DEFAULT_MAX_SESSIONS: usize = 1024;

/// Default bound on the emitter's in-memory record buffer.
const DEFAULT_MAX_RECORDS: usize = 1000;

/// Reason surfaced when a `pre_tool_call` transform cannot be applied.
const TRANSFORM_UNSUPPORTED: &str = "agent_hooks:transform_unsupported (a pre_tool_call transform \
     cannot be applied through the goose tool-inspection seam; failing closed)";

/// Human-readable, payload-free reason for a blocked or escalated tool call.
fn verdict_reason(verdict: &Verdict) -> String {
    let reason = verdict.reason.as_deref().unwrap_or("").trim();
    let message = verdict.message.as_deref().unwrap_or("").trim();
    match (reason.is_empty(), message.is_empty()) {
        (false, false) if reason != message => format!("{reason}: {message}"),
        (false, _) => reason.to_string(),
        (true, false) => message.to_string(),
        (true, true) => "blocked by agent-hooks policy".to_string(),
    }
}

/// Bounded, per-session cache of [`AgentContextBuilder`]s.
///
/// Each session needs its own builder so the agent-hooks `sequence` field stays
/// monotonic within a session. The cache is FIFO-bounded so a long-running
/// process serving many sessions cannot grow without limit.
struct SessionBuilders {
    agent_id: String,
    max: usize,
    map: HashMap<String, AgentContextBuilder>,
    order: VecDeque<String>,
}

impl SessionBuilders {
    fn new(agent_id: String, max: usize) -> Self {
        Self {
            agent_id,
            max: max.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn builder_for(&mut self, session_id: &str) -> &mut AgentContextBuilder {
        if !self.map.contains_key(session_id) {
            while self.order.len() >= self.max {
                match self.order.pop_front() {
                    Some(oldest) => {
                        self.map.remove(&oldest);
                    }
                    None => break,
                }
            }
            self.map.insert(
                session_id.to_string(),
                AgentContextBuilder::new(&self.agent_id, FRAMEWORK, session_id),
            );
            self.order.push_back(session_id.to_string());
        }
        self.map
            .get_mut(session_id)
            .expect("builder inserted above is present")
    }
}

/// A goose [`ToolInspector`] that enforces agent-hooks interceptor verdicts at
/// the `pre_tool_call` interception point. Build one with [`Self::builder`].
pub struct AgentHooksInspector {
    /// Shared emitter (owns the interceptors). `emit` needs `&mut self` and
    /// awaits interceptors, so it is guarded by an async mutex; every emission
    /// is therefore an atomic read-check-record transaction.
    emitter: AsyncMutex<InterceptionEmitter>,
    /// Per-session context builders. Held only across the synchronous context
    /// build (never across an `.await`).
    builders: Mutex<SessionBuilders>,
    /// Whether verdicts are enforced (`Enforce`) or only recorded
    /// (`EvaluateOnly`).
    enforcing: bool,
}

impl AgentHooksInspector {
    /// Start building an inspector.
    pub fn builder() -> AgentHooksInspectorBuilder {
        AgentHooksInspectorBuilder::new()
    }
}

#[async_trait]
impl ToolInspector for AgentHooksInspector {
    fn name(&self) -> &'static str {
        INSPECTOR_NAME
    }

    async fn inspect(
        &self,
        session_id: &str,
        tool_requests: &[ToolRequest],
        _messages: &[Message],
        _goose_mode: GooseMode,
    ) -> Result<Vec<InspectionResult>> {
        let mut results = Vec::new();

        for request in tool_requests {
            // A malformed tool call has no action to govern; other inspectors
            // skip it too, and the agent surfaces the parse error separately.
            let Ok(tool_call) = &request.tool_call else {
                continue;
            };
            let name = tool_call.name.as_ref();
            let args = Value::Object(tool_call.arguments.clone().unwrap_or_default());

            // Build the context under the sync lock, then release it before the
            // async emit so the builder mutex is never held across an `.await`.
            let mut ctx: AgentContext = {
                let mut builders = self
                    .builders
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                builders
                    .builder_for(session_id)
                    .pre_tool_call(&request.id, name, args)
            };

            let outcome = {
                let mut emitter = self.emitter.lock().await;
                emitter.emit(&mut ctx).await
            };

            let mapped: Option<(InspectionAction, String)> = match outcome {
                Ok(permit) => {
                    if self.enforcing {
                        match permit.record.verdict.decision {
                            Decision::Allow => None,
                            // A transform we cannot apply here fails closed.
                            Decision::Transform => {
                                Some((InspectionAction::Deny, TRANSFORM_UNSUPPORTED.to_string()))
                            }
                            // Defensive: emit() only returns Ok when the action
                            // proceeds, so a Deny here would be an SDK invariant
                            // break — still fail closed.
                            Decision::Deny => Some((
                                InspectionAction::Deny,
                                verdict_reason(&permit.record.verdict),
                            )),
                        }
                    } else {
                        // evaluate_only: recorded, never enforced.
                        None
                    }
                }
                // emit() returns Err only in enforce mode (evaluate_only always
                // proceeds), so a block here is already mode-checked.
                Err(InterceptionBlocked { record }) => {
                    let reason = verdict_reason(&record.verdict);
                    let action = if record.verdict.approval.is_some() {
                        // A liftable deny maps to goose's native approval flow.
                        InspectionAction::RequireApproval(Some(reason.clone()))
                    } else {
                        InspectionAction::Deny
                    };
                    Some((action, reason))
                }
            };

            if let Some((action, reason)) = mapped {
                results.push(InspectionResult {
                    tool_request_id: request.id.clone(),
                    action,
                    reason,
                    confidence: 1.0,
                    inspector_name: INSPECTOR_NAME.to_string(),
                    finding_id: None,
                });
            }
        }

        Ok(results)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Builder for [`AgentHooksInspector`].
pub struct AgentHooksInspectorBuilder {
    interceptors: Vec<Box<dyn Interceptor>>,
    mode: EnforcementMode,
    timeout: Option<Duration>,
    composition: Option<CompositionConfig>,
    record_sink: Option<Box<dyn Fn(&InterceptionRecord) + Send + Sync>>,
    max_records: Option<usize>,
    max_sessions: usize,
    agent_id: String,
}

impl AgentHooksInspectorBuilder {
    /// A new builder: enforce mode, 5s timeout, defaults for the rest.
    pub fn new() -> Self {
        Self {
            interceptors: Vec::new(),
            mode: EnforcementMode::Enforce,
            timeout: Some(DEFAULT_TIMEOUT),
            composition: None,
            record_sink: None,
            max_records: Some(DEFAULT_MAX_RECORDS),
            max_sessions: DEFAULT_MAX_SESSIONS,
            agent_id: FRAMEWORK.to_string(),
        }
    }

    /// Register an interceptor. Interceptors run in registration order under
    /// the emitter's composition profile.
    pub fn register(mut self, interceptor: Box<dyn Interceptor>) -> Self {
        self.interceptors.push(interceptor);
        self
    }

    /// Enforce verdicts (default) or only record them (`EvaluateOnly`).
    pub fn mode(mut self, mode: EnforcementMode) -> Self {
        self.mode = mode;
        self
    }

    /// Per-interceptor timeout; `None` disables it. A breach fails closed.
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// Composition profile for combining multiple interceptor verdicts.
    pub fn composition(mut self, composition: CompositionConfig) -> Self {
        self.composition = Some(composition);
        self
    }

    /// Callback invoked with every audit record (for durable persistence).
    pub fn record_sink(
        mut self,
        sink: impl Fn(&InterceptionRecord) + Send + Sync + 'static,
    ) -> Self {
        self.record_sink = Some(Box::new(sink));
        self
    }

    /// Bound on the emitter's in-memory record buffer; `None` is unbounded.
    pub fn max_records(mut self, max: Option<usize>) -> Self {
        self.max_records = max;
        self
    }

    /// Bound on the per-session context-builder cache.
    pub fn max_sessions(mut self, max: usize) -> Self {
        self.max_sessions = max.max(1);
        self
    }

    /// `agent.id` stamped on every emitted context.
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = agent_id.into();
        self
    }

    /// Build the inspector.
    ///
    /// In `Enforce` mode with no interceptors, agent-hooks fails closed on every
    /// emission (`host_error:no_interceptor`), so every governed tool call is
    /// denied; this logs a warning to make that misconfiguration visible.
    pub fn build(self) -> AgentHooksInspector {
        let enforcing = matches!(self.mode, EnforcementMode::Enforce);
        if enforcing && self.interceptors.is_empty() {
            tracing::warn!(
                "AgentHooksInspector built in enforce mode with no interceptors; every governed \
                 tool call will be denied (agent-hooks fails closed on an empty enforce emitter)"
            );
        }

        let mut emitter = InterceptionEmitter::new(self.mode, None);
        if let Some(composition) = self.composition {
            emitter.set_composition(composition);
        }
        if let Some(timeout) = self.timeout {
            emitter.set_timeout(timeout);
        }
        if let Some(max) = self.max_records {
            emitter.set_max_records(max);
        }
        if let Some(sink) = self.record_sink {
            emitter.set_record_sink(sink);
        }
        for interceptor in self.interceptors {
            emitter.register(interceptor);
        }

        AgentHooksInspector {
            emitter: AsyncMutex::new(emitter),
            builders: Mutex::new(SessionBuilders::new(self.agent_id, self.max_sessions)),
            enforcing,
        }
    }
}

impl Default for AgentHooksInspectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

type InspectorFactory = Box<dyn Fn() -> AgentHooksInspector + Send + Sync>;

/// Process-wide factory installed once at startup. Write-once, so reads are
/// race-free; each `Agent` gets a freshly built inspector (its own emitter and
/// interceptor instances).
static FACTORY: OnceLock<InspectorFactory> = OnceLock::new();

/// Install the process-wide inspector factory. Call once, before any
/// [`crate::agents::Agent`] is created; every agent built afterward registers a
/// fresh [`AgentHooksInspector`] in its tool-inspection pipeline.
///
/// Returns `Err` if a factory was already installed.
pub fn install(
    factory: impl Fn() -> AgentHooksInspector + Send + Sync + 'static,
) -> Result<(), &'static str> {
    FACTORY
        .set(Box::new(factory))
        .map_err(|_| "agent-hooks inspector factory already installed")
}

/// Build an inspector from the installed factory, if any. Used by the agent's
/// tool-inspection-manager wiring.
pub(crate) fn make_inspector() -> Option<AgentHooksInspector> {
    FACTORY.get().map(|factory| factory())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_hooks::Transform;
    use rmcp::model::CallToolRequestParams;
    use rmcp::object;
    use serde_json::json;

    /// An interceptor that always returns a fixed verdict.
    struct FakeInterceptor {
        verdict: Verdict,
    }

    #[async_trait]
    impl Interceptor for FakeInterceptor {
        async fn intercept(&self, _ctx: &AgentContext) -> Verdict {
            self.verdict.clone()
        }
        fn name(&self) -> Option<String> {
            Some("fake".to_string())
        }
    }

    fn tool_request(id: &str, name: &str) -> ToolRequest {
        ToolRequest {
            id: id.to_string(),
            tool_call: Ok(
                CallToolRequestParams::new(name.to_string()).with_arguments(object!({ "k": "v" })),
            ),
            metadata: None,
            tool_meta: None,
        }
    }

    fn inspector_with(verdict: Verdict, mode: EnforcementMode) -> AgentHooksInspector {
        AgentHooksInspector::builder()
            .mode(mode)
            .register(Box::new(FakeInterceptor { verdict }))
            .build()
    }

    #[tokio::test]
    async fn allow_produces_no_finding() {
        let inspector = inspector_with(Verdict::allow(), EnforcementMode::Enforce);
        let results = inspector
            .inspect("s1", &[tool_request("r1", "lookup")], &[], GooseMode::Auto)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn hard_deny_blocks_tool() {
        let inspector = inspector_with(
            Verdict::deny(Some("tool_denied".into()), Some("disabled by policy".into())),
            EnforcementMode::Enforce,
        );
        let results = inspector
            .inspect(
                "s1",
                &[tool_request("r1", "delete_account")],
                &[],
                GooseMode::Auto,
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].action, InspectionAction::Deny);
        assert_eq!(results[0].tool_request_id, "r1");
        assert_eq!(results[0].inspector_name, "agent_hooks");
    }

    #[tokio::test]
    async fn escalate_requires_approval() {
        let inspector = inspector_with(
            Verdict::escalate(Some("needs_review".into()), None),
            EnforcementMode::Enforce,
        );
        let results = inspector
            .inspect(
                "s1",
                &[tool_request("r1", "wire_transfer")],
                &[],
                GooseMode::Auto,
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(matches!(
            results[0].action,
            InspectionAction::RequireApproval(_)
        ));
    }

    #[tokio::test]
    async fn transform_fails_closed() {
        let verdict = Verdict {
            decision: Decision::Transform,
            transform: Some(Transform {
                path: "$target".into(),
                value: json!({ "k": "[REDACTED]" }),
            }),
            ..Verdict::allow()
        };
        let inspector = inspector_with(verdict, EnforcementMode::Enforce);
        let results = inspector
            .inspect("s1", &[tool_request("r1", "lookup")], &[], GooseMode::Auto)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].action, InspectionAction::Deny);
    }

    #[tokio::test]
    async fn evaluate_only_never_enforces() {
        let inspector = inspector_with(
            Verdict::deny(Some("tool_denied".into()), None),
            EnforcementMode::EvaluateOnly,
        );
        let results = inspector
            .inspect(
                "s1",
                &[tool_request("r1", "delete_account")],
                &[],
                GooseMode::Auto,
            )
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn multiple_requests_are_each_governed() {
        let inspector = inspector_with(
            Verdict::deny(Some("tool_denied".into()), None),
            EnforcementMode::Enforce,
        );
        let results = inspector
            .inspect(
                "s1",
                &[tool_request("r1", "a"), tool_request("r2", "b")],
                &[],
                GooseMode::Auto,
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool_request_id, "r1");
        assert_eq!(results[1].tool_request_id, "r2");
    }
}
