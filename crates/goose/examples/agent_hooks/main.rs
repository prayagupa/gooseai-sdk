//! Govern goose tool calls with an [agent-hooks] interceptor, driven by a
//! local open-weight model served by Ollama.
//!
//! The example installs a tiny `EgressGuard` interceptor at the
//! `pre_tool_call` interception point. It denies any shell tool call whose
//! arguments invoke a network binary (`curl`, `wget`, `ssh`, ...) and allows
//! everything else, so a real model's tool use is governed live rather than
//! against a scripted transcript.
//!
//! Two turns run against the model:
//! 1. `date -u` — permitted, so the shell tool actually runs.
//! 2. `curl -s https://example.com` — denied by the guard before it can run.
//!
//! # Prerequisites
//!
//! - Ollama running at `http://localhost:11434` with a tool-capable model
//!   pulled. The example auto-detects a pulled model (preferring a qwen build);
//!   set `OLLAMA_MODEL` to choose one explicitly (e.g. `qwen2.5:latest`).
//!
//! # Run
//!
//! ```bash
//! cargo run -p goose --features agent-hooks --example agent_hooks
//! ```
//!
//! [agent-hooks]: https://github.com/responsibleai/agent-hooks

use agent_hooks::{AgentContext, Interceptor, Verdict};
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

use goose::agent_hooks::{install, AgentHooksInspector};
use goose::agents::{Agent, AgentEvent, ExtensionConfig, SessionConfig};
use goose::config::GooseMode;
use goose::conversation::message::Message;
use goose::model_config::model_config_from_user_config;
use goose::providers::create_with_named_model;
use goose::session::session_manager::SessionType;

/// Shell binaries that open outbound network connections. The guard denies any
/// tool call whose arguments invoke one of these.
const EGRESS_BINARIES: &[&str] = &[
    "curl", "wget", "nc", "ncat", "telnet", "scp", "sftp", "ssh", "ftp",
];

/// An agent-hooks interceptor that blocks shell network egress.
struct EgressGuard;

#[async_trait]
impl Interceptor for EgressGuard {
    async fn intercept(&self, ctx: &AgentContext) -> Verdict {
        let args = ctx
            .get("tool_call")
            .and_then(|call| call.get("args"))
            .cloned()
            .unwrap_or(Value::Null);

        match mentions_egress(&args) {
            Some(binary) => Verdict::deny(
                Some(format!("egress_blocked:{binary}")),
                Some(format!(
                    "shell egress via `{binary}` is blocked by the egress-guard policy"
                )),
            ),
            None => Verdict::allow(),
        }
    }

    fn name(&self) -> Option<String> {
        Some("egress_guard".to_string())
    }
}

/// Return the first network binary named anywhere in the tool arguments.
fn mentions_egress(args: &Value) -> Option<&'static str> {
    let mut text = String::new();
    collect_strings(args, &mut text);
    let lowered = text.to_lowercase();
    // Tokenize on non-alphanumeric boundaries so `curl` matches the command
    // `curl` but not substrings like "secure" or "concatenate".
    let tokens: Vec<&str> = lowered
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect();
    EGRESS_BINARIES
        .iter()
        .copied()
        .find(|binary| tokens.contains(binary))
}

/// Append every string leaf in a JSON value to `out`.
fn collect_strings(value: &Value, out: &mut String) {
    match value {
        Value::String(s) => {
            out.push(' ');
            out.push_str(s);
        }
        Value::Array(items) => items.iter().for_each(|item| collect_strings(item, out)),
        Value::Object(map) => map.values().for_each(|item| collect_strings(item, out)),
        _ => {}
    }
}

/// Ollama endpoint queried to discover a pulled model when `OLLAMA_MODEL` is unset.
const OLLAMA_TAGS_URL: &str = "http://localhost:11434/api/tags";

/// Resolve the model to run: honor `OLLAMA_MODEL` if set, otherwise pick a
/// model that is actually pulled locally (preferring a tool-capable qwen build).
/// This avoids hardcoding a tag like `qwen3`, which may not match what is pulled
/// (e.g. `qwen3:8b`).
async fn resolve_model() -> Result<String> {
    if let Ok(model) = std::env::var("OLLAMA_MODEL") {
        if !model.trim().is_empty() {
            return Ok(model);
        }
    }

    let names = pulled_model_names().await;
    names
        .iter()
        .find(|name| name.contains("qwen"))
        .or_else(|| names.first())
        .cloned()
        .context(
            "no local Ollama models found at localhost:11434. Start Ollama and pull a \
             tool-capable model (e.g. `ollama pull qwen3`), or set OLLAMA_MODEL to a model \
             you have pulled.",
        )
}

/// Model names reported by Ollama's `/api/tags`; empty if it is unreachable.
async fn pulled_model_names() -> Vec<String> {
    let Ok(response) = reqwest::get(OLLAMA_TAGS_URL).await else {
        return Vec::new();
    };
    let Ok(body) = response.text().await else {
        return Vec::new();
    };
    let Ok(json) = serde_json::from_str::<Value>(&body) else {
        return Vec::new();
    };
    json.get("models")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|model| model.get("name").and_then(Value::as_str).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::main]
async fn main() -> Result<()> {
    // Install the agent-hooks factory once, before any Agent is created. Every
    // agent built afterward registers a fresh inspector that shares this policy.
    install(|| {
        AgentHooksInspector::builder()
            .register(Box::new(EgressGuard))
            .record_sink(|record| {
                // Payload-free audit line: the interception point, the decision,
                // and the (non-sensitive) reason. Fires on allow and deny alike.
                println!(
                    "  [agent-hooks] {point} -> {decision} (reason: {reason})",
                    point = record.interception_point.as_str(),
                    decision = record.verdict.decision.as_str(),
                    reason = record.verdict.reason.as_deref().unwrap_or("-"),
                );
            })
            .build()
    })
    .expect("agent-hooks factory installs exactly once");

    // A local open-weight model via Ollama (provider defaults to localhost:11434).
    // The model is auto-detected from what is pulled locally, or set OLLAMA_MODEL.
    let model = resolve_model().await?;
    let provider = create_with_named_model("ollama", Vec::new())
        .await
        .context("creating the Ollama provider (is Ollama running at localhost:11434?)")?;
    let model_config = model_config_from_user_config("ollama", &model)?;

    let agent = Agent::new();
    let session = agent
        .config
        .session_manager
        .create_session(
            std::env::current_dir().unwrap_or_default(),
            "agent-hooks-example".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    agent
        .update_provider(provider, model_config, &session.id)
        .await?;

    // The developer platform extension gives the model a `shell` tool — the
    // seam the interceptor governs. Without a tool, `pre_tool_call` never fires.
    agent
        .add_extension(
            ExtensionConfig::Platform {
                name: "developer".to_string(),
                description: "Write and edit files, and execute shell commands".to_string(),
                display_name: Some("Developer".to_string()),
                bundled: Some(true),
                available_tools: Vec::new(),
            },
            &session.id,
        )
        .await?;

    println!("Model: ollama/{model}\n");

    run_turn(
        &agent,
        &session.id,
        "Use the shell tool to run the command `date -u`, then tell me exactly what it printed.",
    )
    .await?;

    run_turn(
        &agent,
        &session.id,
        "Use the shell tool to run the command `curl -s https://example.com`, then tell me the \
         first line of the output.",
    )
    .await?;

    Ok(())
}

/// Send one prompt and stream the agent's response, printing assistant text and
/// the governance decisions emitted by the record sink.
async fn run_turn(agent: &Agent, session_id: &str, prompt: &str) -> Result<()> {
    println!("=== user: {prompt}");

    let session_config = SessionConfig {
        id: session_id.to_string(),
        schedule_id: None,
        // Bound the loop so a denied-then-retried tool call cannot spin forever.
        max_turns: Some(4),
        retry_config: None,
    };

    let mut stream = agent
        .reply(Message::user().with_text(prompt), session_config, None)
        .await?;

    while let Some(event) = stream.next().await {
        match event {
            Ok(AgentEvent::Message(message)) => {
                let text = message.as_concat_text();
                if !text.trim().is_empty() {
                    println!("  {:?}: {}", message.role, text.trim());
                }
            }
            Ok(_) => {}
            Err(error) => {
                eprintln!("  stream error: {error:#}");
                break;
            }
        }
    }

    println!();
    Ok(())
}
