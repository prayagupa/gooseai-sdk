# agent-hooks interceptor + a local model

This example governs a real model's tool calls with an [agent-hooks] interceptor.
goose hosts agent-hooks as a tool inspector at the **`pre_tool_call`** interception
point (see [`goose::agent_hooks`](../../src/agent_hooks.rs)), so any interceptor can
allow, deny, or require approval for a tool call before it runs.

The interceptor here — `EgressGuard` — denies any shell command that invokes a
network binary (`curl`, `wget`, `ssh`, ...) and allows everything else. The model
runs locally through Ollama, so the tool calls being governed come from a real
open-weight model rather than a scripted transcript.

## What it does

Two turns run against the model:

| Turn | Prompt asks the model to run | Guard verdict | Result |
| ---- | ---------------------------- | ------------- | ------ |
| 1    | `date -u`                    | `allow`       | shell runs, model reports the date |
| 2    | `curl -s https://example.com`| `deny`        | tool call is blocked before it runs |

Each governed tool call also prints a payload-free audit line from the emitter's
record sink, for example:

```
  [agent-hooks] pre_tool_call -> deny (reason: egress_blocked:curl)
```

## Prerequisites

- [Ollama](https://ollama.com) running at `http://localhost:11434`.
- A tool-capable model pulled, e.g.:

  ```bash
  ollama pull qwen3      # or: ollama pull qwen2.5
  ```

## Run

```bash
cargo run -p goose --features agent-hooks --example agent_hooks
```

The example auto-detects a pulled model (preferring a qwen build). Choose one
explicitly with `OLLAMA_MODEL` (use the exact tag reported by `ollama list`):

```bash
OLLAMA_MODEL=qwen2.5:latest cargo run -p goose --features agent-hooks --example agent_hooks
```

## Verdict mapping

goose's tool-inspection seam decides allow / deny / require-approval, so the
bridge maps agent-hooks verdicts as follows:

| agent-hooks verdict          | goose action        |
| ---------------------------- | ------------------- |
| `allow`                      | run the tool call   |
| `deny`                       | block               |
| `escalate` (deny + approval) | ask for approval    |
| `transform`                  | block (fail closed) |

A `pre_tool_call` `transform` cannot be applied through this seam and is treated
as a fail-closed deny. See [`goose::agent_hooks`](../../src/agent_hooks.rs) for the
full trust model — agent-hooks is a **cooperative control contract, not a security
boundary**.

[agent-hooks]: https://github.com/responsibleai/agent-hooks
