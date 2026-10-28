---
name: ai-agent-harness
description: "Reference implementation and engineering patterns for building crash-safe, deterministic AI agent harnesses with strict human-in-the-loop approval, fail-closed guardrails, and optimistic concurrency."
---

# AI Agent Harness Skill

This skill guides AI coding agents and human developers on how to adopt, copy, or rebuild the engineering patterns from `ai-agent-harness`.

The core philosophy of this harness is:
> **The language model proposes, the code decides.**

The language model is treated as an untrusted, probabilistic decision-making engine. Every tool call, state mutation, and guardrail evaluation is deterministic, fail-closed, and verifiable.

---

## When to Use This Skill

Activate this skill when:
- Designing or reviewing an AI agent system.
- Implementing human-in-the-loop approval that cannot be bypassed by race conditions or replay attacks.
- Preventing crashes and data loss with write-ahead checkpointing and crash recovery.
- Hardening existing agent stacks (such as **LangGraph** or **Pydantic AI**) with the same patterns.
- Preventing prompt injections, secret leakage, or unauthorized filesystem/network egress.

---

## The Three Adoption Modes

| Mode | When to Choose | How to Implement |
|---|---|---|
| **1. Use (Dependency)** | Building a greenfield Python service that needs reliable stateful agent loops. | Pin a release tag: [install line](../../../README.md#install) ([ADR 0013](../../../docs/adr/0013-package-distribution.md)). |
| **2. Copy (Modular Bricks)** | You already have an established framework but need a proven component. | Use or copy bricks (together with `core/` and `policy.py`; stores also need `persistence.py` and `persistence_sql.py`) ([ADR 0019](../../../docs/adr/0019-modular-bricks-and-security-guards.md)). E.g., `ToolExecutor`, `SecretGuard`, or `PostgresCheckpointStore`. |
| **3. Rebuild (Invariants in Your Stack)** | Using LangGraph, Pydantic AI, or an in-house engine and must satisfy strict compliance. | Follow the recipes in [Adopt in other frameworks](references/adopt-in-x.md) and respect the 9 invariants. |

---

## The 9 Non-Negotiable Invariants

Any robust agent system must uphold these invariants:

1. **Explicit Terminal Status:** Every execution ends with a defined status (`done`, `refused`, `blocked`, `limit_reached`, `failed`, `cancelled`) or pauses waiting for a human, never an unhandled exception or silent abort.
2. **Strict Protocol Compliance:** History strictly alternates (`user` ↔ `assistant`). Every tool call request has a matching tool result block.
3. **Single Tool Execution Gate:** Tools run exclusively through one centralized executor (`ToolExecutor`). No direct invocation of `tool.run` outside the gate.
4. **Approval Binding (`pending_id`):** `IRREVERSIBLE` actions never run without human approval (denied below `act_with_approval`). Approval must be bound to the specific call shown to the human (`pending_id`). Resuming cannot apply an approval to a different tool call ([ADR 0014](../../../docs/adr/0014-bind-resume-to-pending-call.md)).
5. **Write-Ahead Checkpointing & Crash Honesty:** State is persisted *before* executing each tool call. A recovered run marks interrupted in-flight calls as unknown outcome (`interrupted`), never re-executing them blindly.
6. **Fail-Closed Guardrails:** Any hook or guardrail that throws an error or returns an unexpected decision blocks execution immediately.
7. **Safe Metadata-Only Observability:** Telemetry and event streams contain metadata only (counters, IDs, tool names), never raw prompts, secrets, or tool arguments ([ADR 0011](../../../docs/adr/0011-operational-limits-and-metadata-events.md)).
8. **Dependency-Isolated Core:** The state machine and core types depend only on standard types and validation primitives, avoiding vendor lock-in.
9. **Versioned State & Forward Migrations:** Saved checkpoints store explicit schema version numbers (`schema_version`) and run forward migrations on restore ([ADR 0010](../../../docs/adr/0010-checkpoint-schema-versioning.md)).

---

## Modular Bricks Quick Reference

The harness provides self-contained building blocks ready to copy or adapt:

- **The 8-Step Tool Gate (`ai_agent_harness.tools.executor.ToolExecutor`):**
  1. Look up the tool (unknown: error to the model).
  2. Validate arguments with Pydantic.
  3. Policy: allowlist and autonomy × risk.
  4. `before_tool` hooks (guardrails) on the validated arguments.
  5. Approval, if required and not yet granted: the run pauses.
  6. Execute with a timeout; errors become safe messages.
  7. Process output: truncate, frame untrusted content (`as_untrusted(...)`).
  8. `after_tool` hooks; a block withholds the result.

- **Risk-Categorized Tools (`@tool(risk=Risk.IRREVERSIBLE)`):**
  Explicit risk tagging on functions (`Risk.READ`, `Risk.WRITE`, `Risk.IRREVERSIBLE`) to govern whether human approval is required.

- **Fail-Closed Security Guards (`ai_agent_harness.guardrails`):**
  - `SecretGuard`: Scans tool arguments and model outputs for common API tokens, private keys, and credential headers (pattern based, see known limitations).
  - `EgressGuard`: Domain/IP allowlisting for outbound HTTP requests.
  - `ProtectedPaths`: Blocks write-tool arguments that touch protected paths (tests, CI, config); symlinks are resolved only with `root=`.
  - `KillSwitch`: Halts runs when your check function says so (a kill file, a flag, a database row).

- **Sub-Agents as Read Tools (`ai_agent_harness.sub_agent`):**
  A specialist runs at `read_only` without `ask_user`, under the caller's tenant, with a deterministic sub-run ID; every status other than `done` comes back as a tool error, and cycles (also back to the top-level agent) are refused.

- **Postgres Optimistic Concurrency (`PostgresCheckpointStore`):**
  Atomic conditional upsert (`INSERT ... ON CONFLICT DO UPDATE ... WHERE revision = :expected RETURNING revision`) preventing concurrent workers from racing or double-executing actions. `SqliteCheckpointStore` uses the same statement.

---

## Reference Guides

- [Recipes for other frameworks](references/adopt-in-x.md): the gate as your tool node (tested), fail-closed guardrails, and crash recovery.
- [Agent project layout](../../../docs/agent-package.md): prose in files, every hard rule in code, runtime data outside the package.
