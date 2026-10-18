# Architecture Decision Records

Every decision affecting architecture, invariants, the public API, security, or dependencies is
documented here with context, rejected alternatives, and consequences. Create new ADRs using the
`write-adr` skill or [`0000-template.md`](0000-template.md). The `Accepted` status is set by a human.

| No. | Decision | Status |
|---|---|---|
| [0001](0001-custom-core-over-agent-framework.md) | Custom lean core instead of an external agent framework | Accepted |
| [0002](0002-canonical-message-format.md) | Canonical message format with provider adapters | Accepted |
| [0003](0003-protocol-invariants-in-runstate.md) | Enforce protocol invariants in `RunState` | Accepted |
| [0004](0004-hooks-fail-closed.md) | Typed hooks with per-stage allowed decisions, fail-closed | Accepted |
| [0005](0005-pause-and-resume-over-blocking-approval.md) | Pause and resume instead of blocking approval gateways | Accepted |
| [0006](0006-optimistic-locking-and-write-ahead.md) | Optimistic locking, claiming on resume, and write-ahead checkpoints | Accepted |
| [0007](0007-sequential-tool-calls.md) | Execute tool calls within a response sequentially | Accepted |
| [0008](0008-no-placeholder-modules.md) | No placeholder modules | Accepted |
| [0009](0009-ai-assisted-development.md) | AI-assisted development: `AGENTS.md`, skills, independent reviewer | Accepted |
| [0010](0010-checkpoint-schema-versioning.md) | Versioned checkpoints with forward migrations | Accepted |
| [0011](0011-operational-limits-and-metadata-events.md) | Metadata-only events, context and input budgets, kill switch | Accepted |
| [0012](0012-format-v2-and-runtime-dependencies.md) | Format v2, runtime dependencies (`deps`), and pre-adapter hardenings | Accepted |
| [0013](0013-package-distribution.md) | Distribution as a versioned package instead of a copied template | Accepted |
| [0014](0014-bind-resume-to-pending-call.md) | Bind `resume` to the pending call (`pending_id`) | Accepted |
| [0015](0015-safe-defaults-for-read-file-and-file-store.md) | Default deny list for `ReadFile`; file store is POSIX-only | Accepted |
| [0016](0016-import-name-ai-agent-harness.md) | Import name `ai_agent_harness` instead of `harness` | Accepted |
| [0017](0017-anthropic-provider-adapter.md) | Anthropic provider adapter as an optional extra | Proposed |
| [0018](0018-postgres-checkpoint-store.md) | PostgreSQL checkpoint store as an optional extra | Proposed |
