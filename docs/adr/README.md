# Architecture Decision Records

Short records of decisions that affect architecture, invariants, the public API, security, or dependencies: context, decision, what was rejected, consequences. New ones follow [`0000-template.md`](0000-template.md) (or use the `write-adr` skill). A human sets `Accepted`.

| No. | Decision | Status |
|---|---|---|
| [0001](0001-custom-core-over-agent-framework.md) | Own core instead of an agent framework | Accepted |
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
| [0017](0017-anthropic-provider-adapter.md) | Anthropic provider adapter as an optional extra | Accepted |
| [0018](0018-postgres-checkpoint-store.md) | PostgreSQL checkpoint store as an optional extra | Accepted |
| [0019](0019-modular-bricks-and-security-guards.md) | Standalone modular bricks, ergonomic shorthands, and built-in security guards | Accepted |
| [0020](0020-sqlite-checkpoint-store.md) | SQLite checkpoint store for local, embedded, and Windows environments | Accepted |
| [0021](0021-openai-provider-adapter.md) | OpenAI-compatible provider adapter as an optional extra | Accepted |
| [0022](0022-opentelemetry-and-smart-truncation.md) | OpenTelemetry event listener and smart tool truncation | Accepted |
| [0023](0023-structured-final-output.md) | Structured final output (`output_model`) with 3-tier capability degradation | Accepted |
| [0024](0024-mcp-integration-and-trust-model.md) | MCP integration with mandatory risk classification and rug-pull prevention | Accepted |
| [0025](0025-streaming-events-and-text-deltas.md) | Streaming events and text deltas with backpressure and cancellation safety | Accepted |
| [0026](0026-token-estimate-and-context-compaction.md) | Token estimate calibrated by the provider; context compaction by clearing old tool results | Accepted |
| [0027](0027-parallel-read-tools.md) | Optional parallel execution of read-only tools | Accepted |
| [0028](0028-guard-precision-and-opt-in-detection.md) | Guard precision, single-pass matching, and opt-in detection | Accepted |
| [0029](0029-checkpoint-encryption-and-retention.md) | Optional checkpoint encryption and retention purge | Accepted |
