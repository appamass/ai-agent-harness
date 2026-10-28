# Changelog

Semantic versioning. Changes to the persisted checkpoint format also list their schema version
(ADR 0010). Prior to `1.0.0`, minor versions may introduce breaking changes to the public API.

## Unreleased

- Changed (breaking, ADR 0032): Python 3.12 is the minimum; 3.11 is no longer supported. CI tests 3.12 to 3.14, and the `lowest` job runs on 3.12. `BaseTool` and `SyncTool` use the PEP 695 syntax (`BaseTool[InputT: BaseModel]`); the unexported module-level `ai_agent_harness.tools.base.InputT` is gone, and type checkers infer the parameter as contravariant. Runtime behavior and tool digests are unchanged.
- New (sub-agents, ADR 0030): `sub_agent(specialist, *, name=None, description, timeout_s=None)` wraps a `read_only` agent as a `Risk.READ` tool. It refuses specialists that can write or ask the user, reports every status other than `done` as a tool error, derives a deterministic sub-run ID from the caller's tenant, agent, run, creation time, step, and batch position, and refuses cycles (also back to the top-level agent) and chains deeper than `MAX_SUB_AGENT_DEPTH` (4). Guarantee 21.
- New (cancel, ADR 0031): `cancel(agent, state, *, recover=False, reason=...)` ends a waiting run, or with `recover=True` a run a crash left `running`, as `cancelled` without a model call or tool. Finished batch results stay; other open calls are answered "not executed" or "outcome unknown". The sub-agent tool cancels a sub-run that would wait. Guarantee 22.
- New: `ModelRouter.retry` and `ModelRouter.worst_case_s()` (the longest time one model call can take with all retries and fallbacks).
- Docs: `docs/agent-package.md` shows how to lay out an agent project on the harness; `docs/guide.md` has a sub-agent section.

## 0.8.0 – 2026-10-23

- New (token estimate, ADR 0026): `Limits.max_context_tokens` (optional, last field; `Limits()` is unchanged) limits the estimated prompt tokens. The estimate starts from the prompt tokens the provider reported (`Usage.prompt_tokens`) and adds new characters at `CHARS_PER_TOKEN` (3). `MODEL_REQUEST` events carry `context_chars` and `context_tokens_est`. Each message is measured once per step (`context.ContextMeter`).
- Changed (checkpoint format, schema v5, ADR 0026): `RunState.context_floor` and token measurement fields (`measured_provider_id`, `measured_chars`, `measured_prompt_tokens`). Migration 4 → 5 sets default values; v4 checkpoints load unchanged.
- New (deterministic context compaction, ADR 0026): Opt-in `Agent.compact_context: bool = False`. When approaching context limits, clears older `ToolResult` contents in the model request view with a placeholder `[Result removed to save space (N chars). Call the tool again if you need it.]`, preserving message alternating roles and protocol. History in checkpoints is preserved. Emits `EventType.CONTEXT_COMPACTED` events.
- New (parallel read tools, ADR 0027): Opt-in concurrent execution of consecutive `Risk.READ` tools marked `parallel_safe=True` via `Agent.max_parallel_tools > 1`. All calls still execute exclusively through the `ToolExecutor` gate under an `asyncio.Semaphore`. Persists write-ahead checkpoints before parallel groups, preserves call order in messages and `batch_done`, preserves mid-group approval binding, and ensures crash honesty.
- Changed (guard precision & opt-in detection, ADR 0028): `SecretGuard` tightened `sk-` prefix matching (eliminating kebab-case false positives), added provider token formats (SendGrid, Twilio, Slack webhooks, Azure connection strings), opt-in `assignments=True` (detects credential assignments with entropy >= 2.5; camelCase names such as `dbPassword` and JSON keys count), and opt-in `decode=True` (budget-bounded single-step URL, Base64 with or without `=` padding and after a path, and Hex decoding; only readable stretches are scanned). `EgressGuard` builds its URL pattern and markers from one table, restricted single-slash parsing to WHATWG special schemes, matches every scheme in any case (`HTTPS:/host` and `MAILTO:` are blocked; `note:/path` and `path:/usr/bin` pass), Pydantic `HttpUrl`/`AnyUrl` detection, and `ignore_keys` parameter. `WeakeningGuard` added modern test/lint directives.
- New (checkpoint encryption & retention purge, ADR 0029): Optional envelope encryption via `ai_agent_harness.encryption.AesGcmCodec` (`pip install "ai-agent-harness[encryption]"`) using AES-256-GCM with authenticated metadata header (`enc:v1:` prefix, AAD over header, fail-closed). All stores accept `codec=`. Encryption runs off the event loop via `asyncio.to_thread`. New `purge_finished(*, older_than: timedelta, tenant_id: str | None = None) -> int` on `SqliteCheckpointStore` and `PostgresCheckpointStore`, safely purging only completed terminal runs evaluated against the database clock.
- Changed (dependencies): the extras now allow the newest majors: `anthropic>=1.15,<3`, `openai>=1.51,<4` (the lower bound rose from 1.40: older versions have no `prompt_tokens_details`, so cached tokens were counted as input), `cryptography>=43,<51`. `uv.lock` holds anthropic 2.1.0, openai 3.37.0, and cryptography 50.0.2; the SDKs use `httpx2` instead of `httpx`. Python 3.14 is tested in CI (classifier added). Anthropic 2 sends the key as `Authorization: Bearer`; the adapter passes the key to the SDK as before.
- Changed (checkpoint format, schema v4): `RunState.output_schema` is replaced by `output_schema_digest` (the `canonical_digest` of the output model's JSON schema). Migration 3 → 4 computes it; v3 checkpoints load unchanged.
- Changed (before release): `StreamDelta` carries answer text only (reasoning arrives as a `ThinkingBlock` in the final response); `BaseTool.digest()` hashes `spec()`, so `ToolRegistry` pins of MCP tools cover the server's schema; `FakeProvider` scripts a streamed call as `Streamed(...)`; `MCPServerConfig.untrusted_output` (MCP output is always untrusted) and `RunState.output_object` (use `output_as`) are removed; `CallableProvider` streams through the router's `complete()` fallback; `FakeProvider` and `Streamed` raise `TypeError` for an unexpected script item; the SQLite store refuses a borrowed connection that has an open transaction.
- Fixed: NaN and infinite values were accepted as tool, stream, and MCP timeouts. `OpenAIProvider` checks `max_tokens_param` before it builds a client.
- Internal: one tool-class factory (`make_tool`) for `@tool`, MCP, and `final_output`; `RunState.standalone()` / `record_result()`, `ToolRegistry.of` / `HookPipeline.of`; the SQL stores share their statements in `persistence_sql.py` (`DEFAULT_TABLE`, `sql_identifier`, `status_index_name`, and `WriteRecord` moved there); the SQLite store applies its PRAGMAs on first use (a borrowed connection at construction); JSON schemas are built once per model class.

- Changed (behavior): tool results above `max_result_chars` now keep head **and** tail (`truncation_mode="head_and_tail"`) instead of only the head. Pass `truncation_mode="head"` for the old behavior. A tool's own `max_result_chars` can only lower the agent's limit.
- Changed: `ModelRouter.providers` is back (the runner reads it to choose the structured-output tier); `RunState.set_output` now takes only the validated model.
- Fixed before release (test-quality review): a stream read cancelled at the moment an item arrived could lose that item (one persistent getter now); the stdio MCP transport left its stderr pipe open after `aclose()` (readers now end by themselves before they are cancelled); OpenAI error kinds were never read from the error body; `OpenAIProvider` tool-call fragments and the SDK timeout, client setup, and `aclose` are covered. The eval gate now judges answers (right tool, forbidden tools, schema-valid arguments, budget) and has negative controls; a v3 checkpoint fixture exists; tests run with a 60 s limit (`pytest-timeout`) and declare `anyio`.
- Fixed before release (strict recheck): a stream closed before its task started no longer overwrites a pending batch or an interrupted call with "Not executed" (`stream_resume` leaves the state and checkpoint untouched); a cancelled read no longer loses the next stream item; MCP tools with required array, object, `anyOf`, or enum parameters load, the model sees the server's real schema, and tool names with dots are mapped; a failed run setup leaves the caller's state unchanged; the provider deadline pauses while a slow consumer holds a delta; deltas without text no longer block retry; tool-call fragments without an index merge into one call; `final_output` accepts free-form output models.
- Fixed before release (review): `final_output` runs through the `ToolExecutor` gate (policy, hooks, events) and is a reserved tool name; without `output_model` it cannot end a run. Resuming a run with structured output requires the same `output_model` (`InvalidStateError` otherwise); output is never validated against itself. The structured-output tier is chosen from all providers of the router.
- Fixed before release (review): stream handles no longer hang on a full queue when closed or cancelled, and `idle_timeout_s` cancels runs whose consumer stopped reading. `ModelRouter.stream` shares the retry loop with `complete` and enforces `request_timeout_s`.
- Fixed before release (review): MCP stdio accepts messages up to 16 MiB, fails pending requests when the server exits or breaks the protocol, ignores spoofed ids, drains stderr, starts servers with a minimal environment, and bounds startup with `connect_timeout_s`. Digests cover the raw server definition (`mcp_tool_digest`), `list_changed` fails closed, omitted optional arguments are not sent as `null`, and error text of untrusted tools is framed. `load_mcp_tools` returns an `MCPToolset` that can be closed.
- Fixed before release (second review): a `final_output` result withheld by an `after_tool` hook no longer ends the run as `done`; iterating a handle after `cancel()` terminates; closing a handle before its task started still persists `CANCELLED`; `await handle` drains unread items; a slow consumer no longer counts against the provider timeout; error text of tools is truncated like results; OpenAI streams without `finish_reason` fail retryably; MCP `tools/list` follows pagination and server-request replies cannot stall the reader.
- Fixed before release (review): `OpenAIProvider(structured_output=..., max_tokens_param=...)` and `reasoning_content` in streams; OpenTelemetry metrics are implemented, non-standard attributes use the `harness.` prefix, open runs are bounded; the client version comes from the installed package.

- New (eval suite): 20 deterministic regression eval cases as a CI release gate in `tests/test_evals.py` using the single-step replay pattern. Evaluates candidate models at recorded production decision points against concrete invariants (tool selection, forbidden tool denial, schema argument conformance, token budgets, and prompt injection immunity) with zero tool execution and zero state mutation.
- New (pilot project): Complete runnable incident responder in `examples/incident_responder.py` as a pilot project. Features Stage 1 read-only investigation (`query_logs`, `get_metrics`, `recent_deploys`) and Stage 2 remediation with mandatory human approval (`rollback_deploy`, `Risk.IRREVERSIBLE`), runtime dependencies (`DeployClient`) via `deps`, emergency freeze via `KillSwitch`, and cross-platform checkpoints with `SqliteCheckpointStore`. Includes an offline `--demo` mode verified in `tests/test_examples.py`.
- New (streaming events and text deltas, ADR 0025): Asynchronous streaming via `stream_start` and `stream_resume` returning a `StreamHandle` yielding `Event | TextDelta`. Supports live token display, lifecycle events, and bounded-queue backpressure (`queue_size`). User cancellation or early loop exit safely terminates execution, guarantees `RunStatus.CANCELLED` write-ahead persistence, and ensures partial stream tokens never enter `RunState.messages` (preserving Invariant 2 protocol validity). Native streaming implemented across `FakeProvider`, `AnthropicProvider`, and `OpenAIProvider`; other providers answer as one delta.
- New (MCP integration, ADR 0024): Model Context Protocol (MCP) tool integration in `ai_agent_harness.tools.mcp` governed by a strict, fail-closed trust model. Discovered MCP tools are materialized as genuine `BaseTool` subclasses executing exclusively through the 8-step gate `ToolExecutor`. Enforces mandatory local risk classification (`MCPServerConfig.tool_risks`; unclassified tools raise `ConfigError` at startup), rug-pull prevention via SHA-256 digest of the raw server definition (`mcp_tool_digest`) against pinned `tool_digests`, opt-in server allowlist validation, and untrusted output framing (always on). Provides `StdioMCPTransport` (subprocess JSON-RPC 2.0) and in-memory `InMemoryMCPTransport` / `MockMCPServer` in the standard library (zero external dependencies).
- New (structured output, ADR 0023): Structured final output (`output_model`) support on `start(..., output_model=...)` and `resume(..., output_model=...)`. Validated outputs are exposed typed on `state.output_as(MyModel)` and serialized as JSON dictionary on `state.output`. Degrades across 3 capability tiers: native provider JSON schema (`Capabilities.structured_output=True`, OpenAI), an instructed `final_output` tool (`Capabilities.tools=True`, Anthropic), and a prompt instruction with JSON text extraction. Schema validation errors trigger diagnostic rejection feedback (`_reject`) for model self-correction up to `max_consecutive_retries`. Increments checkpoint format to `SCHEMA_VERSION=3` with an automatic migration from v2 (now v4, see above).
- New (observability, ADR 0022): `OpenTelemetryListener` in `ai_agent_harness.observability` for emitting OpenTelemetry spans and the GenAI client metrics `gen_ai.client.token.usage` and `gen_ai.client.operation.duration` (`pip install "ai-agent-harness[opentelemetry]"`). Maps run workflow spans, client model spans with usage attributes (`gen_ai.usage.*`), tool execution spans, and provider retry/failover events with strictly zero prompt, argument, or secret leakage.
- New (tool output truncation, ADR 0022): Smart `head_and_tail` truncation on `ToolExecutor` (retaining 60% head and 40% tail separated by an omission notice), ensuring stack traces, exit codes, and test failure summaries at the end of tool outputs are never lost. Configurable via `truncation_mode` and `max_result_chars` on `ToolExecutor`, `Agent`, and per-tool on `BaseTool` and `@tool(...)`.
- New (provider adapter, ADR 0021): `ai_agent_harness.providers.openai.OpenAIProvider` for OpenAI Chat Completions API and compatible servers (Ollama, vLLM, OpenRouter, and Gemini-compatible endpoints) as an optional extra: `pip install "ai-agent-harness[openai]"`. Supports tools, vision, prompt caching token accounting, thinking content extraction, error classification without prompt leaks, and mid-run failover with `without_foreign_thinking`.
- New (persistence, ADR 0020): `ai_agent_harness.persistence_sqlite.SqliteCheckpointStore` built entirely on standard library `sqlite3` (zero extra dependencies). Cross-platform support (Linux, macOS, Windows) solving the Windows `fcntl` limitation of `JsonFileCheckpointStore`. Optimistic concurrency via atomic conditional upsert (`INSERT ... ON CONFLICT DO UPDATE ... WHERE revision = ? RETURNING revision`), WAL mode, busy timeout, and operations helpers (`list_runs`, `delete_run`, `delete_tenant`).
- New (AI kit & skills): Standalone `ai-kit/` containing skills, an invariants guide, and recipes for putting the gate, guardrails, and crash recovery into other frameworks (`adopt-in-x.md`, the gate recipe runs in CI).
- Positioning: Redesigned `README.md` and `AGENTS.md` around the three adoption paths: Use (dependency), Copy (standalone bricks), and Rebuild (invariants in your stack).

## 0.7.0 – 2026-10-21

- New (modular standalone bricks, ADR 0019): `ToolExecutor`, `HookPipeline`, `ModelRouter.complete`, and all built-in guardrails can now be used standalone without constructing a `RunState` or `Agent`. `ToolExecutor` accepts a plain tool list and defaults `policy`, `hooks`, `emit`, and `state`.
- New (ergonomics): `@tool(risk=...)` decorator (`ai_agent_harness.tool`; `risk` is required, so a function is never read-only by accident) turns typed sync or async functions into validated `SyncTool` / `BaseTool` instances (with optional `state: RunState` injection hidden from the JSON schema). `CallableProvider` wraps an async `(ModelRequest) -> ModelResponse | str` function as a `ModelProvider`. `Agent.create(...)` provides a concise factory for small to complete agents.
- New (security guardrails): `SecretGuard` (blocks private keys, a list of common API tokens including Google, npm, PyPI, and live Stripe keys, AWS keys, JWTs, Bearer and Basic authorization headers, and embedded URL credentials in tool arguments, tool results, and model responses, and in user input when enabled; `find_secret` and `redact` are standalone helpers), `EgressGuard` (`allowed_domains` is required, `()` blocks all URLs; HTTPS plus an allowed-domain list for URLs in tool arguments and model responses; fields named like a URL, such as `url` or `base_url`, must hold a complete allowed URL; rejects backslashes, control characters, odd host characters, invalid ports, IP literals, `localhost`, `.local`/`.internal`; block reasons do not repeat the host), and `WeakeningGuard` (blocks check-suppression directives such as `# type: ignore`, `# noqa`, `# ruff: noqa`, `# pylint: disable`, `# nosec`, `pytest.mark.skipif`, `unittest.skip`, and `sys.exit(0)` on write tools). All three are pattern checks with known gaps, see `docs/known-limitations.md`.
- New (tool integrity & untrusted data): `BaseTool.digest()` (full SHA-256 over name, description, input schema, risk, timeout, and `untrusted_output`) and `ToolRegistry(..., expected_digests=...)` pin tools; a pin without a registered tool, or a changed tool, is a `ConfigError` at registration or on `registry.verify()`. `as_untrusted(source, text)` and `BaseTool.untrusted_output = True` wrap external tool output in an `<untrusted>` frame after truncation and escape any `<untrusted` or `</untrusted` inside it (the frame can exceed `max_result_chars` by its own length). The frame is a hint to the model, not a security boundary.
- Behavior change: `ReadFile` now denies more names by default (see below). Code that read `credentials.json`, `.npmrc`, or similar through `ReadFile` must pass its own `denied=` list.
- Security: `ProtectedPaths(..., root=...)` optionally resolves symlinks against a root directory; `DEFAULT_DENIED_READS` in `ReadFile` now also blocks `.envrc`, `.ssh`, `.aws`, `*.p12`, `*.pfx`, `.netrc`, `.npmrc`, `.pypirc`, `id_rsa`, `id_ed25519`, and `credentials.json`.

## 0.6.0 – 2026-10-18

- Docs: the README is now a short start (about 145 lines): what it is, a no-key demo with a real transcript, a hello-agent block, what is included, status, and a
  documentation map. The detail moved to `docs/`: `guide.md` (build with it), `how-it-works.md` (code map, run loop, gate, autonomy), `guarantees.md` (all 16
  guarantees with the test that enforces each), `production-checklist.md`, `known-limitations.md`. Nothing was dropped.
- Tests: executable blocks in the README and the new guides run in CI; `tests/test_links.py` checks every relative link and anchor; the README demo transcript is
  compared with the real demo output; the README has a length limit.

- New: `ai_agent_harness.persistence_postgres.PostgresCheckpointStore`, a checkpoint store for several workers, as an optional extra:
  `pip install "ai-agent-harness[postgres]"` (ADR 0018). One atomic conditional upsert implements the optimistic locking, loads go
  through `restore_state`, `setup()` / `schema_sql()` create the table, `list_runs` serves approval queues and recovery jobs,
  `delete_run` / `delete_tenant` support retention and erasure. Works with a connection string or an `AsyncConnectionPool`.
- Tests: the store contract tests run against a real PostgreSQL when `HARNESS_TEST_POSTGRES_DSN` is set (CI does); one in four stress
  scenarios runs on PostgreSQL. A mutation check showed the concurrency tests fail if the revision condition is removed.
- Removed (unused, nothing read them): the `BaseTool.idempotent` class attribute and `ModelRouter.providers`. A tool that sets `idempotent` keeps working; the value is ignored.
- Internal: the cancellation-safe write logic of the file store moved to `persistence.settle_write` and is shared; `check_ids` is no longer private.

## 0.5.0 – 2026-10-18

- New: `ai_agent_harness.providers.anthropic.AnthropicProvider` for the Anthropic Messages API, as an optional extra:
  `pip install "ai-agent-harness[anthropic]"` (ADR 0017). The core and `import ai_agent_harness` still need only Pydantic.
  No SDK-level retries (the router owns them), no sampling parameters, thinking signatures round-trip, errors carry only
  status and type, prompt caching on by default.
- New: `examples/repo_assistant.py`, a complete runnable agent (tools, human approval with `pending_id`, limits, kill file,
  checkpoints and `--resume`, `--demo` without an API key), tested offline in `tests/test_examples.py` and opt-in live in
  `tests/live/test_examples_live.py`. `examples/` is checked like the library (ruff, mypy strict, pyright strict) and is not part of the wheel.
- New: opt-in live tests (`pytest -m live`, `tests/live/`); the `live` marker is registered and deselected by default.
- New: `AnthropicProvider.aclose()` closes the connections of a client the adapter created (found in the first live run:
  without it Python 3.14 reports unclosed sockets). Call it when you are done.
- Docs: `docs/first-real-run.md` has a 5-minute Anthropic setup and the first results: the 7 live tests passed against
  Claude Haiku 5.5 on 2026-10-18. Eighteen real responses are recorded as test fixtures
  (`tests/fixtures/anthropic/recorded/`: plain answers, tool calls, thinking together with a tool call, prompt-cache usage, a truncation run with thinking blocks). Fixtures for redacted
  thinking and refusal are still hand-written.
  Only Anthropic was run live (Claude Haiku 5.5). A second live run found a bug in the test helper (it closed the client
  between steps); after the fix the third run passed 7 of 7 without warnings, and the fourth run (with the example agent) 11 of 11.

## 0.4.0 – 2026-10-18

**Breaking:** the import package is renamed from `harness` to `ai_agent_harness` (ADR 0016) — replace
`from harness import ...` with `from ai_agent_harness import ...` and logger names `harness.*` with
`ai_agent_harness.*`. `resume` of a waiting run requires `pending_id` (ADR 0014). `ReadFile` denies secret files by
default and `JsonFileCheckpointStore` refuses platforms without `fcntl` (ADR 0015). Persisted format unchanged (v2).

- Fix (security): a stale approval or answer could apply to a different call. Example: two approvers see
  the dialog for action A; the first approves, the run pauses on action B; the second approver's click
  executed B although nobody had seen it. `resume(..., pending_id=state.pending_id)` is now required for
  `waiting_for_input` and `waiting_for_approval`; a missing or different value raises `InvalidStateError`
  before any state changes. Migration: pass the `pending_id` that was shown with the dialog.
- New: `RunState.pending_id` (derived, never persisted).
- Fix (security, behavior change): `ReadFile` refuses `.env`, `.env.*`, `*.env`, `secrets`, `.git`, `*.pem`, and `*.key`
  under its root by default (`DEFAULT_DENIED_READS`). It is `Risk.READ` and allowed at `read_only`, so it could
  read secrets before. Both the requested and the resolved path are checked (names only; see README Known
  Limitations). New public constant
  `DEFAULT_DENIED_READS`. Migration: pass `denied=(...)` to change the list, `denied=()` for the old behavior.
- Fix (behavior change): `JsonFileCheckpointStore` raises `ConfigError` on platforms without `fcntl` (e.g. Windows).
  Before, it ran there without any cross-process lock, so two workers could execute one action twice.
- Fix: `Limits` rejects NaN, infinity, bools, and non-numbers (`ConfigError`). `max_seconds=nan` silently disabled
  the time limit. Types that are not `int`/`float` subclasses (e.g. `numpy.int64`) are now rejected too.
- Docs: README Known Limitations lists the `ProtectedPaths` symlink gap and the `ReadFile` deny list.
- Docs: corrected Use Case 01 — AAD does not detect replay of an older complete checkpoint row.
- Docs: project status is alpha; no live model provider run yet.
- CI: installs the exact versions from `uv.lock` (`uv sync --locked`), so new tool releases cannot break CI;
  a `lowest` job tests the lower bounds (e.g. `pydantic==2.13`); Dependabot proposes weekly updates.
- Packaging: authors and project URLs (`github.com/appamass/ai-agent-harness`), license holder `appamass`,
  classifiers `3 - Alpha`, `POSIX`, and `MacOS`. All ADRs 0001–0016 accepted; ADR 0013 decided (versioned package via Git release tags).

## 0.3.0 – 2026-10-16

**Persisted format v2** (automatic migration from v1; older versions cannot read v2).

- New: `start/resume(..., deps=...)` and `state.deps_as(Type)` for runtime dependencies (never persisted).
- New: `Usage.cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`; `Usage.total` includes them.
- New: `ThinkingBlock.provider_id` and `without_foreign_thinking()` for cross-provider failover.
- New: `ToolCall.arguments_error` — malformed JSON is returned to the model for self-correction instead of terminating the run.
- New: `KillSwitch`, `Limits.max_context_chars`, `Limits.max_input_chars`, `restore_state()` with key verification.
- Fix: After a crash mid-tool-batch, completed tools retain their results (write-ahead before every tool).
- Fix: `last_text` concatenates max-token continuations; `Message.text` no longer glues blocks together without separators.
- Fix: Hanging `SyncTool`s can no longer starve the checkpoint store (dedicated worker thread pool).
- Fix: `ProtectedPaths` detects paths under arbitrary field names and as nested dictionary keys.
- Fix: Cancellation during a checkpoint write always ends as `CancelledError` (even on repeated cancellation
  or when another worker has taken over the run) rather than `ConcurrencyError`.
- Fix: `deps` are unbound after each run and never pickled or deep-copied; overly long `arguments_error` strings are truncated rather than rejected.
- Fix: File checkpoint store no longer leaves a lock file per run and detects case-insensitive filesystem collisions.
- Fix: Events no longer include provider error messages.
- Dependency: `pydantic>=2.13`.
- Docs: Use Cases 04 (incident responder), 05 (provider switch), 06 (sub-agents & next-action eval);
  updated Use Cases 01–03 for v0.3.0 (results after crash, cost by category, `deps`), UC2 + "only done when tests pass",
  UC3 + multi-day clarifying question.

## 0.2.0

First complete release: runner with protocol invariants, hooks, tool execution gate, policy matrix,
router, checkpoints with optimistic locking and write-ahead, executable use cases, and ADRs 0001–0009.
