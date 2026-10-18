# Next Steps

Only the essentials: what this harness needs to move from a verified foundation to a live production
system, plus the recommended adoption sequence when starting your first project. Non-essential ideas
are listed under "Deliberately Deferred" at the bottom.

**Ground Rule for Every Step:** Keep the core lean. New integrations belong in separate modules with
**optional** dependencies (`pip install "ai-agent-harness[anthropic]"`), dedicated tests covering
bypass and failure modes, an ADR, `mypy --strict`, `pyright`, and green CI. No placeholder modules (ADR 0008).

| Step | Topic | Effort | Depends On |
|---|---|---|---|
| 0 | Verify the foundation & pin a release tag | Small | – |
| 1 | First real model provider adapter | Medium | 0 |
| 1a | First real run following [`first-real-run.md`](first-real-run.md) | Small | 1 |
| 2 | First pilot project: Incident Responder ([Use Case 04](use-cases/04-incident-responder.md)) | Medium | 1a |
| 3 | Database checkpoint store (`PostgresCheckpointStore`) | Medium | 0 |
| 4 | First 20 eval cases as a CI release gate | Small | 1a |
| 5 | Structured final output (`output_model`) | Medium | 1 |
| 6 | MCP integration with mandatory risk classification | Medium | 1 |
| 7 | Streaming events and text deltas | Medium | 1 |

---

## 0. Verify the Foundation & Pin a Release Tag

**Why:** Establish a clean, reproducible baseline before connecting external providers or starting
downstream projects.

- [ ] Locally: `pip install -e ".[dev]"`, then `ruff check ai_agent_harness examples tests`, `mypy ai_agent_harness examples tests`, `pyright`, `pytest`
- [ ] Push repository and verify CI is green across Python 3.11, 3.12, and 3.13
- [x] **Decide on package distribution** (ADR 0013 — decided: versioned package on PyPI): copied template vs. versioned package (`ai-agent-harness`) + project scaffold
- [x] **Create and commit a lockfile** (`uv.lock`; CI installs it with `uv sync --locked`, Dependabot proposes updates)
- [ ] If downstream projects require an older Pydantic 2.x minor version, test that version in CI before lowering the floor in `pyproject.toml` (currently tested: `>=2.13,<3`)
- [x] Review ADRs 0001–0016 and set approved decisions to `Accepted`
- [ ] Tag the baseline commit (`v0.4.0`) so downstream projects can pin a known-good version

**Done when:** All checks are green and the baseline tag is pushed.

---

## 1. First Real Model Provider Adapter

**Why:** This is the biggest unknown in any agent system. Only a real adapter proves how the canonical
message format (thinking blocks, tool calls, stop reasons, error taxonomy) behaves against live APIs.

**Scope:**
- `ai_agent_harness/providers/<provider>.py` following the `add-provider-adapter` skill, with the vendor SDK as an optional dependency
- Request/response translation, passing own thinking blocks through unchanged, sending images only when `Capabilities.vision` is true
- Error classification: HTTP 429 / 5xx / overload / transient network → `retryable=True`, all others `False`
- Full contract from `providers/base.py`: strip foreign thinking blocks via `without_foreign_thinking`, populate `ToolCall.arguments_error` on malformed JSON, populate `Usage` cache/reasoning fields, and set cache breakpoints on the stable prefix
- Add a second provider adapter right after to verify that the canonical format is truly provider-neutral

**Done when:**
- [x] Unit tests with wire-format API responses through the real SDK (`tests/test_anthropic_provider.py`): text, tool calls, thinking, `MAX_TOKENS`, refusal, every error class. *Hand-written from the documented shape until the first live run records real ones.*
- [x] Message histories produced by the adapter pass `RunState.check_protocol()`
- [ ] Mid-run failover test: history containing thinking blocks from Provider A is cleanly accepted by Provider B
- [x] A separately marked live test (`pytest -m live`, excluded from default CI) exists (`tests/live/`) — **ran green against Claude Haiku 5.5 on 2026-10-18 (7 tests)**
- [ ] [Use Case 01](use-cases/01-finance-and-communications-assistant.md) executed once with a real model instead of `FakeProvider`
- [ ] First real run completed following [`first-real-run.md`](first-real-run.md) Stage A with zero open findings
- [x] ADR documenting provider choice and handling of provider-specific features (ADR 0017)

---

## 2. First Pilot Project: Incident Responder

**Why:** Before building agents for external customers or financial transactions, validate the harness
internally with technical operators in the loop.

**Scope:**
- Implement [Use Case 04 (`04-incident-responder.md`)](use-cases/04-incident-responder.md) in its own repository
- Start strictly in Stage 1 (`Autonomy.READ_ONLY`): read logs, metrics, and recent deployments; post root-cause hypotheses
- Only after 30+ clean runs, enable Stage 2 (`Autonomy.ACT_WITH_APPROVAL`) for rollback/restart tools with `JsonFileCheckpointStore` (single host) and `KillSwitch` wired to your change-freeze calendar

**Done when:**
- [ ] Stage 1 runs reliably in production on real alerts without hallucinated tools or leaked credentials
- [ ] Approval flow (`waiting_for_approval` → `resume(..., approved=True, pending_id=...)`) verified end-to-end across processes

---

## 3. Database Checkpoint Store (`PostgresCheckpointStore`)

**Why:** Mandatory as soon as two or more worker processes or containers can resume the same run.
`JsonFileCheckpointStore` is designed for a single host and stores plaintext JSON.

**Scope:**
- `PostgresCheckpointStore` implementing the exact same optimistic locking protocol:
  `UPDATE checkpoints SET state_json = :raw, revision = :new_rev WHERE tenant_id = :t AND run_id = :r AND revision = :expected`
- Composite primary key `(tenant_id, run_id)` and an index on `(tenant_id, status)` for approval queues and recovery jobs
- Optional envelope encryption: harden `EncryptedCheckpointStore` from [Use Case 01](use-cases/01-finance-and-communications-assistant.md) with per-tenant KMS keys and crypto-shredding
- Query helper to list all runs in a given status (e.g., `waiting_for_approval`)

**Done when:**
- [x] The full persistence test suite runs against Postgres (shared contract tests, stress test, CI service container)
- [x] Concurrent worker collision, mid-write crash, and tampered row tests pass
- [x] Loading runs exclusively via `restore_state(raw, tenant_id=..., run_id=...)` (ADR 0010); `schema_sql()` for migrations; `delete_run` / `delete_tenant` for retention
- [x] ADR covering database schema and deletion strategy (ADR 0018). *Envelope encryption with per-tenant KMS keys is not built in: documented as a wrapper pattern.*

---

## 4. First 20 Eval Cases as a CI Release Gate

**Why:** Prompts, model versions, and tools change over time. Without a regression eval suite, quality
degrades silently.

**Scope:**
- Turn 20 representative tasks from your pilot project into deterministic replay evals using the single-step replay pattern in [Use Case 06](use-cases/06-sub-agents-and-replay-eval.md)
- Assert on concrete invariants: expected tool chosen, forbidden tools never called, valid arguments, and token/latency budget compliance

**Done when:**
- [ ] 20 eval cases run reproducibly against recorded checkpoints
- [ ] Model or system-prompt changes require a green eval run before deployment

---

## 5. Structured Final Output (`output_model`)

**Why:** Many production workflows require a validated Pydantic object at the end of a run (classification,
extraction, structured decision with rationale) rather than free-form text.

**Scope:**
- `start(agent, text, output_model=MyModel)` → stores the validated Pydantic object on `RunState`
- Graceful degradation based on `Capabilities` (`HARNESS_TAXONOMY.md`, Section 8):
  1. Native structured output schema when supported by the provider,
  2. Otherwise a forced final tool call carrying the schema,
  3. Otherwise Pydantic validation of the response text with specific error feedback via retry
- Retries count toward `max_consecutive_retries`; exhaustion ends the run as `failed` with a clear reason

**Done when:**
- [ ] All three fallback strategies tested, including permanently invalid model output
- [ ] Structured output survives checkpoint serialization and `resume`
- [ ] ADR covering `Capabilities` and public API extensions

---

## 6. MCP Integration with Mandatory Risk Classification

**Why:** Model Context Protocol (MCP) servers provide a rich ecosystem of ready-made tools, but also
introduce a major attack surface (rug-pull tool descriptions, untrusted third-party servers). Any MCP
integration must preserve all harness invariants.

**Scope:**
- Wrap MCP tools as `BaseTool` instances; **every** MCP tool requires an explicit local `Risk` classification in configuration — unclassified tools are rejected at startup (fail-closed)
- Server allowlist, pinned versions, and hash verification of tool descriptions
- All MCP tool outputs treated strictly as untrusted external data

**Done when:**
- [ ] Tested against a local mock MCP server: missing risk classification, mutated tool description, unapproved server, tool errors, and timeouts
- [ ] All `PermissionPolicy` and `HookPipeline` rules apply identically to MCP tools
- [ ] ADR defining the trust model for MCP servers

---

## 7. Streaming

**Why:** Interactive UIs need live progress feedback during a run rather than waiting for completion.

**Scope:**
- Async iterator over run `Event`s plus incremental text deltas when the provider adapter streams
- User cancellation cleanly terminates the stream and persists `RunStatus.CANCELLED` (preserving the existing invariant)
- Zero impact on protocol invariants or checkpoints: only complete, validated messages are committed to `RunState`

**Done when:**
- [ ] Tests covering mid-stream cancellation, mid-stream provider failure, and slow consumers (backpressure)
- [ ] Message history after a streamed run passes `check_protocol()`
- [ ] ADR defining the streaming API

---

## Deliberately Deferred

Build these only when a concrete project requires them — otherwise they become dead weight:

- General-purpose guardrail library promoted from the use cases (recipient allowlists, IBAN/card exfiltration guard, cumulative spend limits, secret redaction)
- OpenTelemetry and token-cost listeners with provider pricing tables
- Automatic context compaction for ultra-long runs (today: context budget cleanly terminates oversized runs with `limit_reached`, ADR 0011)
- Router-level circuit breaker that skips a failing provider across runs (today achievable as a `ModelProvider` wrapper, see [Use Case 05](use-cases/05-provider-failover.md))
- Built-in sub-agent tool with approval propagation (today: read-only specialist sub-agents in [Use Case 06](use-cases/06-sub-agents-and-replay-eval.md)), workflow graphs, and parallel execution of `Risk.READ` tools (ADR 0007)
- Hook stage for redacting/transforming tool outputs in-place (today `after_tool` can block or retry)
