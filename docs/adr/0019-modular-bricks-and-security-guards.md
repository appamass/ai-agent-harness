# 0019 – Standalone Modular Bricks, Ergonomic Shorthands, and Built-In Security Guards

- **Status:** Accepted
- **Date:** 2026-10-21
- **Affects:** `ai_agent_harness/__init__.py`, `core/hooks.py`, `core/types.py`, `context.py`, `guardrails.py`, `providers/base.py`, `providers/router.py`, `tools/base.py`, `tools/registry.py`, `tools/executor.py`, `tools/builtin.py`, `runtime/agent.py`

## Context

Downstream projects adopt this repository at very different scales:

1. **Individual building blocks ("pick and choose"):** Some projects already have their own control loop or workflow engine and only want the deterministic guardrails, the 8-step tool execution gate (`ToolExecutor`), or the retry/failover `ModelRouter` — without adopting `Agent` or managing `RunState`. Previously, `HookPipeline`, `ToolExecutor.execute`, and `ModelRouter.complete` required callers to construct `RunState`, `PermissionPolicy`, `HookPipeline`, and `Emit` callbacks manually.
2. **Small script agents vs. full multi-worker systems:** Writing a 10-line script agent previously required defining a `BaseModel` subclass and a `BaseTool` subclass per tool, a `ModelProvider` class for custom model functions, and instantiating `ModelRouter`, `ContextBuilder`, and `ToolRegistry` explicitly.
3. **Supply-chain, credential-leak, exfiltration, and tool-tampering threats:** Real-world agent deployments face leaked API tokens or private keys in files and tool outputs, outbound URL/SSRF data exfiltration, symlink bypasses on protected paths, mutated tool descriptions ("rug pulls"), prompt injection via untrusted tool outputs, and code-check weakening (`# type: ignore`, `pytest.mark.skip`). Previously, several of these defenses lived only as inline examples in `docs/use-cases/`.

## Decision

1. **Standalone bricks without `RunState` ceremony:**
   - `HookPipeline` methods accept `state: RunState | None`, creating an ephemeral `RunState(agent_name="standalone")` when `None` is passed so `Hook` implementations continue to receive a valid `RunState`.
   - `ToolExecutor` accepts a plain iterable of tools, defaults `policy`, `hooks`, and `emit`, and makes `state` optional on `execute(call, state=None, *, approved=False)`.
   - `ModelRouter.complete(request, emit=None)` makes `emit` optional.
   - Built-in guardrails expose direct synchronous check/redaction methods (`SecretGuard.find_secret`, `SecretGuard.redact`, `EgressGuard.check_url`, `EgressGuard.check_text`, `WeakeningGuard.check_text`, `ProtectedPaths.is_protected`) usable without hooks or async code.
2. **Ergonomic shorthands from 5-line agents to full setups:**
   - `@tool(risk=...)` converts a typed sync or async Python function into a validated `SyncTool` or `BaseTool` (deriving a strict `extra="forbid"` Pydantic model from type annotations, hiding an optional `state: RunState` parameter from the model schema, and running `validate_definition()` immediately). `risk` is required: a function never becomes read-only by default, because READ tools skip approval and the write guards.
   - `CallableProvider` wraps an async `(ModelRequest) -> ModelResponse | str` function as a `ModelProvider`.
   - `Agent.create(...)` normalizes a provider (or async callable), system prompt string, tool list, and hook list while keeping `Agent`'s dataclass fields strictly typed (`ModelRouter`, `ContextBuilder`, `ToolRegistry`, `HookPipeline`).
3. **Built-in deterministic security and supply-chain guards (stdlib + Pydantic only):**
   - `SecretGuard` blocks private keys, a fixed list of common API/platform tokens (including Google, npm, PyPI, and live Stripe keys), AWS access keys, JWTs, Bearer and Basic authorization headers, and embedded URL credentials in `before_tool`, `after_tool`, `after_model`, and optionally `on_user_input`, and names only the pattern category in the reason. `redact()` is a standalone helper, the hook never rewrites content.
   - `EgressGuard` requires `allowed_domains` (pass `()` to block all URLs), so a default never silently blocks every answer with a link, and enforces `https://` and an explicit domain allowlist in `before_tool` and `after_model`, blocking direct IP addresses, `localhost`, `.local`/`.internal` hosts, link-local/metadata addresses (`169.254.169.254`), embedded URL credentials, backslashes, control characters, invalid ports, and hosts with characters outside `a-z0-9.-`. Free text is scanned for `scheme://`, `scheme:/`, `scheme:\`, and opaque schemes such as `mailto:`; fields named like a URL (`url`, `href`, `base_url`, or listed in `url_keys`) must hold a complete allowed URL. Reasons do not repeat the host. It reads URLs as text: no redirects, no DNS (see known limitations).
   - `WeakeningGuard` blocks write tools from inserting linter, typechecker, security-scanner, or test suppression directives (`# type: ignore`, `# noqa`, `# ruff: noqa`, `# pylint: disable`, `# nosec`, `pytest.mark.skipif`, `unittest.skip`, `// @ts-ignore`, and similar).
   - `ProtectedPaths(..., root=...)` optionally resolves symlinks and relative paths against a project root directory, blocking symlinks that point into protected paths or outside the root.
   - `DEFAULT_DENIED_READS` in `ReadFile` is expanded to cover `.envrc`, `.ssh`, `.aws`, `*.p12`, `*.pfx`, `.netrc`, `.npmrc`, `.pypirc`, `id_rsa`, `id_ed25519`, and `credentials.json`.
   - `BaseTool.digest()` computes a full SHA-256 over `(name, description, canonical_input_schema, risk, timeout_s, untrusted_output)`, so a tool that keeps its text but lowers its risk is noticed. `ToolRegistry(..., expected_digests=...)` verifies pins on registration and in `verify()`, and rejects a pin that matches no registered tool. The executor does not re-check on every call.
   - `as_untrusted(source, text)` and `BaseTool.untrusted_output = True` wrap external tool outputs in `<untrusted source="...">` blocks after truncation, so the closing tag is never clipped (the result can exceed `max_result_chars` by the frame). Any `<untrusted` or `</untrusted` inside the content is escaped as `&lt;`, not deleted, because deleting can join the remaining parts into a new tag. The frame is a hint for the model, not a boundary.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Split repository into multiple PyPI/Git sub-packages (`harness-guardrails`, `harness-tools`, etc.) | Multiplies versioning, lockfile, and release overhead when `core` already has only `pydantic` as a runtime dependency |
| Widen `Agent` dataclass field types to unions (`router: ModelRouter \| ModelProvider`) | Breaks strict type checking (`mypy --strict`, `pyright`) for callers reading `agent.router` or `agent.tools` and complicates `dataclasses.replace` |
| External regex/DLP or URL-parsing dependencies | Violates the zero-extra-core-dependency principle (ADR 0001); Python's `re`, `urllib.parse`, and `ipaddress` suffice |

## Consequences

- Positive: Individual layers (`guardrails`, `ToolExecutor`, `ModelRouter`, `CheckpointStore`) can be used or copied standalone in 2–5 lines of code; small agents require minimal boilerplate while the 9 invariants stay enforced by `start` and `resume` (a `ToolExecutor` used alone does not enforce human approval, see known limitations); common supply-chain and exfiltration defenses ship out of the box.
- Negative / Cost: Larger public API surface in `ai_agent_harness/__init__.py` (14 additional exports) that must be maintained across minor releases.
- Verification: `tests/test_standalone_bricks.py`, `tests/test_security_guards.py`, `tests/test_function_tools.py`, `tests/test_docs.py`, and `tests/test_architecture.py`.
