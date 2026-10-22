# 0019 – Standalone Bricks, Shorthands, and Built-In Security Guards

- **Status:** Accepted
- **Date:** 2026-10-21
- **Affects:** `ai_agent_harness/__init__.py`, `core/hooks.py`, `core/types.py`, `context.py`, `guardrails.py`, `providers/base.py`, `providers/router.py`, `tools/base.py`, `tools/registry.py`, `tools/executor.py`, `tools/builtin.py`, `runtime/agent.py`

## Context

1. Projects with their own loop want only the guardrails, the `ToolExecutor` gate, or the `ModelRouter`, but these needed a `RunState`, policy, hook pipeline, and emit callback.
2. A 10-line script agent needed a model class and a tool class per tool, a provider class, and explicit router, context, and registry.
3. Real deployments face leaked tokens and keys, URL-based data exfiltration (SSRF), symlink bypasses of protected paths, mutated tool descriptions, injection through tool output, and weakened code checks (`# type: ignore`, skipped tests). Several defenses existed only as examples in docs.

## Decision

1. **Bricks without `RunState`:** `HookPipeline` methods accept `state: RunState | None` (an ephemeral state is created). `ToolExecutor` takes a plain list of tools, defaults `policy`, `hooks`, `emit`, and `execute(call, state=None, *, approved=False)`. `ModelRouter.complete(request, emit=None)`. Guards offer direct sync methods: `SecretGuard.find_secret` / `redact`, `EgressGuard.check_url` / `check_text`, `WeakeningGuard.check_text`, `ProtectedPaths.is_protected`.
2. **Shorthands:** `@tool(risk=...)` turns a typed sync or async function into a validated tool with a strict `extra="forbid"` model; an optional `state: RunState` parameter is hidden from the model schema; `risk` is required, because READ skips approval and write guards. `CallableProvider` wraps an async function. `Agent.create(...)` normalizes provider, prompt, tools, and hooks while `Agent` itself stays strictly typed.
3. **Guards (stdlib and Pydantic only):**
   - `SecretGuard` blocks private keys, common API tokens, AWS keys, JWTs, authorization headers, and URL credentials in `before_tool`, `after_tool`, `after_model`, optionally `on_user_input`; it names only the category. `redact()` is a helper; the hook never rewrites content.
   - `EgressGuard` needs `allowed_domains` (`()` blocks all URLs) and enforces `https://` plus the allowlist in `before_tool` and `after_model`. It blocks IPs, `localhost`, `.local`/`.internal`, metadata addresses, URL credentials, backslashes, control characters, bad ports, odd host characters. Free text is scanned for `scheme://`, `scheme:/`, `scheme:\`, and opaque schemes like `mailto:`; fields named `url`, `href`, `base_url`, or in `url_keys` must hold a full allowed URL. Reasons never repeat the host. It reads URLs as text: no redirects, no DNS.
   - `WeakeningGuard` blocks write tools from adding suppression directives (`# type: ignore`, `# noqa`, `# nosec`, `pytest.mark.skipif`, `unittest.skip`, `// @ts-ignore`, …).
   - `ProtectedPaths(..., root=...)` resolves symlinks and relative paths against a root and blocks targets in protected paths or outside the root.
   - `DEFAULT_DENIED_READS` also covers `.envrc`, `.ssh`, `.aws`, `*.p12`, `*.pfx`, `.netrc`, `.npmrc`, `.pypirc`, `id_rsa`, `id_ed25519`, `credentials.json`.
   - `BaseTool.digest()` is a SHA-256 over name, description, canonical input schema, risk, timeout, `untrusted_output`, so lowering a risk is noticed. `ToolRegistry(..., expected_digests=...)` verifies pins on registration and `verify()` and rejects a pin that matches no tool. The executor does not re-check per call.
   - `as_untrusted(source, text)` and `BaseTool.untrusted_output = True` wrap external output in `<untrusted source="...">` after truncation, so the closing tag is never cut. `<untrusted` and `</untrusted` inside the content are escaped as `&lt;`, not deleted, because deleting can join the rest into a new tag. The frame is a hint for the model, not a boundary.

## Rejected

- Several Git sub-packages: multiplies versions and releases while the core needs only `pydantic`.
- Union types on `Agent` fields: break strict typing for callers reading `agent.router` or `agent.tools`.
- External regex, DLP, or URL libraries: `re`, `urllib.parse`, and `ipaddress` suffice (ADR 0001).

## Consequences

A larger public API (14 more exports). A `ToolExecutor` used alone does not enforce human approval (`docs/known-limitations.md`). Tested in `tests/test_standalone_bricks.py`, `test_security_guards.py`, `test_function_tools.py`, `test_docs.py`, `test_architecture.py`.
