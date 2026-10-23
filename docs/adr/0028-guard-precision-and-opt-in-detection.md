# 0028 – Guard Precision, Single-Pass Matching, and Opt-In Detection

- **Status:** Accepted (maintainer, 2026-10-23)
- **Date:** 2026-10-23
- **Affects:** `guardrails.py`, `docs/known-limitations.md`

## Context

The built-in security guards (`SecretGuard`, `EgressGuard`, and `WeakeningGuard`) run synchronously in the event loop before and after model and tool executions. While effective as lightweight heuristics, practical usage revealed several pain points:
1. **False positives:**
   - `SecretGuard` flagged benign kebab-case tokens starting with `sk-` (e.g., `sk-learn-is-a-long-kebab-case-name`).
   - `EgressGuard` treated any `scheme:/path` notation as a URL, falsely blocking benign strings such as `note:/path` or config keys like `path:/usr/bin`.
2. **Maintenance drift:** `EgressGuard` maintained two separate manual schema pattern tables for free text and structured URL keys.
3. **Scan efficiency:** `SecretGuard` and `WeakeningGuard` iterated through multiple regexes sequentially instead of using a single compiled alternation pass.
4. **Blind spots:**
   - Secrets hidden via simple encodings (URL encoding, Base64, Hex).
   - High-entropy secrets passed in explicit configuration assignments (e.g. `db_password = '...'`) or structured argument fields.
   - Pydantic models with `AnyUrl` or `HttpUrl` fields whose names were not in the default URL key list.
   - Modern test and linter suppression directives (e.g. Rust, Go, TypeScript/Jest, ESLint, Java) not present in `WeakeningGuard`.

## Decision

1. **SecretGuard precision and provider formats:**
   - Tighten OpenAI-style keys: require known prefixes (`sk-proj-`, `sk-ant-`, `sk-svcacct-`, `sk-admin-`) or alphanumeric character sets (`sk-[A-Za-z0-9]{20,}`), resolving kebab-case false positives.
   - Add distinctive provider token formats: SendGrid (`SG.`), Twilio (`SK`), Slack incoming webhooks, and Azure connection strings.
   - Combine all active patterns into a single compiled alternation regex with named groups, identifying matched categories in a single linear pass.

2. **Opt-in detection modes in SecretGuard:**
   - `assignments=True`: Inspects structured tool arguments whose keys suggest credentials (`password`, `secret`, `token`, `api_key`, etc.) and scans free text for `key = value` assignment syntax. Excludes common placeholders (`changeme`, `${...}`, `<...>`, `***`) and requires minimum length (8+) and Shannon entropy (>= 2.5).
   - `decode=True`: Budget-bounded single-step decoding. Detects URL-encoded strings, and extracts Base64 and Hex candidates. Discards candidates whose decoded bytes do not meet a printable ASCII density threshold (>= 85%). Joins candidate decoded texts within a bounded length budget and scans once with the compiled regex.

3. **EgressGuard schema unification and precision:**
   - Define a single schema table (`_SCHEMES_TABLE`) from which URL regexes and structured parsers are derived.
   - Restrict single-slash (`scheme:/` and `scheme:\`) detection exclusively to WHATWG special schemes (`http`, `https`, `ws`, `wss`, `ftp`, `file`). Generic schemes require double slashes (`scheme://`), allowing benign notations like `note:/path` or `path:/usr/bin`.
   - Inspect Pydantic tool argument models for `HttpUrl` and `AnyUrl` type annotations, treating matching fields as URLs regardless of their field name.
   - Add `ignore_keys` parameter to downgrade specific URL-like fields to free-text scanning rather than strict URL syntax parsing. The blocking reason identifies the field name when blocked.

4. **WeakeningGuard extensions and single-pass matching:**
   - Combine all suppression patterns into a single compiled alternation regex.
   - Add modern directives: `eslint-disable`, `// nolint`, `#[allow(`, `#[ignore]`, `@SuppressWarnings`, `@Disabled`, `it.skip(`, `describe.skip(`, `xit(`, `t.Skip(`.

## Consequences

- Prevents false blocks on kebab-case identifiers and POSIX paths.
- Linear execution time preserved under `TestNoQuadraticBacktracking`.
- Higher detection rate for obfuscated or assigned credentials when opt-in flags are enabled.
- Guards remain in-process deterministic heuristics; sandbox isolation, least privilege, and network-level firewalls remain necessary.
