# 0028 – Guard Precision and Opt-In Detection

- **Status:** Accepted (maintainer, 2026-10-23)
- **Date:** 2026-10-23
- **Affects:** `guardrails.py`, `docs/known-limitations.md`

## Context

The built-in security guards (`SecretGuard`, `EgressGuard`, and `WeakeningGuard`) run synchronously in the event loop before and after model and tool executions. While effective as lightweight heuristics, practical usage revealed several pain points:
1. **False positives:**
   - `SecretGuard` flagged benign kebab-case tokens starting with `sk-` (e.g., `sk-learn-is-a-long-kebab-case-name`).
   - `EgressGuard` treated any `scheme:/path` notation as a URL, falsely blocking benign strings such as `note:/path` or config keys like `path:/usr/bin`.
2. **Maintenance drift:** `EgressGuard` maintained two separate manual schema pattern tables for free text and structured URL keys.
3. **Scan efficiency:** `SecretGuard` and `WeakeningGuard` search many patterns one after the other. A single alternation of all patterns was considered and measured: on 1 MB of text it is slower (`SecretGuard` 1.7x, `WeakeningGuard` 4.5x), because one large regex cannot skip ahead per pattern. So the patterns stay separate.
4. **Blind spots:**
   - Secrets hidden via simple encodings (URL encoding, Base64, Hex).
   - High-entropy secrets passed in explicit configuration assignments (e.g. `db_password = '...'`) or structured argument fields.
   - Pydantic models with `AnyUrl` or `HttpUrl` fields whose names were not in the default URL key list.
   - Modern test and linter suppression directives (e.g. Rust, Go, TypeScript/Jest, ESLint, Java) not present in `WeakeningGuard`.

## Decision

1. **SecretGuard precision and provider formats:**
   - Tighten OpenAI-style keys: require known prefixes (`sk-proj-`, `sk-ant-`, `sk-svcacct-`, `sk-admin-`) or alphanumeric character sets (`sk-[A-Za-z0-9]{20,}`), resolving kebab-case false positives.
   - Add distinctive provider token formats: SendGrid (`SG.`), Twilio (`SK`), Slack incoming webhooks, and Azure connection strings.
   - Keep the patterns as separate compiled regexes, searched in order. This also keeps every custom pattern valid as written (inline flags such as `(?i)`, named groups, backreferences) and lets category names be any text.

2. **Opt-in detection modes in SecretGuard:**
   - `assignments=True`: Inspects structured tool arguments whose keys suggest credentials (`password`, `secret`, `token`, `api_key`, etc.; camelCase such as `dbPassword` is split like URL keys) and scans free text for `key = value` assignment syntax (also `dbPassword: value` and JSON keys). Excludes common placeholders (`changeme`, `${...}`, `<...>`, `***`) and requires minimum length (8+) and Shannon entropy (>= 2.5).
   - `decode=True`: Budget-bounded single-step decoding. Detects URL-encoded strings, and extracts Base64 and Hex candidates. A Base64 run is decoded with missing or extra `=` padding (lenient decoders accept both) and at each of the 4 offsets. A path in front of the data (`example/api/v1/upload/<base64>`) is part of the run and moves the 4-character blocks; at the right offset the path only becomes a few garbage bytes before the data. Only readable stretches of 12 or more characters of the decoded bytes are kept, so binary data such as hashes and the garbage bytes drop out. The kept texts are joined, limited to 50,000 characters in all, and scanned once with the patterns.

3. **EgressGuard schema unification and precision:**
   - Build the free-text URL pattern and its cheap pre-check (`_URL_MARKERS`, substring checks on the lower-cased text; a regex was measured to be 6x slower on prose) in one place (`_build_url_patterns`), so they cannot drift apart.
   - Restrict single-slash (`scheme:/` and `scheme:\`) detection exclusively to WHATWG special schemes (`http`, `https`, `ws`, `wss`, `ftp`, `file`). All schemes match in any case (`HTTPS:/host` and `MAILTO:` are URLs, as in WHATWG parsers); the pre-check matches in any case too. Generic schemes require double slashes (`scheme://`), allowing benign notations like `note:/path` or `path:/usr/bin`.
   - Inspect Pydantic tool argument models for `HttpUrl` and `AnyUrl` type annotations, treating matching fields as URLs regardless of their field name.
   - Add `ignore_keys` parameter to downgrade specific URL-like fields to free-text scanning rather than strict URL syntax parsing. The blocking reason identifies the field name when blocked.

4. **WeakeningGuard extensions:**
   - Keep the suppression patterns separate, as for `SecretGuard`.
   - Add modern directives: `eslint-disable`, `// nolint`, `#[allow(`, `#[ignore]`, `@SuppressWarnings`, `@Disabled`, `it.skip(`, `describe.skip(`, `xit(`, `t.Skip(`.

## Consequences

- Prevents false blocks on kebab-case identifiers and POSIX paths.
- Linear execution time preserved under `TestNoQuadraticBacktracking`, which also covers the opt-in modes and the new patterns.
- Higher detection rate for obfuscated or assigned credentials when opt-in flags are enabled.
- Guards remain in-process deterministic heuristics; sandbox isolation, least privilege, and network-level firewalls remain necessary.
