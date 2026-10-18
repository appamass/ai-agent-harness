# Anthropic wire-format fixtures

HTTP response bodies of the Anthropic Messages API (`POST /v1/messages`), used by
`tests/test_anthropic_provider.py` through a mock HTTP transport. The real SDK parses them, so the tests
exercise real serialization, parsing, and error classes — only the network is replaced.

- **`recorded/`** — responses **recorded from the live API** (model `claude-haiku-5-5`, live runs on 2026-10-18; only the
  message `id` was replaced by `msg_fixture`):
  a plain answer; tool calls (`read_file`, `add_one`, `write_file`, `save_note`); a final answer after a tool result;
  a truncation run with thinking blocks (`truncation-run-*`); **thinking together with a tool call**
  (`thinking-with-tool-use-*`); thinking followed by an answer; and answers that show **prompt caching** in `usage`
  (`cache-write-answer`, `cache-hit-answer`).
  Every file is translated and round-tripped by `test_recorded_real_responses_translate_and_round_trip`.
  Add more with `HARNESS_RECORD_DIR=/tmp/recorded pytest -m live`, review them (no personal data, no secrets),
  and copy them in. Haiku 5.5 thinks before a tool call only some of the time.
- **`*.json` in this folder** — **hand-written** from the documented response shape, with exact values for precise
  assertions and for cases that have **no recording yet: redacted thinking and refusal**. Replace those two by recordings
  when you have them.
