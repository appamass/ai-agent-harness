# 0025 – Streaming Events and Text Deltas

- **Status:** Accepted (maintainer, 2026-10-22)
- **Date:** 2026-10-22
- **Affects:** `core/events.py`, `providers/base.py`, `providers/router.py`, `providers/anthropic.py`, `providers/openai.py`, `runtime/runner.py`

## Context

UIs need live progress, incremental text, and a working "Stop", without weakening the guarantees: partial text must never enter the history (protocol), cancellation must persist `cancelled` (crash honesty), content must not reach the `EventBus` (ADR 0011), and a slow consumer must not cause unbounded memory.

## Decision

1. **API.** `stream_start(...)` and `stream_resume(...)` return a `StreamHandle`, an async iterator over `Event | TextDelta`. `await handle` returns the final `RunState` (unread items are discarded meanwhile). `async with` or `aclose()` stops early with `RunStatus.CANCELLED`; a bare `break` does not.
2. **No partial text in state.** Deltas go only to the consumer. `RunState.messages` gets the complete, validated `ModelResponse`, so `check_protocol()` holds after a cancel or mid-stream failure (`failed` is persisted).
3. **Backpressure.** A bounded queue (`queue_size`) between run task and consumer. A full queue pauses the run for at most `idle_timeout_s`; then the run is cancelled and `CANCELLED` is persisted.
4. **Cancellation never blocks.** The stream sink owns its lifecycle: closing it first makes every later write non-blocking, so the `CANCELLED` events and checkpoint cannot hang on a full queue. A run task is only cancelled once it has started; a task closed before that ends without running its body: a new run is saved as `CANCELLED`, a resumed run keeps its state and checkpoint unchanged (it was not claimed yet). After `cancel()`, iteration ends once buffered items are read.
5. **Providers.** `stream(request)` on the provider contract, implemented by `FakeProvider`, `AnthropicProvider`, `OpenAIProvider`; others (such as `CallableProvider`) fall back to `complete()` as one delta. `StreamDelta` carries answer text only: reasoning arrives as a `ThinkingBlock` in the final response. `ModelRouter.stream` and `complete` share one retry and failover loop; streams are bounded by `RetryPolicy.request_timeout_s`, and the time a slow consumer holds up `on_delta` is not charged to the provider. Once a delta reached the consumer, a failure is raised as it is: no retry, no failover.
6. **Deltas are unfiltered live output.** `after_model` hooks and guards see the response only when it is complete, so a response they block has already been shown. Do not forward deltas to end users when that matters.

## Rejected

- Text deltas on the `EventBus`: content in telemetry (ADR 0011).
- An unbounded queue: memory grows with a slow client.

## Consequences

Close handles with `async with`. Tested in `tests/test_streaming.py` (cancel, early close, abandonment, full queue, slow consumer, router timeout).
