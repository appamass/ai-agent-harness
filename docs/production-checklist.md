# Production Readiness Checklist (Per Project)

Work through this list before you let an agent act on real data. Items marked with a document link explain how.

- [ ] Real model adapter with accurate `retryable` error classification and at least one fallback
- [ ] Checkpoint store: `JsonFileCheckpointStore` (POSIX) or `SqliteCheckpointStore` for a single host; `PostgresCheckpointStore` for several workers
      (TLS, least-privilege role, encryption at rest or `AesGcmCodec`, a retention policy using `purge_finished` / `delete_run` / `delete_tenant`)
- [ ] UI or notification channel for clarifying questions and approvals: show `state.status_reason`, and send back the `pending_id` that was shown
- [ ] Explicitly configure the autonomy level and `allowed_tools`; the default is `read_only`
- [ ] Tune `Limits` to your workload's cost and latency budget: `max_steps`, `max_total_tokens`, `max_seconds`, `max_consecutive_retries`,
      `max_context_chars` (3 characters count as one token, sized for the **smallest** context window in your router), `max_context_tokens` (optional, calibrated by
      the tokens your provider reports), and `max_input_chars`.
      `max_total_tokens` counts `Usage.total`, which **includes** cache read and write tokens
- [ ] Wire `KillSwitch` to a runtime toggle (feature flag, file, or database row) that operations can flip without redeploying
- [ ] Stop cleanly on `SIGTERM` and run a recovery job, so that deploys and crashes do not leave runs `running` forever. Never use `recover=True`
      on a run whose worker may still live: the tool can run twice (see [Shutdown and crash recovery](guide.md#shutdown-and-crash-recovery))
- [ ] Subscribe event listeners for logging, tracing, and cost tracking; alert on `failed` and `limit_reached`
- [ ] Keep secrets strictly inside tool code and adapters — never in prompts, context, or checkpoints
- [ ] Build project-specific evals with representative tasks and run them in CI as a release gate
- [ ] Define data retention and deletion policies for checkpoints and logs (GDPR / privacy compliance; see [Retention and Deletion](guide.md#retention-and-deletion))
- [ ] Read [`known-limitations.md`](known-limitations.md)
