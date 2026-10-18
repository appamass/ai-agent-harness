# Use Cases

High-stakes scenarios demonstrating what this harness guarantees out of the box — and where its
boundaries lie. Each use case includes a threat model, architecture diagram, reference implementations
of project-specific building blocks, and end-to-end walkthroughs.

**Every scenario is executable.** All Python blocks starting with `# executable` are executed
sequentially in CI by `tests/test_docs.py`, including every `assert` statement. If the harness API
changes, the test suite fails before documentation can drift.

| No. | Use Case | Core Topics |
|---|---|---|
| [01](01-finance-and-communications-assistant.md) | Personal Assistant with Bank Account, Email, and Social Media | Lethal trifecta, prompt injection, AES-256-GCM encrypted checkpoints, recipient allowlist, IBAN/card exfiltration guard, payment limits, idempotency, GDPR crypto-shredding |
| [02](02-autonomous-coding-agent-fleet.md) | Autonomous Coding Agent Fleet in CI | Reward hacking, protected paths, `WeakeningGuard`, `TestsMustPass` completion hook, mid-side-effect crash recovery, concurrent workers |
| [03](03-multi-tenant-customer-support.md) | Multi-Tenant Customer Support with Refunds | Tenant isolation, per-tenant refund rules, provider outage & failover, batch recovery after total outage, multi-day `ask_user` pauses, token cost breakdown by category |
| [04](04-incident-responder.md) | On-Call Incident Responder — **Recommended First Project** | Read-only diagnosis, pod crash mid-tool-batch, mobile approval in another process, double-click protection, change freeze via `KillSwitch`, credentials via `deps` |
| [05](05-provider-failover.md) | Mid-Run Model Provider Failover | Stripping foreign thinking signatures (`without_foreign_thinking`), self-correcting broken tool JSON (`arguments_error`), cache & reasoning token accounting, circuit breaker boundaries |
| [06](06-sub-agents-and-replay-eval.md) | Specialist Sub-Agents as Tools & Single-Step Replay Evals | Read-only specialist sub-agent as a `BaseTool`, per-agent limits, evaluating candidate models on recorded production checkpoints without executing tools |

**Deliberately Not Claimed as a Built-in Use Case Today:**

- **In-place pseudonymization of tool outputs via a hook:** `after_tool` can allow, block, or retry a
  tool result, but cannot rewrite its content. Redact personal data inside the tool implementation
  itself before returning the string. Checkpoints and the model provider receive the returned tool
  result verbatim unless `after_tool` blocks it.
- **MCP integration with risk classification:** Planned as Step 6 in [`../next-steps.md`](../next-steps.md).
- **Automated credit or lending decisions for natural persons:** While technically similar to Use Cases 1
  and 3, consumer credit scoring is classified as a high-risk AI system under the EU AI Act — do not use
  it as an initial pilot project.

**Legend used across documents:** ✅ Included in core harness · 🧩 Reference implementation in this
document (copy and harden per project) · 🔲 Project-specific infrastructure.

Use Case 01 uses `cryptography`, which is included in the development dependencies (`.[dev]`) rather
than core runtime dependencies. If `cryptography` is not installed locally, `tests/test_docs.py` marks
that single doc test as `skipped` — in CI it always runs.
