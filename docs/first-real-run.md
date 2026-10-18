# The First Real Run

A runbook for your first contact with a live language model. The goal is **to learn, not to ship**:
How does the model interact with your tools, how often do guardrails fire, and what do latency and
token usage look like? Everything is configured so that any mistake stays cheap, visible, and harmless.

---

## 1. Prerequisites

Do not start a live run until every item below is checked:

- [ ] **Step 0 from [`next-steps.md`](next-steps.md)** is complete: local and CI checks are green, ADRs reviewed.
- [ ] **A real model provider adapter** exists and has tests backed by recorded API responses.
      `AnthropicProvider` is included (see 1a below); for any other provider, write your own first
      (skill `add-provider-adapter`). The core deliberately ships without provider SDKs.
- [ ] **API key** is stored in an environment variable or secret manager — never in the repository,
      prompts, context, or checkpoints.
- [ ] **Spend limit at the provider** is configured (daily or monthly hard cap in the provider console).
      `Limits` cap a single run; the provider spend limit caps total damage from a bug or runaway loop.
- [ ] **No real customer data:** dedicated test tenant, synthetic test data, zero access to production systems.
- [ ] A human operator is present during the runs and knows how to trigger the kill switch.

---

## 1a. Connect Anthropic (Claude) — about 5 minutes

1. **Credits and key.** In the [Anthropic Console](https://console.anthropic.com) add a few dollars of credit
   (leave auto-reload off: the balance is your hard spend cap), then create an API key named `harness-live-test`.
2. **Install** the optional extra:

   ```bash
   pip install "ai-agent-harness[anthropic]"
   ```

3. **Set the key in your own shell** — never in a file, a prompt, a commit, or a chat:

   ```bash
   export ANTHROPIC_API_KEY=...        # paste the key here, in your terminal only
   ```

4. **Use it.** `AnthropicProvider` reads `ANTHROPIC_API_KEY` itself. This block needs a key, so it is not run by the test suite:

   ```python
   import asyncio

   from ai_agent_harness import Agent, ContextBuilder, Limits, ModelRouter, RetryPolicy, ToolRegistry, start
   from ai_agent_harness.providers.anthropic import AnthropicProvider

   provider = AnthropicProvider("claude-haiku-5-5")
   agent = Agent(
       name="first-live-run",
       router=ModelRouter(provider, retry=RetryPolicy(max_attempts=3)),
       context=ContextBuilder("You are a concise assistant."),
       tools=ToolRegistry([]),
       limits=Limits(max_steps=5, max_total_tokens=20_000, max_seconds=60),  # a bug cannot run up a bill
   )
   async def main() -> None:
       state = await start(agent, "Say hello in one sentence.")
       print(state.status, state.last_text, state.usage)
       await provider.aclose()  # release the HTTP connections when you are done

   asyncio.run(main())
   ```

5. **Or try the complete example** — an assistant that reads a project and asks before it writes
   (`python examples/repo_assistant.py`, see [`examples/README.md`](../examples/README.md); `--demo` needs no key).

6. **Run the live test suite** (eleven tests, a few cents with Haiku; skipped without a key, never part of a default run):

   ```bash
   pytest -m live -v                       # model: HARNESS_LIVE_MODEL (default claude-haiku-5-5)
   HARNESS_RECORD_DIR=/tmp/recorded pytest -m live   # also saves every response as JSON for the test fixtures
   ```

   They check: a plain answer; a tool call with `ReadFile` (and that file contents never reach events); three sequential
   tool turns (with thinking on, the API verifies the signatures we send back — the model may skip thinking in short tasks); a write tool that waits for approval, rejects a stale
   `pending_id`, and runs after the right approval; truncated answers ending in a defined status; an unknown model and a wrong
   key (not retryable, key never in the error).

Model behavior and API rules that the adapter handles for you (see `ai_agent_harness/providers/anthropic.py` and ADR 0017): no
sampling parameters, thinking signatures sent back unchanged, errors reduced to status and type, no hidden SDK retries, prompt
caching on by default.

**Results of the first live run** (fill in after you ran it; then update the status notes in the README and in [`known-limitations.md`](known-limitations.md)):

| Date | Model | Live tests | Tokens in / out | Cost | Findings |
|---|---|---|---|---|---|
| 2026-10-18 | `claude-haiku-5-5` | Run 1: 7 of 7 passed. Run 2: 6 of 7, the failure was a bug in the test helper (it closed the client between steps), fixed. Run 3: 7 of 7 passed, no `ResourceWarning`. **Run 4 (with the example agent and 4 more tests): 11 of 11 passed** | see Console | a few cents per run | Adapter did not close its HTTP connections (`ResourceWarning`, Python 3.14): `AnthropicProvider.aclose()`. Live test helper closed the client too early and the recorder overwrote files: fixed. Thinking together with a tool call was recorded in run 4 and its signature was sent back successfully; prompt caching is visible in `usage` (cache read/write). Example agent run by hand: a note saved only after approval, `.env` could not be read, Ctrl-C exits cleanly. **Not proven yet:** redacted thinking, refusals, 429/529 from the real API, other models, other providers, long runs. |

---

## 2. Configuration for the First Runs

Tight limits, read-only tools, persistent checkpoints, metadata-only logging, and an emergency stop
that can be flipped without redeploying (here: touching a `KILL` file in the working directory).
The block below runs in the test suite using `FakeProvider` — in your live run, replace it with your
provider adapter.

```python
# executable
import asyncio
import logging
import tempfile
from pathlib import Path

from ai_agent_harness import (
    Agent,
    Autonomy,
    ContextBuilder,
    HookPipeline,
    JsonFileCheckpointStore,
    KillSwitch,
    Limits,
    ModelRouter,
    ProtectedPaths,
    ReadFile,
    RetryPolicy,
    RunStatus,
    ToolRegistry,
    start,
)
from ai_agent_harness.core import Event
from ai_agent_harness.observability import log_listener
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

workdir = Path(tempfile.mkdtemp())
(workdir / "notes.txt").write_text("Quarterly metric: 42", encoding="utf-8")
kill_flag = workdir / "KILL"  # creating this file stops all runs immediately

provider = FakeProvider([  # live run: replace with your chosen provider adapter
    tool_response(("read_file", {"path": "notes.txt"})),
    text_response("The quarterly metric is 42."),
])

agent = Agent(
    name="first-run",
    router=ModelRouter(provider, retry=RetryPolicy(max_attempts=2, request_timeout_s=60)),
    context=ContextBuilder("You answer questions about files in the working directory. Keep answers concise."),
    tools=ToolRegistry([ReadFile(workdir)]),  # read-only; do not register WRITE or IRREVERSIBLE tools yet
    autonomy=Autonomy.READ_ONLY,
    hooks=HookPipeline([KillSwitch(lambda _: kill_flag.exists()), ProtectedPaths()]),
    checkpoints=JsonFileCheckpointStore(workdir / "checkpoints"),
    limits=Limits(
        max_steps=8,
        max_total_tokens=50_000,
        max_seconds=120,
        max_consecutive_retries=2,
        max_context_chars=60_000,
        max_input_chars=5_000,
    ),
)

logging.basicConfig(level=logging.INFO)
agent.events.subscribe(log_listener)  # metadata only, safe to ship to central logs
seen: list[Event] = []


async def remember(event: Event) -> None:
    seen.append(event)


agent.events.subscribe(remember)

state = asyncio.run(start(agent, "What is in notes.txt?", tenant_id="test-tenant"))
assert state.status is RunStatus.DONE
assert "42" in state.last_text
assert all("Quarterly metric" not in str(e.data) for e in seen)  # content never leaks into events

kill_flag.touch()  # kill switch: the next run stops before even calling the model
stopped = asyncio.run(start(agent, "And now?", tenant_id="test-tenant"))
assert stopped.status is RunStatus.BLOCKED
assert provider.remaining == 0
```

`max_context_chars` applies per model request and is a character-based estimate (≈ 4 characters per token).
For your first runs, keep it well below the model's context window.

---

## 3. Staged Rollout

Only advance to the next stage once the current stage completes with zero open findings. For a full
worked example of Stages A and B, see [Use Case 04 (Incident Responder)](use-cases/04-incident-responder.md).

| Stage | Autonomy | Tools | Scope |
|---|---|---|---|
| A | `read_only` | Read-only tools only | 10–20 prepared tasks, run and inspected one by one |
| B | `suggest` | Read + `WRITE` against a disposable test copy | Every write action inspected via the approval preview |
| C | `act_with_approval` | Same as B + `IRREVERSIBLE` against sandbox accounts | Only with recipient allowlists and amount limits from the use cases |

**Never in the first runs:** `autonomous`, live accounts (email, banking, social), multiple tenants,
or unattended overnight/batch jobs.

---

## 4. Post-Run Inspection Checklist

- [ ] **Status and reason** (`state.status`, `state.status_reason`): as expected? If `limit_reached` — which limit?
- [ ] **Resource usage** (`state.step`, `state.usage`, `state.elapsed_s`) proportional to the task
- [ ] **Tool calls** in `state.messages`: sensible, no retry loops, no hallucinated tool names
- [ ] **Blocked actions** (`tool_blocked` events): at which stage (`source`: policy, before_tool, after_tool)?
      Inspect the tool result or `state.status_reason` — was the block expected?
- [ ] **Checkpoint** loads cleanly (`await agent.checkpoints.load(...)`) with identical status and history
- [ ] **Logs** contain metadata only: no file contents, tool arguments, or response text
- [ ] **Provider billing** in the provider console matches `state.usage`

---

## 5. Trigger the Kill Switch Immediately If…

- The model repeatedly attempts to call non-existent or unoffered tools
- `ProtectedPaths` triggers (attempts to modify tests, CI, or thresholds)
- Retries (`retry`) or refusals spike without the run making progress
- Cost or latency per task significantly exceeds expectations
- Any sensitive content appears in logs or events

Afterward: understand the root cause, capture it as a failing test first, and only then implement the fix (`AGENTS.md`).

---

## 6. What to Commit Back to the Repository

- **Recorded responses** from the live model (scrubbed of API keys and personal data) as test fixtures for the adapter
- **Every finding as a regression test:** reproduce bad behavior as a failing test first, then fix it
- **Tuned default `Limits`** with rationale documented in an ADR
- **An initial eval suite** built from the Stage A tasks — the foundation for automated release-gate evals (`next-steps.md`)
