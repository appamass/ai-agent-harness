# Adopting Harness Invariants in Other Frameworks

If your loop already exists (LangGraph, Pydantic AI, or your own), you do not have to adopt the whole runtime.
Take the part that matters most and put it where your framework executes tools: the **gate**.

Every code block marked `# executable` below runs in CI (`tests/test_docs.py`).

---

## Recipe 1: `ToolExecutor` as your tool node

`ToolExecutor` is a standalone brick. Give it the tool calls your framework got from the model and it does the
rest: argument validation, policy (autonomy × risk), hooks, approval pause, timeout, output truncation and
untrusted-content framing, and safe error messages. In LangGraph this is the body of your tool node; in any other
loop it replaces the line where you call the tool function directly.

```python
# executable
import asyncio

from ai_agent_harness import Autonomy, PermissionPolicy, Risk, ToolCall, ToolExecutor, ToolResult, tool
from ai_agent_harness.core.state import PendingCall


@tool(risk=Risk.READ)
def lookup(city: str) -> str:
    """Looks up a city."""
    return f"{city}: sunny"


@tool(risk=Risk.IRREVERSIBLE)
def send_email(to: str) -> str:
    """Sends an email."""
    return f"sent to {to}"


gate = ToolExecutor([lookup, send_email], PermissionPolicy(autonomy=Autonomy.AUTONOMOUS))


async def tool_node(calls: list[ToolCall]) -> list[ToolResult | PendingCall]:
    """Your framework's tool node: nothing runs except through the gate."""
    return [await gate.execute(call) for call in calls]


async def main() -> None:
    read, mail = await tool_node(
        [
            ToolCall(id="c1", name="lookup", arguments={"city": "Bern"}),
            ToolCall(id="c2", name="send_email", arguments={"to": "a@example.com"}),
        ]
    )
    assert isinstance(read, ToolResult) and "sunny" in read.content
    # IRREVERSIBLE never runs without a human, even at the highest autonomy: you get a pause, not a result
    assert isinstance(mail, PendingCall) and mail.kind == "approval"

    # Bind the approval to the call the human saw: compare ids yourself, then run exactly that call
    shown = mail.call
    approved = await gate.execute(shown, approved=True)
    assert isinstance(approved, ToolResult) and not approved.is_error

    # Invalid arguments and unknown tools come back as safe errors, never as exceptions
    bad = await gate.execute(ToolCall(id="c3", name="lookup", arguments={}))
    assert isinstance(bad, ToolResult) and bad.is_error


asyncio.run(main())
```

What you still own when you use the gate without `Agent`: the `pending_id` binding and the checkpoint. A bare
executor has no run state, so keep the pending call (id, tool, arguments) in your own store, show exactly that to the
human, and only call `execute(call, approved=True)` for the same id (see
[known limitations](../../../../docs/known-limitations.md)). With `start` and `resume` this is enforced for you
([ADR 0014](../../../../docs/adr/0014-bind-resume-to-pending-call.md)).

---

## Recipe 2: Guardrails that fail closed

Hooks run before and after every tool call. A hook that raises, or returns something that is not a valid decision,
blocks the call. Reuse the built-in `SecretGuard`, `EgressGuard`, `ProtectedPaths`, and `KillSwitch`, or write your own
`Hook`, and hand them to the executor: `ToolExecutor(tools, policy, hooks=[...])`.

---

## Recipe 3: Crash recovery without double execution

When a worker dies in the middle of a tool call, the wrong answer is to run the call again. The harness does this
instead:

1. **Write-ahead:** persist the state *before* every tool call, including the calls still planned and the results so far.
2. **Honest recovery:** resume with `recover=True`. The call that was in flight gets a synthetic error result,
   an "interrupted ... outcome is unknown" error. The model, or a human, decides what to do. Finished calls keep their real result.
3. **Optimistic locking:** save with `ON CONFLICT … DO UPDATE … WHERE revision = :expected_revision`. If another worker took the
   run, the save fails with `ConcurrencyError` before any tool can run twice.

The stores (`SqliteCheckpointStore`, `PostgresCheckpointStore`, `JsonFileCheckpointStore`) implement this. If you rebuild
it in your own stack, these three rules are the part to keep.

Recover a run only when its worker is surely dead. The lock stops two recoveries, but not a recovery of a run that is still alive: its tool
can run twice. Wait until the run has been silent for longer than a live worker can be, as the
[recovery job](../../../../docs/guide.md#shutdown-and-crash-recovery) does.
