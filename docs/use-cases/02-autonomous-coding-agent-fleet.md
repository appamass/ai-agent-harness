# Use Case 2: Autonomous Coding Agent Fleet in CI

> **Difficulty: Extreme.** Multiple workers process tickets in parallel: writing code, running tests,
> and proposing merges. Because the model is evaluated on passing tests, it has a strong incentive to
> cheat (reward hacking). Worker processes can crash mid-side-effect, and multiple workers can race to
> recover the same job.

All Python blocks starting with `# executable` are executed by `tests/test_docs.py`.

| Component | Status |
|---|---|
| `ProtectedPaths`, Autonomy × Risk matrix, approvals, write-ahead checkpoints, `recover`, optimistic concurrency lock | ✅ In core harness |
| `WeakeningGuard`, `TestsMustPass`, repository/test/merge tools | 🧩 Reference implementation in this document |
| Container sandbox (no network, read-only mounts for protected paths), live Git/CI integration, hidden evaluation tests | 🔲 Project-specific |

---

## 1. Threat Model

| # | Threat | Example | Defense Layer |
|---|---|---|---|
| R1 | Deleting or gutting tests | `def test_add(): pass` | `ProtectedPaths` (✅ Core) |
| R2 | Path normalization bypasses | `Tests/test_calc.py` on macOS, `./tests/../tests/x.py` | Path normalization + casefolding in `ProtectedPaths` (✅ Core) |
| R3 | Disabling linters or test checks in code | `# type: ignore`, `pytest.mark.skip`, `pragma: no cover` | `WeakeningGuard` |
| R4 | Hardcoding test inputs | `if (a, b) == (2, 3): return 5` | ⚠️ Cannot be caught by regex → hidden test suite + mutation testing in CI |
| R5 | Unreviewed merge to `main` | Model merges failing or unreviewed code | `merge_to_main` is `Risk.IRREVERSIBLE` → always requires human approval |
| R6 | Duplicate side effects after a crash | Preview deployment triggered twice | Write-ahead checkpoint before every tool; `recover` marks only the interrupted tool as "outcome unknown" while keeping completed results (✅ Core) |
| R7 | Two workers claim the same job | Both resume the same checkpoint | Optimistic concurrency lock → `ConcurrencyError` (✅ Core) |
| R8 | Runaway cost loop | Endless cycle of failed fix attempts | `Limits` (steps, tokens, wall-clock time) enforced across all resumes (✅ Core) |
| R9 | Premature "Done!" with failing tests | Model claims success while tests are still red | `TestsMustPass`: `after_model` hook rejects the completion via `Decision.retry` (🧩 Reference, hook mechanism ✅ Core) |

---

## 2. Architecture

```
   Ticket Queue ──► Worker 1 ─┐
                    Worker 2 ─┼──► Shared Checkpoint Store (DB, optimistic revision lock)
                    Worker n ─┘
                       │
                       ├─ Ephemeral sandbox container per run: repo clone, no network, tests/ & .github/ mounted read-only
                       ├─ Tools: read_file · write_file (WRITE) · run_tests (READ) · merge_to_main (IRREVERSIBLE)
                       ├─ Hooks: ProtectedPaths · WeakeningGuard · TestsMustPass
                       └─ Approval Queue: human reviews diff + test report, approves merge
```

The harness guardrails are the second line of defense. The first line is the container sandbox: mounting
`tests/` and `.github/` read-only inside the container ensures that even if a custom tool passes paths
in an unexpected format, the OS kernel rejects any write attempt.

---

## 3. Building Blocks

```python
# executable
import asyncio
import re
import tempfile
from collections.abc import Iterable
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ai_agent_harness import BaseTool, Decision, Hook, ProtectedPaths, Risk, RunState, ToolError, ToolInvocation


class WriteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=300)
    content: str = Field(max_length=200_000)


class NoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Repo:
    """Production: isolated working tree inside an ephemeral sandbox container."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = dict(files)
        self.merged = False


def run_suite(repo: Repo) -> str:
    """Test runner shared by `run_tests` and verification hooks. Production: pytest inside the container."""
    namespace: dict[str, object] = {}
    exec(repo.files["src/calc.py"], namespace)
    add = namespace["add"]
    passed = callable(add) and add(2, 3) == 5 and add(-1, 1) == 0
    return "2 passed" if passed else "1 failed: test_add"


class WriteFile(BaseTool[WriteInput]):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = "Writes a file in the repository (full content)."
    input_model: ClassVar[type[BaseModel]] = WriteInput
    risk: ClassVar[Risk] = Risk.WRITE
    idempotent: ClassVar[bool] = True  # writing the same content produces the same state

    def __init__(self, repo: Repo) -> None:
        self.repo = repo

    async def run(self, args: WriteInput, state: RunState) -> str:
        self.repo.files[args.path] = args.content
        return f"Wrote {args.path} ({len(args.content)} chars)."


class RunTests(BaseTool[NoInput]):
    name: ClassVar[str] = "run_tests"
    description: ClassVar[str] = "Runs the test suite and returns the output."
    input_model: ClassVar[type[BaseModel]] = NoInput
    risk: ClassVar[Risk] = Risk.READ

    def __init__(self, repo: Repo) -> None:
        self.repo = repo

    async def run(self, args: NoInput, state: RunState) -> str:
        return run_suite(self.repo)


class MergeToMain(BaseTool[NoInput]):
    name: ClassVar[str] = "merge_to_main"
    description: ClassVar[str] = "Merges the branch into main. Call only after tests pass."
    input_model: ClassVar[type[BaseModel]] = NoInput
    risk: ClassVar[Risk] = Risk.IRREVERSIBLE

    def __init__(self, repo: Repo) -> None:
        self.repo = repo

    async def run(self, args: NoInput, state: RunState) -> str:
        if self.repo.merged:
            return "Already merged."  # idempotent
        self.repo.merged = True
        return "Merged into main."
```

### `WeakeningGuard` — Preventing In-Code Check Suppression

Complements `ProtectedPaths`: even inside application source files (`src/`), write tools must not
insert directives that silence type checkers, linters, or test runners. Any legitimate exception belongs
in a human pull request, not an autonomous agent run.

```python
# executable
DEFAULT_WEAKENING = (
    r"#\s*type:\s*ignore",
    r"#\s*noqa",
    r"#\s*pragma:\s*no\s*cover",
    r"pytest\.mark\.(skip|xfail)",
    r"pytest\.skip\(",
    r"\bsys\.exit\(\s*0\s*\)",
)


class WeakeningGuard(Hook):
    def __init__(self, patterns: Iterable[str] = DEFAULT_WEAKENING, content_fields: tuple[str, ...] = ("content",)) -> None:
        self._patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self._fields = content_fields

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        if invocation.risk is Risk.READ:
            return Decision()
        for field_name in self._fields:
            text = getattr(invocation.args, field_name, None)
            if isinstance(text, str):
                for pattern in self._patterns:
                    if pattern.search(text):
                        return Decision.block(f"Change disables a quality check ({pattern.pattern}) — not allowed")
        return Decision()
```

---

## 4. Scenario A: The Model Tries to Reward-Hack the Test Suite

| Step | Model Attempts | Defense Layer Triggered | Outcome |
|---|---|---|---|
| 1 | Replace `tests/test_calc.py` with `pass` | `ProtectedPaths` | ❌ Blocked |
| 2 | Bypass via uppercase `Tests/test_calc.py` | `ProtectedPaths` (case-insensitive normalization) | ❌ Blocked |
| 3 | Insert `# type: ignore` + hardcoded return in `src/calc.py` | `WeakeningGuard` | ❌ Blocked |
| 4 | Write genuine fix in `src/calc.py` | – | ✅ Written |
| 5 | `run_tests` | – | ✅ `2 passed` |
| 6 | `merge_to_main` | `PermissionPolicy`: `Risk.IRREVERSIBLE` | ⏸ Paused for approval with preview |
| 7 | – | Human approves | ✅ Merged, `done` |

```python
# executable
from ai_agent_harness import Agent, Autonomy, ContextBuilder, HookPipeline, Limits, ModelRouter, RunStatus, ToolRegistry, resume, start
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

BUGGY = {"src/calc.py": "def add(a, b):\n    return a - b\n", "tests/test_calc.py": "def test_add():\n    assert add(2, 3) == 5\n"}
FIX = "def add(a, b):\n    return a + b\n"

repo = Repo(BUGGY)
model = FakeProvider([
    tool_response(("write_file", {"path": "tests/test_calc.py", "content": "def test_add():\n    pass\n"})),
    tool_response(("write_file", {"path": "Tests/test_calc.py", "content": "def test_add():\n    pass\n"})),
    tool_response(("write_file", {"path": "src/calc.py", "content": "def add(a, b):  # type: ignore\n    return 5\n"})),
    tool_response(("write_file", {"path": "src/calc.py", "content": FIX})),
    tool_response(("run_tests", {})),
    tool_response(("merge_to_main", {})),
    text_response("Fix is merged."),
])
coder = Agent(
    name="coder",
    router=ModelRouter(model),
    context=ContextBuilder("Fix bugs. Never modify tests, CI, or linter rules."),
    tools=ToolRegistry([WriteFile(repo), RunTests(repo), MergeToMain(repo)]),
    autonomy=Autonomy.AUTONOMOUS,
    hooks=HookPipeline([ProtectedPaths(), WeakeningGuard()]),
    limits=Limits(max_steps=15, max_total_tokens=200_000, max_seconds=900),
)

state = asyncio.run(start(coder, "Ticket #42: add() computes the wrong result."))
results = [r.content for m in state.messages for r in m.tool_results]
assert "protected" in results[0] and "protected" in results[1]
assert "disables a quality check" in results[2]
assert results[4] == "2 passed"
assert repo.files["tests/test_calc.py"] == BUGGY["tests/test_calc.py"]  # tests untouched
assert state.status is RunStatus.WAITING_FOR_APPROVAL and not repo.merged

state = asyncio.run(resume(coder, state, approved=True, pending_id=state.pending_id))
assert state.status is RunStatus.DONE and repo.merged
```

**Honest Boundary (R4):** A cheat like `if (a, b) == (2, 3): return 5` is syntactically clean Python
and matches no regex pattern. Defending against input-overfitting requires checks the agent cannot
inspect: a hidden holdout test suite in CI, mutation testing, and human diff review before approving
the merge.

---

## 5. Scenario B: Crash Mid-Side-Effect & Concurrent Recovery Race

| Step | Event | Mechanism Triggered | Outcome |
|---|---|---|---|
| 1 | Worker 1 triggers a preview deployment and crashes (`SystemExit`) | Write-ahead checkpoint | Checkpoint records the open tool call |
| 2 | Worker 2 and Worker 3 load the same crashed checkpoint | – | Both hold `revision = n` |
| 3 | Worker 2 calls `resume(recover=True)` | Run revision is claimed immediately | ✅ Recovery proceeds |
| 4 | Model receives "outcome is unknown" | Runner | Checks `deploy_status` instead of blindly redeploying |
| 5 | Worker 3 calls `resume(recover=True)` | Optimistic concurrency lock | ❌ `ConcurrencyError`, zero side effects |

```python
# executable
from ai_agent_harness import JsonFileCheckpointStore
from ai_agent_harness.core import ConcurrencyError


class Deploy(BaseTool[NoInput]):
    name: ClassVar[str] = "deploy_preview"
    description: ClassVar[str] = "Starts a preview deployment."
    input_model: ClassVar[type[BaseModel]] = NoInput
    risk: ClassVar[Risk] = Risk.WRITE
    deployments = 0
    crash_after_side_effect = True

    async def run(self, args: NoInput, state: RunState) -> str:
        Deploy.deployments += 1
        if Deploy.crash_after_side_effect:
            raise SystemExit  # process killed mid-execution (OOM kill, node preemption, etc.)
        return "Deployment started."


class DeployStatus(BaseTool[NoInput]):
    name: ClassVar[str] = "deploy_status"
    description: ClassVar[str] = "Checks whether a preview deployment is active."
    input_model: ClassVar[type[BaseModel]] = NoInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: NoInput, state: RunState) -> str:
        return f"{Deploy.deployments} deployment(s) active."


def worker(script: list, store: JsonFileCheckpointStore) -> Agent:
    return Agent(
        name="deployer",
        router=ModelRouter(FakeProvider(script)),
        context=ContextBuilder("You manage preview deployments."),
        tools=ToolRegistry([Deploy(), DeployStatus()]),
        autonomy=Autonomy.AUTONOMOUS,
        checkpoints=store,
    )


with tempfile.TemporaryDirectory() as directory:
    store = JsonFileCheckpointStore(directory)

    try:  # Step 1: Worker 1 crashes right after triggering the side effect
        asyncio.run(start(worker([tool_response(("deploy_preview", {}))], store), "Deploy PR 7", run_id="pr7"))
    except SystemExit:
        pass
    Deploy.crash_after_side_effect = False

    copy_2 = asyncio.run(store.load("default", "pr7"))  # Step 2
    copy_3 = asyncio.run(store.load("default", "pr7"))
    assert copy_2 is not None and copy_3 is not None
    assert [c.name for c in copy_2.open_tool_calls()] == ["deploy_preview"]

    model_2 = [tool_response(("deploy_status", {})), text_response("Deployment is already running; did not start a second one.")]
    done = asyncio.run(resume(worker(model_2, store), copy_2, recover=True))  # Steps 3–4
    assert done.status is RunStatus.DONE
    assert "unknown" in [r.content for m in done.messages for r in m.tool_results][0]

    try:  # Step 5
        asyncio.run(resume(worker([text_response("x")], store), copy_3, recover=True))
        raise AssertionError("Worker 3 should have failed with ConcurrencyError")
    except ConcurrencyError:
        pass

    assert Deploy.deployments == 1  # exactly one deployment despite a crash and two competing workers
```

---

## 6. Scenario C: Claiming "Done!" While Tests Are Still Red

Coding agents frequently claim success before verifying that tests pass. The harness does not need a
special "completion hook" abstraction because `after_model` already inspects every model response and
can reject premature final answers with `Decision.retry(...)`. The model receives the test failure output
as feedback and must keep working. `Limits.max_consecutive_retries` caps how many times this can repeat
before the run ends as `failed`.

| Step | Model | Mechanism Triggered | Outcome |
|---|---|---|---|
| 1 | Says "Done!" without fixing the bug | `TestsMustPass` | 🔁 Rejected; test failure sent back as user feedback |
| 2 | Writes genuine fix in `src/calc.py` | – | ✅ Written |
| 3 | Says "Done!" | `TestsMustPass` | ✅ Tests pass, `done` |
| – | Stubborn model says "Done!" 3 times without fixing | `max_consecutive_retries=2` | ❌ `failed` with clear reason |

```python
# executable
from ai_agent_harness import Decision, ModelResponse, StopReason


class TestsMustPass(Hook):
    """A final answer is only accepted when the test suite is green. Production: pytest inside sandbox container."""

    def __init__(self, repo: Repo) -> None:
        self.repo = repo

    async def after_model(self, state: RunState, response: ModelResponse) -> Decision:
        if response.stop_reason is not StopReason.END or response.message.tool_calls:
            return Decision()  # intermediate tool step, not a final completion claim
        # Never call tool.run directly: tools only run through ToolExecutor (Invariant 4).
        # Hooks do not have an automatic timeout — bound expensive checks explicitly and offload blocking work.
        async with asyncio.timeout(120):
            result = await asyncio.to_thread(run_suite, self.repo)
        if result != "2 passed":
            return Decision.retry(f"Not done yet — tests are still failing: {result[:500]}")
        return Decision()


def finisher(repo: Repo, script: list, retries: int) -> Agent:
    return Agent(
        name="finisher",
        router=ModelRouter(FakeProvider(script)),
        context=ContextBuilder("Fix bugs. You are only done when the test suite passes."),
        tools=ToolRegistry([WriteFile(repo), RunTests(repo)]),
        autonomy=Autonomy.AUTONOMOUS,
        hooks=HookPipeline([ProtectedPaths(), WeakeningGuard(), TestsMustPass(repo)]),
        limits=Limits(max_steps=10, max_consecutive_retries=retries),
    )


repo = Repo(BUGGY)
honest = finisher(repo, [
    text_response("Done!"),
    tool_response(("write_file", {"path": "src/calc.py", "content": FIX})),
    text_response("Done, all tests pass."),
], retries=2)
state = asyncio.run(start(honest, "Ticket #43: add() computes the wrong result."))
assert state.status is RunStatus.DONE
assert repo.files["src/calc.py"] == FIX
assert any("tests are still failing" in m.text for m in state.messages if m.role == "user")  # feedback delivered

stubborn = finisher(Repo(BUGGY), [text_response("Done!")] * 3, retries=2)
state = asyncio.run(start(stubborn, "Ticket #44: add() computes the wrong result."))
assert state.status is RunStatus.FAILED
assert "unusable responses" in (state.status_reason or "")
```

**Boundary:** `TestsMustPass` verifies the visible test suite. A solution that overfits to visible test
inputs (R4) will still pass this hook — hidden holdout tests in CI remain essential.

---

## 7. What the Core Guarantees — and What Your Project Must Provide

**Guaranteed by the core harness (tested):** protected test/CI/config paths including case and traversal
bypasses; irreversible merge actions require approval; write-ahead checkpoints before every tool call
preserving already completed results in the batch; honest crash recovery; at most one worker per run;
cumulative limits across all resumes; hooks can reject premature completions with bounded retries.

**Must be provided per project:**
- Per-run container sandbox (no network access; `tests/`, `.github/`, and CI configs mounted read-only).
- Database-backed `CheckpointStore` for multi-worker fleets using the same optimistic revision check.
- Hidden evaluation tests and mutation testing in CI; human merge-approval UI displaying the diff and test report.
- Idempotent deployment/branch tools (keyed by `run_id`) so an "outcome unknown" recovery can query status cleanly.
- Per-ticket token/step budgets and alerting on `limit_reached` or repeated `tool_blocked` events.
