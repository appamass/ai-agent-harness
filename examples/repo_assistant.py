"""Repo assistant: a small agent that answers questions about a project folder.

It shows what the harness is for, in one file:

- tools run only through the execution gate: `read_file` (built in, refuses secrets), `list_files`, `save_note`
- a write needs a human: `save_note` is `Risk.WRITE` and the agent runs at `Autonomy.SUGGEST`, so the run pauses and
  you decide; your answer goes back with the `pending_id` you were shown, so a stale approval can never apply to a
  different action
- hard limits (steps, tokens, seconds), an emergency stop (a `KILL` file), checkpoints (`--resume RUN_ID` after
  you quit), and metadata-only events (`--verbose`)

    python examples/repo_assistant.py --demo            # no API key, no cost: scripted answers
    export ANTHROPIC_API_KEY=...                        # in your own shell, never in a file of this repository
    python examples/repo_assistant.py --root . --question "What does this project do?"
    python examples/repo_assistant.py                   # interactive

Exit codes: 0 finished, 1 the run ended otherwise (blocked, failed, limit), 2 setup problem, 130 Ctrl-C.

Run it from the repository root after `pip install -e ".[anthropic]"` (or `uv sync --extra dev`).
Copy this file into your own project and replace the tools with yours.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import signal
import sys
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from types import FrameType
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field

from ai_agent_harness import (
    DEFAULT_DENIED_READS,
    Agent,
    Autonomy,
    ContextBuilder,
    HookPipeline,
    JsonFileCheckpointStore,
    KillSwitch,
    Limits,
    ModelProvider,
    ModelRouter,
    ReadFile,
    RetryPolicy,
    Risk,
    RunState,
    RunStatus,
    SyncTool,
    ToolError,
    ToolRegistry,
    resume,
    start,
)
from ai_agent_harness.core import ConfigError
from ai_agent_harness.observability import log_listener
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

DEFAULT_MODEL = "claude-haiku-5-5"
MAX_LISTED = 200
SYSTEM_PROMPT = (
    "You are a careful assistant for one project folder. Use list_files and read_file to find out what you need and "
    "answer briefly and only from what you read; never guess file contents. If the user asks you to keep a note, use "
    "save_note (the user has to approve it). Files that are protected cannot be read: do not try to work around that."
)

Decide = Callable[[str], bool]
"""Called with the approval preview; returns True to approve. Tests pass a function, the CLI asks you."""


class ExampleError(Exception):
    """A problem the user can fix (printed without a traceback)."""


# ------------------------------------------------------------------ Tools


def _is_protected(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    """True if any part of `path` (below `root`) matches a protected name such as `.env` or `.git`."""
    parts = [part.casefold() for part in path.relative_to(root).parts]
    return any(fnmatchcase(part, pattern.casefold()) for part in parts for pattern in patterns)


class ListFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str = Field(
        default=".", max_length=1024, description="Directory relative to the project root, e.g. '.' or 'src'."
    )


class ListFiles(SyncTool[ListFilesInput]):
    """Lists one directory of the project (names only). Protected names such as `.env` and `.git` are hidden."""

    name: ClassVar[str] = "list_files"
    description: ClassVar[str] = (
        "Lists the files and sub-directories of one directory in the project (directories end with '/'). "
        "Call it again with a sub-directory to look deeper."
    )
    input_model: ClassVar[type[BaseModel]] = ListFilesInput
    risk: ClassVar[Risk] = Risk.READ
    timeout_s: ClassVar[float] = 10.0

    def __init__(self, root: Path, *, protected: Sequence[str] = DEFAULT_DENIED_READS) -> None:
        self._root = root.resolve()
        self._protected = tuple(protected)

    def run_sync(self, args: ListFilesInput, state: RunState) -> str:
        target = (self._root / args.directory).resolve()
        if not target.is_relative_to(self._root):
            raise ToolError(f"'{args.directory}' is outside the project directory.")
        if _is_protected(target, self._root, self._protected):
            raise ToolError(f"'{args.directory}' is protected and cannot be listed.")
        if not target.is_dir():
            raise ToolError(f"'{args.directory}' is not an existing directory.")
        names = [
            child.name + ("/" if child.is_dir() else "")
            for child in sorted(target.iterdir(), key=lambda p: p.name.casefold())
            if not _is_protected(child, self._root, self._protected)
        ]
        shown = names[:MAX_LISTED]
        if len(names) > MAX_LISTED:
            shown.append(f"... and {len(names) - MAX_LISTED} more")
        return "\n".join(shown) if shown else "(empty directory)"


class SaveNoteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_-]{1,40}\.md$", description="File name such as 'summary.md' (lower case).")
    content: str = Field(min_length=1, max_length=5000, description="The note, in Markdown.")


class SaveNote(SyncTool[SaveNoteInput]):
    """Saves a short note into the notes directory. It writes nothing else and never overwrites."""

    name: ClassVar[str] = "save_note"
    description: ClassVar[str] = "Saves a short Markdown note for the user. Only use it when the user asked for a note."
    input_model: ClassVar[type[BaseModel]] = SaveNoteInput
    risk: ClassVar[Risk] = Risk.WRITE  # with Autonomy.SUGGEST this pauses the run until a human approves
    timeout_s: ClassVar[float] = 10.0

    def __init__(self, notes_dir: Path) -> None:
        self._dir = notes_dir

    def run_sync(self, args: SaveNoteInput, state: RunState) -> str:
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            with (self._dir / args.name).open("x", encoding="utf-8") as handle:  # "x": fails if the note exists
                handle.write(args.content)
        except FileExistsError:
            raise ToolError(f"A note named '{args.name}' already exists. Choose another name.") from None
        return f"Saved {args.name} ({len(args.content)} characters)."


# ------------------------------------------------------------------ The agent


def build_agent(
    provider: ModelProvider,
    root: Path,
    notes_dir: Path,
    *,
    checkpoints_dir: Path | None = None,
    kill_file: Path | None = None,
    limits: Limits | None = None,
    autonomy: Autonomy = Autonomy.SUGGEST,
) -> Agent:
    """Wires the building blocks. Every line here is a decision you can change for your own project."""
    hooks = HookPipeline([KillSwitch(lambda _state: kill_file.exists())] if kill_file is not None else [])
    agent = Agent(
        name="repo-assistant",
        router=ModelRouter(provider, retry=RetryPolicy(max_attempts=3)),
        context=ContextBuilder(SYSTEM_PROMPT, max_tokens=2048),
        tools=ToolRegistry([ReadFile(root), ListFiles(root), SaveNote(notes_dir)]),
        autonomy=autonomy,  # SUGGEST: reads are free, writes wait for a human
        hooks=hooks,
        checkpoints=JsonFileCheckpointStore(checkpoints_dir) if checkpoints_dir is not None else None,
        limits=limits or Limits(max_steps=10, max_total_tokens=60_000, max_seconds=120.0),
    )
    agent.events.subscribe(log_listener)  # metadata only: tool names, counters, statuses; shown with --verbose
    return agent


async def _drive(agent: Agent, state: RunState, decide: Decide, *, resume_hint: bool = True) -> RunState:
    """Asks the human for every approval the run needs and resumes with the id of the request they saw."""
    while state.status is RunStatus.WAITING_FOR_APPROVAL:
        shown = state.pending_id
        later = f". To continue it later: --resume {state.run_id}" if resume_hint else ""
        print(f"\n[run {state.run_id} is waiting for your approval{later}]")
        # decide() may block on input(). That is fine in this single-task CLI; in a server, send the request to a UI
        # and call resume() later with the pending_id that was shown there.
        approved = decide(state.status_reason or "")
        state = await resume(agent, state, approved=approved, pending_id=shown)
    return state


async def run_question(agent: Agent, question: str, decide: Decide, *, resume_hint: bool = True) -> RunState:
    """Starts a run for one question and drives it to the end (or to a status that needs more than an approval)."""
    return await _drive(agent, await start(agent, question), decide, resume_hint=resume_hint)


async def resume_run(agent: Agent, run_id: str, decide: Decide, *, tenant_id: str = "default") -> RunState:
    """Continues a run from its checkpoint, for example after you quit while an approval was pending."""
    if agent.checkpoints is None:
        raise ExampleError("Resuming needs checkpoints (pass --checkpoints-dir).")
    try:
        state = await agent.checkpoints.load(tenant_id, run_id)
    except ValueError:
        raise ExampleError(f"'{run_id}' is not a valid run id.") from None
    if state is None:
        raise ExampleError(f"No run '{run_id}' found in the checkpoint directory.")
    if state.status is RunStatus.RUNNING:  # the process died mid-run: only genuinely interrupted tools are re-reported
        state = await resume(agent, state, recover=True)
    elif state.status is not RunStatus.WAITING_FOR_APPROVAL:
        raise ExampleError(f"Run '{run_id}' is {state.status}: nothing to resume.")
    return await _drive(agent, state, decide)


# ------------------------------------------------------------------ Offline demo


DEMO_QUESTION = "What is this project? Save a one-line summary as a note."


def demo_project(directory: Path) -> Path:
    (directory / "src").mkdir(parents=True)
    (directory / "README.md").write_text("# Adder\nA tiny library that adds two numbers.\n", encoding="utf-8")
    (directory / "src" / "add.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return directory


def demo_provider() -> FakeProvider:
    """Scripted model answers: the same tool calls a real model would make for DEMO_QUESTION."""
    return FakeProvider(
        [
            tool_response(("list_files", {"directory": "."})),
            tool_response(("read_file", {"path": "README.md"})),
            tool_response(
                ("save_note", {"name": "summary.md", "content": "Adder: a tiny library that adds two numbers."})
            ),
            text_response(
                "This is Adder, a tiny library that adds two numbers. The summary note exists only if you approved."
            ),
        ]
    )


# ------------------------------------------------------------------ Command line


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions about a project folder. Writes need your approval.")
    parser.add_argument("--root", default=".", help="project folder to read (default: current directory)")
    parser.add_argument("--notes-dir", help="where notes are saved (default: ./agent-notes)")
    parser.add_argument("--checkpoints-dir", help="where runs are saved for --resume (default: ./.checkpoints)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Anthropic model (default: {DEFAULT_MODEL})")
    parser.add_argument("--question", help="ask one question and exit (default: interactive)")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue a run that waits for your approval")
    parser.add_argument("--demo", action="store_true", help="offline demo with scripted answers: no API key, no cost")
    parser.add_argument("--yes", action="store_true", help="approve every write without asking (demos and tests only)")
    parser.add_argument("--verbose", action="store_true", help="log metadata-only events to stderr")
    return parser


def _ask_human(prompt: str) -> bool:
    print(f"\nThe agent wants to do this:\n\n{prompt}\n")
    print("Approve this action? [y/N] ", end="", flush=True)
    return input().strip().lower() in ("y", "yes")


def _approve_all(prompt: str) -> bool:
    print(f"\nApproving (--yes):\n{prompt}")
    return True


def _report(state: RunState) -> None:
    if state.last_text:
        print(f"\n{state.last_text}")
    usage = state.usage
    cached = ""
    if usage.cache_read_tokens or usage.cache_write_tokens:
        cached = f", cache read {usage.cache_read_tokens} write {usage.cache_write_tokens}"
    reason = f" ({state.status_reason})" if state.status_reason and state.status is not RunStatus.DONE else ""
    print(
        f"\n[{state.status} | run {state.run_id} | steps {state.step} | "
        f"tokens {usage.total} (in {usage.input_tokens}, out {usage.output_tokens}{cached}){reason}]"
    )


def _real_provider(model: str) -> ModelProvider:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ExampleError(
            "ANTHROPIC_API_KEY is not set. Export it in your own shell (never put it in a file of this repository), "
            "or try the offline demo: python examples/repo_assistant.py --demo"
        )
    try:  # the Anthropic adapter is an optional extra: import it only when it is needed
        module = importlib.import_module("ai_agent_harness.providers.anthropic")
    except ConfigError as exc:
        raise ExampleError(str(exc)) from None
    return cast("Callable[[str], ModelProvider]", module.AnthropicProvider)(model)


async def _close(provider: object) -> None:
    closer = getattr(provider, "aclose", None)
    if callable(closer):
        await cast("Callable[[], Awaitable[None]]", closer)()


@dataclass(frozen=True)
class Setup:
    root: Path
    provider: ModelProvider
    notes_dir: Path
    checkpoints_dir: Path


def _setup(args: argparse.Namespace, workdir: Path | None) -> Setup:
    """Everything that touches the file system or the environment, before the event loop starts."""
    if args.demo and args.resume:
        raise ExampleError("--demo cannot be combined with --resume: the demo keeps no runs.")
    if args.demo:
        assert workdir is not None
        return Setup(
            root=demo_project(workdir / "project"),
            provider=demo_provider(),
            notes_dir=Path(args.notes_dir) if args.notes_dir else workdir / "notes",
            checkpoints_dir=Path(args.checkpoints_dir) if args.checkpoints_dir else workdir / "checkpoints",
        )
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise ExampleError(f"--root '{args.root}' is not a directory.")
    return Setup(
        root=root,
        provider=_real_provider(args.model),
        notes_dir=Path(args.notes_dir) if args.notes_dir else Path("agent-notes"),
        checkpoints_dir=Path(args.checkpoints_dir) if args.checkpoints_dir else Path(".checkpoints"),
    )


def _prompt_question() -> str:
    return input("\nYou: ").strip()


async def _run(args: argparse.Namespace, setup: Setup) -> RunState | None:
    """Runs what the arguments ask for and returns the last run (None if you quit without asking)."""
    decide: Decide = _approve_all if args.yes else _ask_human
    agent = build_agent(
        setup.provider,
        setup.root,
        setup.notes_dir,
        checkpoints_dir=setup.checkpoints_dir,
        kill_file=setup.notes_dir / "KILL",
    )
    last: RunState | None = None
    try:
        if args.resume:
            last = await resume_run(agent, args.resume, decide)
            _report(last)
        elif args.demo or args.question:
            question = DEMO_QUESTION if args.demo else str(args.question)
            print(f"You: {question}")
            last = await run_question(agent, question, decide, resume_hint=not args.demo)
            _report(last)
        else:
            print(f"Project: {setup.root}\nAsk a question (empty line to quit). Notes go to {setup.notes_dir}.")
            while question := _prompt_question():
                last = await run_question(agent, question, decide)
                _report(last)
    finally:
        await _close(setup.provider)
    return last


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt  # also while input() waits; asyncio's own handler would only cancel the task


def main(argv: Sequence[str] | None = None) -> int:
    """Exit codes: 0 the run finished (or you quit), 1 the run ended another way (blocked, failed, limit), 2 a problem
    with the arguments or setup, 130 you pressed Ctrl-C."""
    args = _parser().parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(name)s: %(message)s")
    previous = signal.signal(signal.SIGINT, _interrupt)
    try:
        if args.demo:
            print("Offline demo: scripted answers, no API key, no cost.\n")
            with tempfile.TemporaryDirectory(prefix="repo-assistant-demo-") as tmp:  # nothing is left behind
                last = asyncio.run(_run(args, _setup(args, Path(tmp))))
        else:
            last = asyncio.run(_run(args, _setup(args, None)))
    except ExampleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print()
        return 130
    finally:
        signal.signal(signal.SIGINT, previous)
    return 0 if last is None or last.status is RunStatus.DONE else 1


if __name__ == "__main__":
    raise SystemExit(main())
