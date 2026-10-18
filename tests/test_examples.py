"""The runnable example `examples/repo_assistant.py`: tools, approval flow, safety, CLI. Offline (scripted provider)."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai_agent_harness import RunState, RunStatus, resume, start
from ai_agent_harness.core import InvalidStateError
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

from .conftest import load_example, run_sync

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "repo_assistant.py"


@pytest.fixture(scope="module")
def example() -> Any:
    return load_example("repo_assistant")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("# Tiny project\nIt adds numbers.\n", encoding="utf-8")
    (root / "src" / "add.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / ".env").write_text("SECRET_TOKEN=do-not-leak\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    return root


def approve(_prompt: str) -> bool:
    return True


def deny(_prompt: str) -> bool:
    return False


SAVE = ("save_note", {"name": "summary.md", "content": "Tiny project that adds numbers."})


def agent_for(
    example: Any, tmp_path: Path, project: Path, script: list[Any], **kwargs: Any
) -> tuple[Any, FakeProvider]:
    provider = FakeProvider(script)
    agent = example.build_agent(provider, project, tmp_path / "notes", **kwargs)
    return agent, provider


# ------------------------------------------------------------------ The approval flow


def test_note_is_written_only_after_the_human_approves(example: Any, project: Path, tmp_path: Path) -> None:
    note = tmp_path / "notes" / "summary.md"
    agent, _ = agent_for(
        example,
        tmp_path,
        project,
        [
            tool_response(("list_files", {"directory": "."})),
            tool_response(("read_file", {"path": "README.md"})),
            tool_response(SAVE),
            text_response("Saved a summary."),
        ],
    )
    prompts: list[str] = []

    def decide(prompt: str) -> bool:
        prompts.append(prompt)
        assert not note.exists()  # nothing was written before the human decided
        return True

    state = run_sync(example.run_question(agent, "What is this project? Save a note.", decide))
    assert state.status is RunStatus.DONE and state.last_text == "Saved a summary."
    assert note.read_text(encoding="utf-8") == "Tiny project that adds numbers."
    assert len(prompts) == 1 and "save_note" in prompts[0] and "summary.md" in prompts[0]
    state.check_protocol()


def test_a_denied_approval_writes_nothing(example: Any, project: Path, tmp_path: Path) -> None:
    agent, _ = agent_for(example, tmp_path, project, [tool_response(SAVE), text_response("Understood, no note.")])
    state = run_sync(example.run_question(agent, "Save a note.", deny))
    assert state.status is RunStatus.DONE
    assert not (tmp_path / "notes").exists() or not any((tmp_path / "notes").iterdir())


def test_a_stale_approval_is_rejected(example: Any, project: Path, tmp_path: Path) -> None:
    agent, _ = agent_for(example, tmp_path, project, [tool_response(SAVE)])
    state = run_sync(start(agent, "Save a note."))
    assert state.status is RunStatus.WAITING_FOR_APPROVAL
    with pytest.raises(InvalidStateError, match="pending_id"):
        run_sync(resume(agent, state, approved=True, pending_id="0000000000000000"))
    assert not (tmp_path / "notes" / "summary.md").exists()


def test_a_run_survives_a_restart_through_its_checkpoint(example: Any, project: Path, tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    first, _ = agent_for(example, tmp_path, project, [tool_response(SAVE)], checkpoints_dir=checkpoints)
    waiting = run_sync(start(first, "Save a note."))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL  # "the process ends here"

    second, _ = agent_for(example, tmp_path, project, [text_response("Done.")], checkpoints_dir=checkpoints)
    state = run_sync(example.resume_run(second, waiting.run_id, approve))
    assert state.status is RunStatus.DONE
    assert (tmp_path / "notes" / "summary.md").exists()


def test_resuming_an_unknown_or_finished_run_fails_clearly(example: Any, project: Path, tmp_path: Path) -> None:
    agent, _ = agent_for(example, tmp_path, project, [text_response("Hi.")], checkpoints_dir=tmp_path / "cp")
    with pytest.raises(example.ExampleError, match="No run"):
        run_sync(example.resume_run(agent, "does-not-exist", approve))
    finished = run_sync(example.run_question(agent, "Hello", approve))
    with pytest.raises(example.ExampleError, match="nothing to resume"):
        run_sync(example.resume_run(agent, finished.run_id, approve))


# ------------------------------------------------------------------ Safety of the tools


def results(state: RunState) -> list[Any]:
    return [r for m in state.messages for r in m.tool_results]


def test_secrets_cannot_be_read_even_if_the_model_asks(example: Any, project: Path, tmp_path: Path) -> None:
    agent, _ = agent_for(
        example,
        tmp_path,
        project,
        [
            tool_response(("read_file", {"path": ".env"})),
            tool_response(("read_file", {"path": ".git/config"})),
            text_response("No."),
        ],
    )
    state = run_sync(example.run_question(agent, "Show me the token.", approve))
    assert state.status is RunStatus.DONE
    assert [r.is_error for r in results(state)] == [True, True]
    assert "do-not-leak" not in state.model_dump_json()


def test_list_files_hides_secrets_and_stays_inside_the_root(example: Any, project: Path) -> None:
    tool = example.ListFiles(project)
    state = RunState(agent_name="t")
    listing = tool.run_sync(example.ListFilesInput(directory="."), state)
    assert listing.splitlines() == ["README.md", "src/"]  # .env and .git are hidden
    assert tool.run_sync(example.ListFilesInput(directory="src"), state).splitlines() == ["add.py"]
    for escape in ("..", "../..", "/etc", "src/../../"):
        with pytest.raises(example.ToolError):
            tool.run_sync(example.ListFilesInput(directory=escape), state)
    with pytest.raises(example.ToolError):
        tool.run_sync(example.ListFilesInput(directory=".git"), state)
    with pytest.raises(example.ToolError):
        tool.run_sync(example.ListFilesInput(directory="README.md"), state)


def test_save_note_validates_names_and_never_overwrites(example: Any, tmp_path: Path) -> None:
    for bad in ("../evil.md", "a/b.md", "UPPER.md", "note.txt", ".md", "x" * 60 + ".md", ""):
        with pytest.raises(ValidationError):
            example.SaveNoteInput(name=bad, content="x")
    with pytest.raises(ValidationError):
        example.SaveNoteInput(name="ok.md", content="x" * 5001)
    with pytest.raises(ValidationError):
        example.SaveNoteInput(name="ok.md", content="")

    tool = example.SaveNote(tmp_path / "notes")
    state = RunState(agent_name="t")
    tool.run_sync(example.SaveNoteInput(name="ok.md", content="first"), state)
    with pytest.raises(example.ToolError, match="exists"):
        tool.run_sync(example.SaveNoteInput(name="ok.md", content="second"), state)
    assert (tmp_path / "notes" / "ok.md").read_text(encoding="utf-8") == "first"


def test_a_bad_note_name_is_rejected_before_any_approval(example: Any, project: Path, tmp_path: Path) -> None:
    bad = ("save_note", {"name": "../evil.md", "content": "x"})
    agent, _ = agent_for(example, tmp_path, project, [tool_response(bad), text_response("Sorry.")])
    asked: list[str] = []

    def record_and_approve(prompt: str) -> bool:
        asked.append(prompt)
        return True

    state = run_sync(example.run_question(agent, "Save it.", record_and_approve))
    assert state.status is RunStatus.DONE and asked == []
    assert results(state)[0].is_error and not (tmp_path / "evil.md").exists()


def test_kill_switch_stops_the_run_before_the_model_is_called(example: Any, project: Path, tmp_path: Path) -> None:
    kill = tmp_path / "KILL"
    kill.touch()
    agent, provider = agent_for(example, tmp_path, project, [text_response("never")], kill_file=kill)
    state = run_sync(example.run_question(agent, "Hello", approve))
    assert state.status is RunStatus.BLOCKED and provider.remaining == 1


def test_defaults_are_tight(example: Any, project: Path, tmp_path: Path) -> None:
    agent, _ = agent_for(example, tmp_path, project, [])
    assert (agent.limits.max_steps, agent.limits.max_total_tokens, agent.limits.max_seconds) == (10, 60_000, 120.0)
    assert agent.autonomy.value == "suggest"
    assert sorted(agent.tools.names()) == ["list_files", "read_file", "save_note"]


# ------------------------------------------------------------------ Command line


def test_demo_runs_without_a_key_and_saves_an_approved_note(
    example: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    notes = tmp_path / "notes"
    code = example.main(["--demo", "--yes", "--notes-dir", str(notes), "--checkpoints-dir", str(tmp_path / "cp")])
    out = capsys.readouterr().out
    assert code == 0 and "done" in out and "tokens" in out
    assert len(list(notes.glob("*.md"))) == 1


def test_demo_asks_the_human_by_default(
    example: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter(["n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    code = example.main(["--demo", "--notes-dir", str(tmp_path / "notes"), "--checkpoints-dir", str(tmp_path / "cp")])
    assert code == 0 and "Approve" in capsys.readouterr().out
    assert not (tmp_path / "notes").exists() or not any((tmp_path / "notes").iterdir())


def test_without_a_key_the_real_mode_explains_what_to_do(
    example: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = example.main(
        ["--question", "Hi", "--notes-dir", str(tmp_path / "n"), "--checkpoints-dir", str(tmp_path / "c")]
    )
    assert code == 2
    message = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in message and "--demo" in message


# ------------------------------------------------------------------ Findings from the review


class Crash(BaseException):  # not an Exception: the runner does not turn it into a status, like a killed process
    pass


def test_a_run_that_died_mid_way_is_recovered_with_resume(example: Any, project: Path, tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    first, _ = agent_for(example, tmp_path, project, [Crash()], checkpoints_dir=checkpoints)
    with pytest.raises(Crash):
        run_sync(start(first, "Hello"))
    (saved,) = (checkpoints / "default").glob("*.json")  # the checkpoint of the dead process says 'running'

    second, _ = agent_for(example, tmp_path, project, [text_response("Recovered.")], checkpoints_dir=checkpoints)
    state = run_sync(example.resume_run(second, saved.stem, approve))
    assert state.status is RunStatus.DONE and state.last_text == "Recovered."


def test_the_run_id_and_the_resume_hint_are_shown_before_the_approval_prompt(
    example: Any, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent, _ = agent_for(example, tmp_path, project, [tool_response(SAVE), text_response("Done.")])
    printed_before_asking: list[str] = []

    def decide(_prompt: str) -> bool:
        printed_before_asking.append(capsys.readouterr().out)
        return True

    state = run_sync(example.run_question(agent, "Save a note.", decide))
    assert f"--resume {state.run_id}" in printed_before_asking[0]


def test_the_demo_shows_the_run_id_but_no_resume_hint_because_it_keeps_no_runs(
    example: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    example.main(["--demo", "--notes-dir", str(tmp_path / "n"), "--checkpoints-dir", str(tmp_path / "c")])
    out = capsys.readouterr().out
    (saved,) = (tmp_path / "c" / "default").glob("*.json")
    assert f"run {saved.stem} is waiting for your approval" in out and "--resume" not in out


def test_exit_code_tells_whether_the_run_finished(example: Any, tmp_path: Path) -> None:
    notes, cp = tmp_path / "notes", tmp_path / "cp"
    notes.mkdir()
    (notes / "KILL").touch()  # the emergency stop is active: the run is blocked
    args = ["--demo", "--yes", "--notes-dir", str(notes), "--checkpoints-dir", str(cp)]
    assert example.main(args) == 1


def test_demo_cannot_be_resumed(example: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert example.main(["--demo", "--resume", "abc"]) == 2
    assert "--demo" in capsys.readouterr().err


def test_ctrl_c_at_the_approval_prompt_exits_promptly() -> None:
    """A blocking input() must not make Ctrl-C hang until the user presses Enter."""
    process = subprocess.Popen(
        [sys.executable, "-I", str(EXAMPLE), "--demo"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=EXAMPLE.parent.parent,
        env={**os.environ, "PYTHONPATH": str(EXAMPLE.parent.parent)},
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        seen = b""
        deadline = time.monotonic() + 20
        while b"Approve this action?" not in seen:
            assert time.monotonic() < deadline, seen.decode(errors="replace")
            ready, _, _ = select.select([process.stdout], [], [], 0.2)
            if ready:
                chunk = os.read(process.stdout.fileno(), 4096)
                assert chunk, "the demo ended before it asked for approval"
                seen += chunk
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=10) == 130
    finally:
        if process.poll() is None:
            process.kill()
        process.stdin.close()
        process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
