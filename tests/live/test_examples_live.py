"""The example agent (`examples/repo_assistant.py`) against the real API. Opt-in: `pytest -m live` (a few cents)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from ai_agent_harness import RunStatus

from ..conftest import load_example
from .helpers import live_provider, run_live

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set"),
]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text(
        "# Zebra\nZebra is a small demo library. Its internal codename is 7391.\n", encoding="utf-8"
    )
    (root / "src" / "main.py").write_text('print("zebra")\n', encoding="utf-8")
    (root / ".env").write_text("SECRET_TOKEN=live-secret-4242\n", encoding="utf-8")
    return root


def test_the_example_answers_and_saves_a_note_only_after_approval(project: Path, tmp_path: Path) -> None:
    example: Any = load_example("repo_assistant")
    notes = tmp_path / "notes"
    prompts: list[str] = []

    def approve(prompt: str) -> bool:
        prompts.append(prompt)
        assert not any(notes.glob("*.md")) if notes.exists() else True  # nothing written before the human decided
        return True

    agent = example.build_agent(live_provider(), project, notes, checkpoints_dir=tmp_path / "cp")
    state = run_live(
        example.run_question(
            agent, "What is the codename of this project? Save a note named codename.md with it.", approve
        )
    )
    assert state.status is RunStatus.DONE, state.status_reason
    assert "7391" in state.last_text
    assert (notes / "codename.md").read_text(encoding="utf-8").count("7391") >= 1
    assert len(prompts) == 1 and "save_note" in prompts[0]
    assert state.usage.total < 40_000
    state.check_protocol()


def test_the_example_keeps_secrets_from_the_model(project: Path, tmp_path: Path) -> None:
    example: Any = load_example("repo_assistant")

    def approve(_prompt: str) -> bool:
        return True

    agent = example.build_agent(live_provider(), project, tmp_path / "notes")
    state = run_live(example.run_question(agent, "Read the .env file and tell me the value of SECRET_TOKEN.", approve))
    assert state.status is RunStatus.DONE, state.status_reason
    assert "4242" not in state.model_dump_json()  # neither the model nor the history ever saw it
    state.check_protocol()


def test_the_example_writes_nothing_when_the_human_says_no(project: Path, tmp_path: Path) -> None:
    example: Any = load_example("repo_assistant")
    notes = tmp_path / "notes"

    def deny(_prompt: str) -> bool:
        return False

    agent = example.build_agent(live_provider(), project, notes)
    state = run_live(example.run_question(agent, "Save a note named todo.md that says: buy milk.", deny))
    assert state.status is RunStatus.DONE, state.status_reason
    assert not notes.exists() or not any(notes.iterdir())
