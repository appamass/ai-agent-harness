"""Documentation tests: code blocks in README.md, docs/guide.md and the ai-kit recipes are executed
so documentation never drifts from the code.

Only ```python blocks whose first line is `# executable` are executed. All blocks in a file
run sequentially in the same module namespace — so a scenario can be built up step by step and
verified with `assert` statements at the end.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from .conftest import REPO_ROOT

DOC_FILES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "guide.md",
    REPO_ROOT / "ai-kit" / "skills" / "ai-agent-harness" / "references" / "adopt-in-x.md",
)
BLOCK = re.compile(r"^```python\n(# executable\n.*?)^```", re.MULTILINE | re.DOTALL)


def test_every_doc_file_has_executable_blocks() -> None:
    for path in DOC_FILES:
        assert BLOCK.findall(path.read_text(encoding="utf-8")), f"{path.name} contains no executable block"


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.stem)
def test_doc_file_runs(path: Path) -> None:
    # Real module so dataclasses/Pydantic can resolve annotations just like in normal code.
    module = types.ModuleType(f"doc_{path.stem.replace('-', '_')}")
    sys.modules[module.__name__] = module
    try:
        for index, block in enumerate(BLOCK.findall(path.read_text(encoding="utf-8")), 1):
            code = compile(block, f"{path.name}#block{index}", "exec", dont_inherit=True)
            try:
                exec(code, module.__dict__)
            except ModuleNotFoundError as exc:
                raise AssertionError(
                    f"{path.name} block {index} needs a module that is not installed: {exc.name}"
                ) from exc
    finally:
        sys.modules.pop(module.__name__, None)


# ------------------------------------------------------------------ The README demo transcript is real

RUN_ID = re.compile(r"[0-9a-f]{32}")


def _normalized(text: str) -> list[str]:
    return [RUN_ID.sub("<id>", line.rstrip()) for line in text.splitlines() if line.strip()]


def test_the_readme_demo_transcript_matches_the_real_demo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from .conftest import load_example

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"```text\n(.*?)```", readme, re.DOTALL)
    assert match is not None, "README.md has no ```text block with the demo transcript"
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    example = load_example("repo_assistant")
    assert example.main(["--demo", "--notes-dir", str(tmp_path / "n"), "--checkpoints-dir", str(tmp_path / "c")]) == 0
    real = _normalized(capsys.readouterr().out)
    shown = _normalized(match.group(1))
    assert shown, "the transcript is empty"
    # Every line of the README transcript is a line of the real demo, in the same order (the README may shorten it)
    position = 0
    for line in shown:
        assert line in real[position:], f"README transcript line not in the real demo output: {line!r}"
        position = real.index(line, position) + 1
