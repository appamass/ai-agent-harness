"""Architectural invariants verified statically (AGENTS.md)."""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "ai_agent_harness"
CORE = HARNESS / "core"
ALLOWED_THIRD_PARTY = {"pydantic"}

# Explicitly approved exceptions (AGENTS.md: no new suppressions without human approval)
APPROVED_SUPPRESSIONS: dict[str, int] = {
    "ai_agent_harness/core/types.py": 1,  # TRY004 on Pydantic validator
    "ai_agent_harness/guardrails.py": 2,  # reportUnknownVariableType on arbitrary JSON structures
    "ai_agent_harness/persistence.py": 3,  # Windows fallback without fcntl
    "ai_agent_harness/providers/router.py": 1,  # defensive fallback in complete()
    "tests/test_config_and_misc.py": 1,
    "tests/test_edge_cases.py": 1,
    "tests/test_resume.py": 1,
    "tests/test_router.py": 1,
}
SUPPRESSION_TAGS = ("type: ignore", "pyright: ignore", "noqa", "pragma: no cover")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_core_imports_only_pydantic_and_stdlib() -> None:
    """Invariant 7: `ai_agent_harness/core/` depends on nothing except Pydantic and the standard library."""
    files = sorted(CORE.glob("*.py"))
    assert files
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 1, f"{path.name} imports from a parent package"
        foreign = _imports(path) - set(sys.stdlib_module_names) - ALLOWED_THIRD_PARTY - {"__future__"}
        assert not foreign, f"{path.name} imports {sorted(foreign)}"


def test_dependencies_point_inward_only_runtime_knows_all_parts() -> None:
    """docs/how-it-works.md: `runtime` is the only place that brings all building blocks together."""
    files = sorted(HARNESS.rglob("*.py"))
    assert files
    for path in files:
        rel = path.relative_to(HARNESS)
        if rel.parts[0] == "runtime" or rel.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "runtime" not in node.module.split("."), f"{rel} must not import runtime"


def test_only_tool_executor_awaits_tool_run() -> None:
    """Invariant 4: Tools run exclusively through the execution gate (`tools/executor.py`)."""
    callers: list[str] = []
    for path in sorted(HARNESS.rglob("*.py")):
        rel = path.relative_to(HARNESS).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run":
                # contextvars.Context.run in SyncTool is synchronous (arg 0 is self.run_sync)
                if rel == "tools/base.py" and isinstance(node.func.value, ast.Name) and node.func.value.id == "context":
                    continue
                callers.append(rel)
    assert callers == ["tools/executor.py"]


def test_no_unapproved_suppression_comments() -> None:
    """AGENTS.md: No `# type: ignore`, `# noqa`, or `pragma: no cover` without explicit approval."""
    found: dict[str, int] = {}
    for folder in ("ai_agent_harness", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            tokens = tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
            count = sum(
                1
                for tok in tokens
                if tok.type == tokenize.COMMENT and any(tag in tok.string for tag in SUPPRESSION_TAGS)
            )
            if count:
                found[rel] = count
    assert found == APPROVED_SUPPRESSIONS


def test_readme_is_a_short_start_with_the_demo_and_the_proof() -> None:
    """The README is the front door: short, with a no-key demo and a link to the full list of guarantees."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert len(lines) <= 160, f"README.md has {len(lines)} lines (limit: 160): move detail into docs/"
    assert "examples/repo_assistant.py --demo" in text
    assert "docs/guarantees.md" in text and "docs/how-it-works.md" in text and "docs/guide.md" in text


def test_agents_md_stays_concise() -> None:
    """ETH Zurich finding (arXiv:2602.11988) & AGENTS.md: concise context instead of bloated overviews."""
    lines = (ROOT / "AGENTS.md").read_text(encoding="utf-8").splitlines()
    assert 40 <= len(lines) <= 100, f"AGENTS.md has {len(lines)} lines (limit: 100)"
