"""Links in the documentation must lead somewhere: every relative link and every `#anchor` is checked."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)


def documents() -> list[Path]:
    found = [
        *ROOT.glob("*.md"),
        *ROOT.glob("docs/**/*.md"),
        *ROOT.glob("examples/**/*.md"),
        *ROOT.glob("tests/**/*.md"),
    ]
    return sorted([*found, ROOT / "llms.txt"])


def strip_code(text: str) -> str:
    return INLINE_CODE.sub("", FENCE.sub("", text))


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, no punctuation except hyphens, spaces become hyphens."""
    text = re.sub(r"`", "", heading).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text)


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    found: set[str] = set()
    for heading in HEADING.findall(FENCE.sub("", path.read_text(encoding="utf-8"))):
        base = slug(heading)
        count = seen.get(base, 0)
        seen[base] = count + 1
        found.add(base if count == 0 else f"{base}-{count}")
    return found


def test_the_documents_to_check_exist() -> None:
    names = {p.relative_to(ROOT).as_posix() for p in documents()}
    assert {"README.md", "docs/README.md", "CONTRIBUTING.md", "SECURITY.md", "llms.txt"} <= names


@pytest.mark.parametrize("path", documents(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_relative_links_and_anchors_resolve(path: Path) -> None:
    problems: list[str] = []
    for target in LINK.findall(strip_code(path.read_text(encoding="utf-8"))):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto: ...
            continue
        file_part, _, anchor = target.partition("#")
        destination = path if not file_part else (path.parent / file_part).resolve()
        if not destination.exists():
            problems.append(f"{target}: no such file")
        elif anchor and destination.suffix == ".md" and anchor not in anchors(destination):
            problems.append(f"{target}: no such heading")
    assert not problems, f"{path.relative_to(ROOT)}: " + "; ".join(problems)


def test_the_link_checker_catches_broken_links(tmp_path: Path) -> None:
    """The checker itself is checked: a broken file link and a broken anchor are both found."""
    doc = tmp_path / "a.md"
    doc.write_text(
        "# Title\n\n## Second part\n\n[ok](#second-part) [bad](#nope) [gone](missing.md)\n", encoding="utf-8"
    )
    found = LINK.findall(doc.read_text(encoding="utf-8"))
    assert found == ["#second-part", "#nope", "missing.md"]
    assert "second-part" in anchors(doc) and "nope" not in anchors(doc)
    assert not (tmp_path / "missing.md").exists()
    assert slug("`resume` & the *pending_id*: Part 2") == "resume--the-pending_id-part-2"
