"""Tests for the architecture decision records.

An ADR index that has drifted from the records is worse than no index: it
implies a decision trail that isn't there. These checks are cheap and catch the
two failure modes that actually happen — a record added without indexing it, and
a record that skips the sections which make it a decision rather than a note.

They deliberately do not check prose quality. A record can pass these and still
be useless; they only stop the collection becoming quietly inconsistent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ADR_DIR = Path(__file__).resolve().parents[1] / "docs/adr"
INDEX = ADR_DIR / "README.md"

REQUIRED_SECTIONS = ("## Status", "## Context", "## Decision", "## Consequences")


def adr_files() -> list[Path]:
    """Return every numbered ADR, in order."""
    return sorted(p for p in ADR_DIR.glob("[0-9][0-9][0-9][0-9]-*.md"))


def test_there_are_records_and_an_index():
    assert INDEX.exists()
    assert adr_files(), "no ADRs found"


@pytest.mark.parametrize("path", adr_files(), ids=lambda p: p.stem)
def test_each_record_has_the_sections_that_make_it_a_decision(path):
    text = path.read_text()
    for section in REQUIRED_SECTIONS:
        assert section in text, f"{path.name} has no '{section}' section"
    # The section that stops an ADR being a justification written after the
    # fact: stating what evidence would reverse it.
    assert "## What would change this" in text, f"{path.name} does not say what would reverse it"


@pytest.mark.parametrize("path", adr_files(), ids=lambda p: p.stem)
def test_each_record_has_a_title_matching_its_number(path):
    first_line = path.read_text().splitlines()[0]
    number = path.name[:4]
    assert first_line.startswith(f"# {number}."), (
        f"{path.name} should start with '# {number}. <title>', got {first_line!r}"
    )


@pytest.mark.parametrize("path", adr_files(), ids=lambda p: p.stem)
def test_each_record_declares_a_recognised_status(path):
    text = path.read_text()
    status = text.split("## Status", 1)[1].split("##", 1)[0].strip().rstrip(".")
    assert status in {"Accepted", "Proposed", "Deprecated"} or status.startswith("Superseded"), (
        f"{path.name} has an unrecognised status: {status!r}"
    )


def test_numbers_are_unique_and_contiguous():
    numbers = [int(p.name[:4]) for p in adr_files()]
    assert len(numbers) == len(set(numbers)), "duplicate ADR numbers"
    assert numbers == list(range(1, len(numbers) + 1)), f"non-contiguous numbering: {numbers}"


def test_every_record_is_listed_in_the_index():
    listed = set(re.findall(r"\((\d{4}-[a-z0-9-]+\.md)\)", INDEX.read_text()))
    on_disk = {p.name for p in adr_files()}
    assert on_disk - listed == set(), f"not in the index: {sorted(on_disk - listed)}"
    assert listed - on_disk == set(), f"indexed but missing: {sorted(listed - on_disk)}"
