"""Property tests: invariants that must hold for every generated input.

Hypothesis generates valid roadmaps, control-character injections, and changelog date
sequences. The properties state what the validator must do for all of them, which the
hand-written examples in ``test_check_workflow.py`` cannot cover.
"""

from __future__ import annotations

import datetime as dt
from typing import Final

from hypothesis import given, settings
from hypothesis import strategies as st

import check_workflow as cw

CONTROL_CHARACTERS: Final[tuple[str, ...]] = tuple(
    chr(code) for code in (*range(0, 9), 11, 12, *range(14, 32), 127)
)


@st.composite
def valid_roadmaps(draw: st.DrawFn) -> str:
    """Draw the text of a roadmap that satisfies every roadmap rule.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.

    Returns
    -------
    str
        A roadmap with a consistent tracker map and phase sections.
    """
    phase_count = draw(st.integers(min_value=1, max_value=4))
    tracker = [
        "## Issue Tracker Map",
        "",
        "| Phase | Parent issue | Branch | Status |",
        "| --- | --- | --- | --- |",
    ]
    sections: list[str] = []
    for number in range(phase_count):
        status = draw(st.sampled_from(cw.PHASE_STATUSES))
        if status in cw.OPEN_STATUSES:
            parent = f"#{draw(st.integers(min_value=1, max_value=999))}"
        else:
            parent = draw(st.sampled_from(["TBD", "#42"]))
        branch = f"feature/p{number}-phase-{number}"
        tracker.append(f"| P{number} Phase {number} | {parent} | `{branch}` | {status} |")
        low = 1 if status == "complete" else 0
        task_count = draw(st.integers(min_value=low, max_value=3))
        lines = [
            f"## Phase {number}: Phase {number}",
            "",
            f"Parent issue: {parent}",
            f"Branch: `{branch}`",
            f"Status: {status}",
            "",
        ]
        for task_number in range(1, task_count + 1):
            checked = status == "complete" or draw(st.booleans())
            box = "x" if checked else " "
            lines.append(f"- [{box}] P{number}.{task_number} Task {task_number}")
            if draw(st.booleans()):
                sub_box = "x" if status == "complete" or draw(st.booleans()) else " "
                lines.append(f"  - [{sub_box}] A subtask of P{number}.{task_number}")
        lines.append("")
        sections.append("\n".join(lines))
    head = ["# Generated Roadmap", "", *tracker, ""]
    return "\n".join(head) + "\n" + "\n".join(sections)


@settings(max_examples=60, deadline=None)
@given(text=valid_roadmaps())
def test_generated_valid_roadmaps_have_no_findings(text: str) -> None:
    """Every roadmap that follows the grammar passes the roadmap checks with no findings."""
    assert cw.check_roadmap("ROADMAP.md", text) == []


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_injected_control_characters_are_always_found(data: st.DataObject) -> None:
    """A control character anywhere in a valid roadmap produces a CONTROL-CHAR finding."""
    text = data.draw(valid_roadmaps())
    position = data.draw(st.integers(min_value=0, max_value=len(text)))
    character = data.draw(st.sampled_from(CONTROL_CHARACTERS))
    corrupted = text[:position] + character + text[position:]
    codes = {d.code for d in cw.scan_text("ROADMAP.md", corrupted.encode("utf-8"))}
    assert "CONTROL-CHAR" in codes


@settings(max_examples=60, deadline=None)
@given(
    dates=st.lists(
        st.dates(min_value=dt.date(2020, 1, 1), max_value=dt.date(2030, 12, 31)),
        min_size=1,
        max_size=6,
    )
)
def test_changelog_order_matches_the_dates(dates: list[dt.date]) -> None:
    """The changelog is flagged for ordering exactly when its dates are not sorted."""
    body = "".join(f"## {day.isoformat()}\n\n- Entry.\n\n" for day in dates)
    diags = cw.check_changelog("CHANGE_LOG.md", "# Change Log\n\n" + body)
    codes = {d.code for d in diags}
    assert ("CHANGELOG-ORDER" in codes) == (dates != sorted(dates))
    assert "CHANGELOG-DATE" not in codes
    assert "CHANGELOG-EMPTY" not in codes
