"""Tests for the green dashboard: the pure summaries, the runner, and the exit paths.

The tests never run the default gates, because that would run this suite from inside
itself. Every ``main`` call passes its own gates or patches ``default_gates``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import dashboard as db


def result(name: str, status: db.Status, returncode: int = 0, tail: str = "") -> db.GateResult:
    """Build a gate result with fixed timing for the summary tests.

    Parameters
    ----------
    name : str
        Gate name.
    status : {"green", "red"}
        Gate status.
    returncode : int, optional
        Exit status to record.
    tail : str, optional
        Output tail to record.

    Returns
    -------
    GateResult
        The constructed result.
    """
    return db.GateResult(name, status, returncode, 0.5, tail)


def python_gate(name: str, code: str) -> db.Gate:
    """Build a gate that runs a short Python program in the current interpreter.

    Parameters
    ----------
    name : str
        Gate name.
    code : str
        Program text passed to ``python -c``.

    Returns
    -------
    Gate
        The constructed gate.
    """
    return db.Gate(name, (sys.executable, "-c", code), f"runs {name}")


def test_overall_status_is_green_only_when_every_gate_is_green() -> None:
    """One red gate turns the whole dashboard red."""
    assert db.overall_status([result("a", "green"), result("b", "green")]) == "green"
    assert db.overall_status([result("a", "green"), result("b", "red", 1)]) == "red"


def test_an_empty_dashboard_is_red() -> None:
    """A run with no gates checks nothing, so it is not green."""
    assert db.overall_status([]) == "red"


def test_text_rendering_for_a_green_dashboard_has_no_red_sections() -> None:
    """A green table states its verdict and shows no output tails."""
    text = db.render_text([result("lint", "green", tail="should not appear")])
    assert "OVERALL: GREEN" in text
    assert "should not appear" not in text
    assert "---" not in text


def test_text_rendering_for_a_red_dashboard_shows_each_red_gate_tail() -> None:
    """A red table names the failing gate, its exit status, and its output."""
    text = db.render_text(
        [result("lint", "green"), result("types", "red", 1, "error: bad annotation")]
    )
    assert "OVERALL: RED" in text
    assert "--- types (exit 1) ---" in text
    assert "error: bad annotation" in text


def test_markdown_rendering_names_the_verdict_and_every_gate() -> None:
    """The Markdown table has a heading, one row per gate, and a red-state instruction."""
    text = db.render_markdown([result("lint", "green"), result("docs", "red", 2)])
    assert text.startswith("## Green dashboard: RED")
    assert "| lint | GREEN | 0.50 |" in text
    assert "| docs | RED | 0.50 |" in text
    assert "Investigate and fix" in text


def test_markdown_rendering_of_a_green_dashboard_omits_the_instruction() -> None:
    """A green Markdown summary does not carry the red-state instruction."""
    text = db.render_markdown([result("lint", "green")])
    assert text.startswith("## Green dashboard: GREEN")
    assert "Investigate and fix" not in text


def test_record_is_machine_readable() -> None:
    """The JSON record carries the overall status and every gate's fields."""
    payload = json.loads(json.dumps(db.record([result("lint", "red", 1, "tail")])))
    assert payload["overall"] == "red"
    assert payload["gates"][0] == {
        "name": "lint",
        "status": "red",
        "returncode": 1,
        "seconds": 0.5,
        "tail": "tail",
    }


def test_run_gate_reports_green_and_keeps_the_output(tmp_path: Path) -> None:
    """A gate that exits with 0 is green, and its output is kept."""
    outcome = db.run_gate(python_gate("ok", "print('hello from the gate')"), tmp_path)
    assert outcome.status == "green"
    assert outcome.returncode == 0
    assert "hello from the gate" in outcome.tail


def test_run_gate_reports_red_with_the_exit_status(tmp_path: Path) -> None:
    """A gate that exits non-zero is red, and its exit status is recorded."""
    code = "import sys; print('failing output'); sys.exit(3)"
    outcome = db.run_gate(python_gate("bad", code), tmp_path)
    assert outcome.status == "red"
    assert outcome.returncode == 3
    assert "failing output" in outcome.tail


def test_run_gate_keeps_only_the_last_output_lines(tmp_path: Path) -> None:
    """Long output is cut to the final lines that a reader needs to investigate."""
    code = "for i in range(40): print('line', i)"
    outcome = db.run_gate(python_gate("long", code), tmp_path)
    lines = outcome.tail.splitlines()
    assert len(lines) == db.TAIL_LINES
    assert lines[-1] == "line 39"


def test_run_gates_keeps_the_order_of_the_gates(tmp_path: Path) -> None:
    """Results come back in the same order as the gates."""
    gates = [python_gate("first", "pass"), python_gate("second", "pass")]
    assert [r.name for r in db.run_gates(gates, tmp_path)] == ["first", "second"]


def test_default_gates_include_seed_gates_only_when_the_generator_exists(tmp_path: Path) -> None:
    """A repository without the project generator does not get the seed gates."""
    bare = {gate.name for gate in db.default_gates(tmp_path, python="python")}
    assert "seed" not in bare
    assert {"lint", "format", "types", "tests", "validator", "docs"} <= bare
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "new_project.py").write_text("", encoding="utf-8")
    full = {gate.name for gate in db.default_gates(tmp_path, python="python")}
    assert {"seed", "seeded-dashboard"} <= full


def test_default_gates_use_the_given_interpreter(tmp_path: Path) -> None:
    """Every default command starts with the interpreter that the caller names."""
    gates = db.default_gates(tmp_path, python="/opt/python-under-test")
    assert all(gate.command[0] == "/opt/python-under-test" for gate in gates)


def test_main_writes_the_json_record_and_returns_zero_when_green(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A green run returns 0, prints the verdict, and writes the JSON record."""
    target = tmp_path / "out" / "dashboard.json"
    code = db.main(
        ["--json", str(target)],
        gates=[python_gate("ok", "pass")],
        root=tmp_path,
    )
    assert code == 0
    assert json.loads(target.read_text(encoding="utf-8"))["overall"] == "green"
    assert "OVERALL: GREEN" in capsys.readouterr().out


def test_main_returns_one_when_any_gate_is_red(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A single red gate makes the command return 1 and report the red verdict."""
    gates = [python_gate("ok", "pass"), python_gate("bad", "raise SystemExit(2)")]
    assert db.main([], gates=gates, root=tmp_path) == 1
    output = capsys.readouterr().out
    assert "OVERALL: RED" in output
    assert "--- bad (exit 2) ---" in output


def test_main_prints_markdown_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Markdown flag switches the printed output to the Markdown table."""
    assert db.main(["--markdown"], gates=[python_gate("ok", "pass")], root=tmp_path) == 0
    assert capsys.readouterr().out.startswith("## Green dashboard: GREEN")


def test_main_uses_the_default_gates_when_none_are_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without explicit gates, the command runs the gates from :func:`default_gates`."""
    chosen = (python_gate("only", "pass"),)

    def fake_defaults(root: Path, python: str = sys.executable) -> tuple[db.Gate, ...]:
        return chosen

    monkeypatch.setattr(db, "default_gates", fake_defaults)
    assert db.main([], root=tmp_path) == 0
    assert "only" in capsys.readouterr().out


def test_main_rejects_unknown_options() -> None:
    """An unknown option makes the argument parser exit with status 2."""
    with pytest.raises(SystemExit) as excinfo:
        db.main(["--no-such-option"], gates=[])
    assert excinfo.value.code == 2
