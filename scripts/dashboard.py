#!/usr/bin/env python3
"""Run every quality gate and report one green or red dashboard.

The dashboard is the single signal that every commit must keep green. It runs each gate
in its own subprocess, records the outcome, and prints a table. Its exit status is 0 only
when every gate is green, so the same command serves as a local habit, a pre-commit
check, and a CI step.

Command-line usage::

    python scripts/dashboard.py                        # table on the terminal
    python scripts/dashboard.py --markdown             # Markdown, for CI step summaries
    python scripts/dashboard.py --json runtime/dashboard.json

The seed gates generate a project and run its dashboard. They run only in a repository
that contains the project generator, ``scripts/new_project.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Status = Literal["green", "red"]

DEFAULT_ROOT: Path = Path(__file__).resolve().parent.parent
TAIL_LINES: int = 12
SEED_TARGET: str = "tmp/dashboard-seed"


@dataclass(frozen=True)
class Gate:
    """A quality gate: a named command that must exit with status 0.

    Attributes
    ----------
    name : str
        Short identifier shown on the dashboard.
    command : tuple of str
        Argument vector, run from the repository root.
    purpose : str
        One sentence that says what the gate protects.
    """

    name: str
    command: tuple[str, ...]
    purpose: str


@dataclass(frozen=True)
class GateResult:
    """The outcome of running one gate.

    Attributes
    ----------
    name : str
        The gate's name.
    status : {"green", "red"}
        ``"green"`` when the command exited with status 0, and ``"red"`` otherwise.
    returncode : int
        The command's exit status.
    seconds : float
        Wall-clock seconds the command took, rounded to two decimals.
    tail : str
        The last lines of combined output, kept so that a red gate can be investigated.
    """

    name: str
    status: Status
    returncode: int
    seconds: float
    tail: str


def default_gates(root: Path, python: str = sys.executable) -> tuple[Gate, ...]:
    """Build the gates that apply to a repository.

    Parameters
    ----------
    root : Path
        Repository root. Whether ``scripts/new_project.py`` exists there decides whether
        the seed gates are included.
    python : str, optional
        Interpreter for every command. The default is the interpreter that runs the
        dashboard.

    Returns
    -------
    tuple of Gate
        The lint, format, type, test, validator, and docs gates, followed by the seed gates
        when the project generator is present.
    """
    gates = [
        Gate("lint", (python, "-m", "ruff", "check", "."), "ruff finds no lint errors"),
        Gate("format", (python, "-m", "ruff", "format", "--check", "."), "code is formatted"),
        Gate("types", (python, "-m", "mypy"), "mypy --strict passes on the configured files"),
        Gate("tests", (python, "-m", "pytest", "-q"), "tests pass and the coverage gate holds"),
        Gate(
            "validator",
            (python, "scripts/check_workflow.py"),
            "the planning surfaces are consistent",
        ),
        Gate(
            "docs",
            (python, "-m", "sphinx", "-b", "html", "docs", "_build/html", "-W", "-q"),
            "the docs build with warnings as errors",
        ),
    ]
    if (root / "scripts" / "new_project.py").is_file():
        seed_command = (
            python,
            "scripts/new_project.py",
            "--name",
            "Dashboard Seed",
            "--slug",
            "dashboard-seed",
            "--dest",
            SEED_TARGET,
            "--force",
        )
        gates.append(
            Gate("seed", seed_command, "the seed renders a project that passes the validator")
        )
        gates.append(
            Gate(
                "seeded-dashboard",
                (python, f"{SEED_TARGET}/scripts/dashboard.py"),
                "the rendered project passes its own full dashboard",
            )
        )
    return tuple(gates)


def run_gate(gate: Gate, cwd: Path) -> GateResult:
    """Run one gate in a subprocess and record its outcome.

    Parameters
    ----------
    gate : Gate
        The gate to run.
    cwd : Path
        Directory to run the command in, normally the repository root.

    Returns
    -------
    GateResult
        The status, exit code, duration, and output tail of the run.
    """
    started = time.perf_counter()
    completed = subprocess.run(gate.command, cwd=cwd, capture_output=True, text=True, check=False)
    seconds = round(time.perf_counter() - started, 2)
    lines = (completed.stdout + completed.stderr).rstrip().splitlines()
    status: Status = "green" if completed.returncode == 0 else "red"
    return GateResult(
        name=gate.name,
        status=status,
        returncode=completed.returncode,
        seconds=seconds,
        tail="\n".join(lines[-TAIL_LINES:]),
    )


def run_gates(gates: Sequence[Gate], cwd: Path) -> list[GateResult]:
    """Run every gate in order and collect the results.

    Parameters
    ----------
    gates : sequence of Gate
        The gates to run, in the order they should appear.
    cwd : Path
        Directory to run every command in.

    Returns
    -------
    list of GateResult
        One result per gate, in the same order as ``gates``.
    """
    return [run_gate(gate, cwd) for gate in gates]


def overall_status(results: Sequence[GateResult]) -> Status:
    """Summarize a dashboard as one status.

    Parameters
    ----------
    results : sequence of GateResult
        The results to summarize.

    Returns
    -------
    {"green", "red"}
        ``"green"`` only when there is at least one result and every result is green.
        An empty dashboard is red, because nothing was checked.
    """
    if results and all(result.status == "green" for result in results):
        return "green"
    return "red"


def render_text(results: Sequence[GateResult]) -> str:
    """Render the dashboard as a plain-text table.

    Parameters
    ----------
    results : sequence of GateResult
        The results to render.

    Returns
    -------
    str
        A table with one row per gate, an overall verdict, and the output tail of each red
        gate.
    """
    rows = [f"{'GATE':<20}{'STATUS':<8}{'SECONDS':>8}"]
    rows.extend(f"{r.name:<20}{r.status.upper():<8}{r.seconds:>8.2f}" for r in results)
    rows.append("")
    if overall_status(results) == "green":
        rows.append("OVERALL: GREEN. Every gate passed.")
    else:
        rows.append("OVERALL: RED. Investigate and fix the red gates before continuing.")
    for result in results:
        if result.status == "red":
            rows.extend(["", f"--- {result.name} (exit {result.returncode}) ---", result.tail])
    return "\n".join(rows)


def render_markdown(results: Sequence[GateResult]) -> str:
    """Render the dashboard as a Markdown table, for a CI step summary.

    Parameters
    ----------
    results : sequence of GateResult
        The results to render.

    Returns
    -------
    str
        A heading with the overall verdict, and a table with one row per gate.
    """
    overall = overall_status(results).upper()
    lines = [
        f"## Green dashboard: {overall}",
        "",
        "| Gate | Status | Seconds |",
        "| --- | --- | ---: |",
    ]
    lines.extend(f"| {r.name} | {r.status.upper()} | {r.seconds:.2f} |" for r in results)
    if overall == "RED":
        lines.extend(["", "Investigate and fix the red gates before continuing."])
    return "\n".join(lines)


def record(results: Sequence[GateResult]) -> dict[str, object]:
    """Build the machine-readable record of a dashboard run.

    Parameters
    ----------
    results : sequence of GateResult
        The results to record.

    Returns
    -------
    dict of str to object
        The overall status and one entry per gate, ready for ``json.dumps``.
    """
    return {"overall": overall_status(results), "gates": [asdict(r) for r in results]}


def main(
    argv: Sequence[str] | None = None,
    *,
    gates: Sequence[Gate] | None = None,
    root: Path = DEFAULT_ROOT,
) -> int:
    """Run the dashboard from the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments to parse. ``None`` reads ``sys.argv``.
    gates : sequence of Gate or None, optional
        Gates to run instead of the defaults. Tests use this to avoid running the full
        suite from inside itself.
    root : Path, optional
        Repository root in which the commands run.

    Returns
    -------
    int
        ``0`` when every gate is green, and ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description="Run every quality gate and report.")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    parser.add_argument("--json", type=Path, metavar="PATH", help="also write a JSON record")
    args = parser.parse_args(argv)
    chosen = gates if gates is not None else default_gates(root)
    results = run_gates(chosen, root)
    print(render_markdown(results) if args.markdown else render_text(results))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record(results), indent=2) + "\n", encoding="utf-8")
    return 0 if overall_status(results) == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
