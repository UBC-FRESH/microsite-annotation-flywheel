"""Integration tests: run the framework command-line tools in subprocesses.

These tests exercise the same entry points that CI and the pre-commit hooks call, so a
broken import or a broken exit status shows up here even when the unit tests pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
SCRIPTS: Path = ROOT / "scripts"


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one script from ``scripts/`` in a subprocess.

    Parameters
    ----------
    name : str
        File name of the script, such as ``check_workflow.py``.
    *args : str
        Arguments passed to the script.

    Returns
    -------
    subprocess.CompletedProcess of str
        The finished process, with its captured output.
    """
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_validator_passes_on_this_repository() -> None:
    """The validator, run as CI runs it, passes on this repository."""
    result = run_script("check_workflow.py", str(ROOT))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 error(s)" in result.stdout


def test_validator_fails_a_defective_body_file(tmp_path: Path) -> None:
    """The validator, run on one body file, fails it when a form feed is present."""
    body = tmp_path / "body.md"
    body.write_text(f"Branch: eature{chr(12)}/p0\n", encoding="utf-8")
    result = run_script("check_workflow.py", "--files", str(body))
    assert result.returncode == 1
    assert "CONTROL-CHAR" in result.stdout


def test_validator_help_documents_the_files_option() -> None:
    """The help text names the file mode, so that people can find the check."""
    result = run_script("check_workflow.py", "--help")
    assert result.returncode == 0
    assert "--files" in result.stdout


def test_dashboard_help_documents_the_outputs() -> None:
    """The dashboard help text names its Markdown and JSON outputs."""
    result = run_script("dashboard.py", "--help")
    assert result.returncode == 0
    assert "--markdown" in result.stdout
    assert "--json" in result.stdout


def test_dashboard_rejects_an_unknown_option() -> None:
    """The dashboard exits with status 2 on an option it does not know."""
    result = run_script("dashboard.py", "--bogus")
    assert result.returncode == 2
