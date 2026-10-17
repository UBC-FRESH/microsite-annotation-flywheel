"""Self-check: the planning surfaces of this project pass the workflow validator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_passes_the_workflow_validator() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_workflow.py"), str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
