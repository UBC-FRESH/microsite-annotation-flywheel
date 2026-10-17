"""Docstring tests: every public function and method in the framework scripts documents its API.

The checks use numpydoc's validator, which reads the numpy-style sections of a docstring and
compares them with the function's signature. The codes that describe optional sections (an
extended summary, a See Also block, an Examples block, and a Yields block for functions that do
not yield) are not required, so they are not enforced. Every other numpydoc code fails the test.
"""

from __future__ import annotations

import importlib
from inspect import getmembers, isclass, isfunction
from pathlib import Path
from typing import Any, Final, cast

import numpydoc.validate as npd_validate
import pytest

SCRIPTS: Final[Path] = Path(__file__).resolve().parents[1] / "scripts"
OPTIONAL_CODES: Final[frozenset[str]] = frozenset({"ES01", "SA01", "EX01", "YD01"})


def public_qualnames() -> list[str]:
    """List the dotted names of every public function and method in ``scripts/``.

    Imported helpers are excluded, and so are names that start with an underscore. Every module
    in the scripts directory is inspected, so a new script is covered without editing this file.

    Returns
    -------
    list of str
        Names such as ``check_workflow.parse_roadmap`` or ``check_workflow.Diagnostic.render``,
        sorted so that the test identifiers are stable.
    """
    names: list[str] = []
    for path in sorted(SCRIPTS.glob("*.py")):
        module = importlib.import_module(path.stem)
        for name, value in getmembers(module, isfunction):
            if not name.startswith("_") and value.__module__ == module.__name__:
                names.append(f"{module.__name__}.{name}")
        for class_name, cls in getmembers(module, isclass):
            if class_name.startswith("_") or cls.__module__ != module.__name__:
                continue
            for method_name, method in getmembers(cls, isfunction):
                if not method_name.startswith("_") and method.__module__ == module.__name__:
                    names.append(f"{module.__name__}.{class_name}.{method_name}")
    return sorted(names)


def test_discovery_finds_the_public_api() -> None:
    """The discovery finds the entry points that the rest of the framework depends on."""
    names = public_qualnames()
    assert "check_workflow.parse_roadmap" in names
    assert "check_workflow.Diagnostic.render" in names
    assert "dashboard.main" in names
    assert len(names) >= 20


@pytest.mark.parametrize("qualname", public_qualnames())
def test_public_api_documents_its_parameters_and_returns(qualname: str) -> None:
    """Every public function and method has the docstring sections that its signature requires."""
    report: dict[str, Any] = npd_validate.validate(qualname)
    errors = cast("list[tuple[str, str]]", report["errors"])
    failures = [f"{code}: {message}" for code, message in errors if code not in OPTIONAL_CODES]
    assert failures == [], qualname
