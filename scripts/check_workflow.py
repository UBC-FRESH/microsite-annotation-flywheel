#!/usr/bin/env python3
"""Check a repository against the UBC-FRESH dev workflow surfaces.

This module is the oracle for the planning surfaces. It parses ``ROADMAP.md`` and
``CHANGE_LOG.md``, resolves local Markdown links and the ``docs/llms.txt`` index, and
scans the text that a commit would include for defects that must not reach a public
repository: control characters, byte-order marks, personal paths, and secret-shaped
strings. Outside a git repository, the scan walks the tree and honors the root
``.gitignore``. It uses only the standard library, so it runs in a fresh clone before
anything is installed.

Command-line usage::

    python scripts/check_workflow.py [ROOT]
    python scripts/check_workflow.py --files PATH [PATH ...]

The exit status is 0 when no errors are found. Warnings are printed, but they do not
fail the run. Finding codes are stable; see ``docs/agents/oracles-and-evidence.rst``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT: Path = Path(__file__).resolve().parent.parent
SELF_RELPATH: str = "scripts/check_workflow.py"
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "_build",
        "tmp",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".hypothesis",
        "htmlcov",
        "runtime",
        "node_modules",
    }
)
SKIP_TOP_LEVEL: frozenset[str] = frozenset({"tests"})
MAX_SCAN_BYTES: int = 1_000_000
PHASE_STATUSES: tuple[str, ...] = ("planned", "active", "pr-open", "complete", "deferred")
OPEN_STATUSES: tuple[str, ...] = ("active", "pr-open", "complete")

REQUIRED_FILES: tuple[str, ...] = (
    "README.md",
    "AGENTS.md",
    "ROADMAP.md",
    "CHANGE_LOG.md",
    "notes/README.md",
    "docs/conf.py",
    "docs/index.rst",
    "docs/llms.txt",
    "docs/requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    ".pre-commit-config.yaml",
    ".gitignore",
    "scripts/check_workflow.py",
    "scripts/dashboard.py",
    ".github/ISSUE_TEMPLATE/phase.md",
    ".github/ISSUE_TEMPLATE/task.md",
    ".github/workflows/ci.yml",
    ".github/workflows/docs.yml",
    "docs/api/index.rst",
    "docs/api/check-workflow.rst",
    "docs/api/dashboard.rst",
    "tests/conftest.py",
    "tests/test_api_docstrings.py",
    "tests/test_workflow.py",
    "tests/test_check_workflow.py",
    "tests/test_cli.py",
    "tests/test_dashboard.py",
    "tests/test_properties.py",
)
RECOMMENDED_FILES: tuple[str, ...] = (
    "LICENSE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "CITATION.cff",
)

PHASE_RE: re.Pattern[str] = re.compile(r"^## Phase (\d+):\s*(.+?)\s*$")
PARENT_RE: re.Pattern[str] = re.compile(r"^Parent issue:\s*(#\d+|TBD)\s*$")
BRANCH_RE: re.Pattern[str] = re.compile(r"^Branch:\s*`([^`\s]+)`\s*$")
STATUS_RE: re.Pattern[str] = re.compile(r"^Status:\s*([a-z-]+)\s*$")
TASK_RE: re.Pattern[str] = re.compile(r"^- \[( |x|X)\] P(\d+)\.(\d+)\s+(.+?)\s*$")
SUBTASK_RE: re.Pattern[str] = re.compile(r"^\s{2,}- \[( |x|X)\]\s+\S")
ISSUE_REF_RE: re.Pattern[str] = re.compile(r"\(#(\d+)\)\s*$")
TRACKER_RE: re.Pattern[str] = re.compile(r"^\|\s*P(\d+)\b")
LINK_RE: re.Pattern[str] = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
CONTROL_RE: re.Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PERSONAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/home/(?![<${])[A-Za-z0-9._-]+/"), "Linux home directory path"),
    (re.compile(r"/Users/(?![<${])[A-Za-z0-9._-]+/"), "macOS home directory path"),
    (re.compile(r"[A-Za-z]:\\Users\\(?![<${])"), "Windows user profile path"),
)
SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "GitHub token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}"), "GitHub fine-grained token"),
    (re.compile(r"\bpypi-[A-Za-z0-9_-]{40,}"), "PyPI token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"), "private key"),
    (
        re.compile(
            r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*"
            r"[\"']?[A-Za-z0-9/+_-]{16,}"
        ),
        "credential assignment",
    ),
)


@dataclass(frozen=True)
class Diagnostic:
    """One finding: its severity, its stable code, its location, and its message.

    Attributes
    ----------
    severity : str
        Either ``"error"`` or ``"warning"``.
    code : str
        Stable finding code, such as ``ROADMAP-STATUS``.
    path : str
        Repository-relative path of the file that the finding concerns.
    line : int
        One-based line number, or ``0`` for a whole-file finding.
    message : str
        Explanation of what to change.
    """

    severity: str
    code: str
    path: str
    line: int
    message: str

    def render(self) -> str:
        """Render the finding as one line of text.

        Returns
        -------
        str
            The severity, code, location, and message, padded into columns.
        """
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{self.severity:<8}{self.code:<22}{where}  {self.message}"


@dataclass
class Task:
    """A task line in a phase section, together with its subtask lines.

    Attributes
    ----------
    phase : int
        The phase number that the task ID claims.
    number : int
        The task number within its phase.
    checked : bool
        Whether the task checkbox is ticked.
    line : int
        One-based line number of the task line.
    issue : int or None
        The child issue number, when the task line names one.
    subtasks : list of tuple of (int, bool)
        One ``(line number, checked)`` pair for each subtask line under the task.
    """

    phase: int
    number: int
    checked: bool
    line: int
    issue: int | None
    subtasks: list[tuple[int, bool]] = field(default_factory=list)


@dataclass
class Phase:
    """A ``## Phase <n>:`` section, its header lines, and its tasks.

    Attributes
    ----------
    number : int
        The phase number from the heading.
    line : int
        One-based line number of the heading.
    parent : str or None
        The ``Parent issue:`` value, such as ``#1`` or ``TBD``.
    branch : str or None
        The ``Branch:`` value, without backticks.
    status : str or None
        The ``Status:`` value.
    tasks : list of Task
        The tasks listed in the section, in file order.
    """

    number: int
    line: int
    parent: str | None = None
    branch: str | None = None
    status: str | None = None
    tasks: list[Task] = field(default_factory=list)


@dataclass(frozen=True)
class IgnoreRule:
    """One rule from the root ``.gitignore``, compiled for the file walk.

    Attributes
    ----------
    pattern : re.Pattern[str]
        The compiled pattern that the candidate must match in full.
    anchored : bool
        ``True`` when the rule contains a slash, so it matches the path from the root. ``False``
        when it matches the name of any path component.
    directory_only : bool
        ``True`` when the rule ends in a slash, so it applies to directories only.
    negated : bool
        ``True`` when the rule starts with ``!``, which re-includes a path that an earlier rule
        ignores.
    """

    pattern: re.Pattern[str]
    anchored: bool
    directory_only: bool
    negated: bool


def error(code: str, path: str, line: int, message: str) -> Diagnostic:
    """Build an error-severity finding.

    Parameters
    ----------
    code : str
        Stable finding code, such as ``LINK-BROKEN``.
    path : str
        Repository-relative path that the finding concerns.
    line : int
        One-based line number, or ``0`` for a whole-file finding.
    message : str
        Explanation of what to change.

    Returns
    -------
    Diagnostic
        A finding with severity ``"error"``.
    """
    return Diagnostic("error", code, path, line, message)


def warning(code: str, path: str, line: int, message: str) -> Diagnostic:
    """Build a warning-severity finding.

    Parameters
    ----------
    code : str
        Stable finding code, such as ``FILE-RECOMMENDED``.
    path : str
        Repository-relative path that the finding concerns.
    line : int
        One-based line number, or ``0`` for a whole-file finding.
    message : str
        Explanation of what to change.

    Returns
    -------
    Diagnostic
        A finding with severity ``"warning"``.
    """
    return Diagnostic("warning", code, path, line, message)


def _git_file_set(root: Path) -> list[Path] | None:
    """List the files that git would commit under a repository root.

    The list is tracked files plus untracked files that git does not ignore, which is the set a
    commit would include. It is used only when the root is the top level of a git working tree. A
    directory inside another repository, such as a generated project, is walked instead.

    Parameters
    ----------
    root : Path
        Directory to inspect.

    Returns
    -------
    list of Path or None
        Absolute paths of the candidate files. ``None`` when git is not installed, or when the root
        is not the top level of a repository, in which case the caller walks the tree.
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
            return None
        listing = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if listing.returncode != 0:
        return None
    return [root / name for name in listing.stdout.split("\0") if name]


def _glob_to_regex(glob: str) -> str:
    """Translate one gitignore glob, without ``**``, into a regular expression.

    Parameters
    ----------
    glob : str
        The pattern, with its ``!``, leading slash, and trailing slash already removed.

    Returns
    -------
    str
        A regular expression for the whole candidate. ``*`` and ``?`` do not match a slash. A
        bracket expression is kept, and a ``!`` at its start negates it. A bracket that is never
        closed is a literal ``[``.
    """
    pieces: list[str] = []
    index = 0
    while index < len(glob):
        char = glob[index]
        if char == "*":
            pieces.append("[^/]*")
        elif char == "?":
            pieces.append("[^/]")
        elif char == "[":
            end = glob.find("]", index + 2)
            if end == -1:
                pieces.append(re.escape(char))
            else:
                body = glob[index + 1 : end]
                pieces.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
                index = end
        else:
            pieces.append(re.escape(char))
        index += 1
    return "".join(pieces)


def _parse_ignore_line(line: str) -> IgnoreRule | None:
    """Parse one line of a .gitignore file into a rule, or return ``None`` if it is not one.

    Parameters
    ----------
    line : str
        One line of the ignore file, without its line ending.

    Returns
    -------
    IgnoreRule or None
        The rule. ``None`` for a blank line, a comment, and an ignore line the reader does not
        apply: a line with a backslash, a pattern with ``**``, a bare slash, or a bracket that is
        not a valid character class. Leaving out an ignore rule scans more files. A negated line
        the reader does not apply becomes a rule that re-includes every path, so no earlier rule
        can hide a file that git would include.
    """
    text = line.rstrip()
    if not text or text.startswith("#"):
        return None
    negated = text.startswith("!")
    if negated:
        text = text[1:]
    directory_only = text.endswith("/")
    text = text.rstrip("/")
    if not text or "\\" in text or "**" in text:
        return _unsupported(negated)
    anchored = "/" in text
    text = text.lstrip("/")
    try:
        pattern = re.compile(_glob_to_regex(text))
    except re.error:
        return _unsupported(negated)
    return IgnoreRule(
        pattern=pattern,
        anchored=anchored,
        directory_only=directory_only,
        negated=negated,
    )


def _unsupported(negated: bool) -> IgnoreRule | None:
    """Return the conservative rule for a line that the reader cannot apply.

    Parameters
    ----------
    negated : bool
        Whether the line starts with ``!``.

    Returns
    -------
    IgnoreRule or None
        A rule that re-includes every path when the line is negated, and ``None`` otherwise.
    """
    if not negated:
        return None
    return IgnoreRule(
        pattern=re.compile(r".*"),
        anchored=False,
        directory_only=False,
        negated=True,
    )


def _last_match(rules: Sequence[IgnoreRule], rel: str, *, is_dir: bool) -> bool:
    """Decide whether one path is ignored, by the last rule that matches it.

    Parameters
    ----------
    rules : sequence of IgnoreRule
        The rules, in file order.
    rel : str
        The path, relative to the root, with forward slashes.
    is_dir : bool
        Whether the path is a directory.

    Returns
    -------
    bool
        ``True`` when the last matching rule ignores the path. ``False`` when no rule matches, or
        when the last matching rule re-includes it.
    """
    ignored = False
    name = rel.rsplit("/", 1)[-1]
    for rule in rules:
        if rule.directory_only and not is_dir:
            continue
        candidate = rel if rule.anchored else name
        if rule.pattern.fullmatch(candidate) is not None:
            ignored = not rule.negated
    return ignored


def is_ignored(rel: str, *, is_dir: bool, rules: Sequence[IgnoreRule]) -> bool:
    """Report whether a path is ignored, the way git reports it.

    A path under an ignored directory is ignored, even when a later rule re-includes it, because
    git does not re-include a file whose directory is excluded.

    Parameters
    ----------
    rel : str
        The path, relative to the root, with forward slashes.
    is_dir : bool
        Whether the path is a directory.
    rules : sequence of IgnoreRule
        The rules from :func:`load_ignore_rules`, in file order.

    Returns
    -------
    bool
        ``True`` when the path, or one of the directories that contain it, is ignored.
    """
    parts = rel.split("/")
    for depth in range(1, len(parts)):
        if _last_match(rules, "/".join(parts[:depth]), is_dir=True):
            return True
    return _last_match(rules, rel, is_dir=is_dir)


def load_ignore_rules(root: Path) -> list[IgnoreRule]:
    """Read the root ``.gitignore`` rules, with a conservative rule for each unsupported negation.

    Only the root file is read. A ``.gitignore`` in a subdirectory is not read, and neither is
    ``.git/info/exclude``, because the walk has no index to consult.

    Parameters
    ----------
    root : Path
        Repository root, or the directory that stands in for one.

    Returns
    -------
    list of IgnoreRule
        The rules in file order. The list is empty when the root has no ``.gitignore``.
    """
    path = root / ".gitignore"
    if not path.is_file():
        return []
    rules: list[IgnoreRule] = []
    for line in _read_lines(path):
        rule = _parse_ignore_line(line)
        if rule is not None:
            rules.append(rule)
    return rules


def _walk_files(root: Path, rules: Sequence[IgnoreRule]) -> Iterator[Path]:
    """Yield the files under a root that the rules do not ignore. Ignored directories are pruned.

    Parameters
    ----------
    root : Path
        Directory to walk.
    rules : sequence of IgnoreRule
        The root ``.gitignore`` rules, from :func:`load_ignore_rules`.

    Yields
    ------
    Path
        Absolute path of each file, in sorted order within each directory.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath).relative_to(root)
        kept: list[str] = []
        for name in sorted(dirnames):
            rel = (here / name).as_posix()
            if name not in SKIP_DIRS and not is_ignored(rel, is_dir=True, rules=rules):
                kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            if not is_ignored((here / name).as_posix(), is_dir=False, rules=rules):
                yield Path(dirpath) / name


def iter_text_files(root: Path) -> Iterator[Path]:
    """Yield the UTF-8 text files that the scans should read.

    The candidates are the files that git would commit, when the root is a repository top level.
    Otherwise they are the files under the root that its ``.gitignore`` does not ignore. The
    ``tests`` tree, the validator itself, symbolic links, oversized files, and files that are not
    valid UTF-8 are skipped in both cases.

    Parameters
    ----------
    root : Path
        Repository root or directory to scan.

    Yields
    ------
    Path
        Absolute path of each file that passes the filters, in sorted order.
    """
    candidates = _git_file_set(root)
    if candidates is None:
        candidates = list(_walk_files(root, load_ignore_rules(root)))
    for path in sorted(candidates):
        rel = path.relative_to(root)
        if rel.parts[0] in SKIP_TOP_LEVEL or rel.as_posix() == SELF_RELPATH:
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.stat().st_size > MAX_SCAN_BYTES or not _is_utf8(path):
            continue
        yield path


def _is_utf8(path: Path) -> bool:
    """Report whether a file decodes as UTF-8.

    Parameters
    ----------
    path : Path
        File to test.

    Returns
    -------
    bool
        ``True`` when the file decodes as UTF-8.
    """
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _read_text(path: Path) -> str:
    """Read a file as text, replacing undecodable bytes.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    str
        The decoded text.
    """
    return path.read_bytes().decode("utf-8", errors="replace")


def _read_lines(path: Path) -> list[str]:
    """Read a file as a list of lines without their line endings.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    list of str
        The lines of the file, split on newline characters.
    """
    return _read_text(path).split("\n")


def _is_local_target(target: str) -> bool:
    """Report whether a link target is a local path that should be resolved.

    Parameters
    ----------
    target : str
        The raw link target, as written inside the parentheses.

    Returns
    -------
    bool
        ``False`` for web, mail, fragment-only, and placeholder targets.
    """
    if target.startswith(("http://", "https://", "mailto:", "#", "<")):
        return False
    return "{{" not in target and "<" not in target


def _strip_fragment(target: str) -> str:
    """Remove a ``#fragment`` suffix from a link target.

    Parameters
    ----------
    target : str
        The link target.

    Returns
    -------
    str
        The part of the target before the first ``#``.
    """
    return target.split("#", 1)[0]


def _issue_ref(text: str) -> int | None:
    """Extract a trailing ``(#n)`` issue reference from a task description.

    Parameters
    ----------
    text : str
        The task description.

    Returns
    -------
    int or None
        The issue number, or ``None`` when the description has no trailing reference.
    """
    match = ISSUE_REF_RE.search(text)
    return int(match.group(1)) if match else None


def parse_roadmap(lines: list[str]) -> list[Phase]:
    """Parse the phase sections of a roadmap into structured records.

    Phase headers, status lines, tasks, and subtasks are recognized by the grammar in
    ``docs/reference/workflow-model.rst``. Lines outside a phase section are ignored.

    Parameters
    ----------
    lines : list of str
        The roadmap, one entry per line, without line endings.

    Returns
    -------
    list of Phase
        One record for each ``## Phase <n>:`` heading, in file order.
    """
    phases: list[Phase] = []
    current: Phase | None = None
    task: Task | None = None
    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if line.startswith("## "):
            match = PHASE_RE.match(line)
            current = Phase(number=int(match.group(1)), line=lineno) if match else None
            if current is not None:
                phases.append(current)
            task = None
            continue
        if current is None:
            continue
        parent = PARENT_RE.match(line)
        if parent and current.parent is None:
            current.parent = parent.group(1)
        branch = BRANCH_RE.match(line)
        if branch and current.branch is None:
            current.branch = branch.group(1)
        status = STATUS_RE.match(line)
        if status and current.status is None:
            current.status = status.group(1)
        task_match = TASK_RE.match(line)
        if task_match:
            task = Task(
                phase=int(task_match.group(2)),
                number=int(task_match.group(3)),
                checked=task_match.group(1) in ("x", "X"),
                line=lineno,
                issue=_issue_ref(task_match.group(4)),
            )
            current.tasks.append(task)
            continue
        subtask = SUBTASK_RE.match(line)
        if subtask and task is not None:
            task.subtasks.append((lineno, subtask.group(1) in ("x", "X")))
    return phases


def parse_tracker(lines: list[str]) -> list[tuple[int, list[str], int]]:
    """Find the rows of the Issue Tracker Map table.

    Parameters
    ----------
    lines : list of str
        The roadmap, one entry per line.

    Returns
    -------
    list of tuple of (int, list of str, int)
        For each row that starts with a phase ID: the phase number, the stripped table
        cells, and the one-based line number.
    """
    rows: list[tuple[int, list[str], int]] = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        match = TRACKER_RE.match(line)
        if match is None:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append((int(match.group(1)), cells, lineno))
    return rows


def _check_phase(path: str, phase: Phase) -> list[Diagnostic]:
    """Check one phase section for missing metadata, status, and task consistency.

    Parameters
    ----------
    path : str
        Name of the roadmap file, used in the findings.
    phase : Phase
        The phase record to check.

    Returns
    -------
    list of Diagnostic
        Findings for this phase. The list is empty when the phase is consistent.
    """
    diags: list[Diagnostic] = []
    label = f"P{phase.number}"
    where = phase.line
    header = (("Parent issue", phase.parent), ("Branch", phase.branch), ("Status", phase.status))
    for name, value in header:
        if value is None:
            diags.append(error("ROADMAP-PHASE-META", path, where, f"{label} has no '{name}:' line"))
    if phase.status is not None and phase.status not in PHASE_STATUSES:
        allowed = ", ".join(PHASE_STATUSES)
        message = f"{label} status '{phase.status}' is not one of {allowed}"
        diags.append(error("ROADMAP-STATUS", path, where, message))
    if phase.parent == "TBD" and phase.status in OPEN_STATUSES:
        message = f"{label} is {phase.status} but has no parent issue; create it first"
        diags.append(error("ROADMAP-PARENT-REQUIRED", path, where, message))
    seen: set[int] = set()
    for task in phase.tasks:
        task_label = f"P{task.phase}.{task.number}"
        if task.phase != phase.number:
            message = f"{task_label} sits under {label}"
            diags.append(error("ROADMAP-TASK-ID", path, task.line, message))
        if task.number in seen:
            diags.append(error("ROADMAP-TASK-ID", path, task.line, f"{task_label} repeats"))
        seen.add(task.number)
        if phase.status != "complete":
            continue
        if not task.checked:
            message = f"{label} is complete but {task_label} is not checked"
            diags.append(error("ROADMAP-COMPLETE-OPEN", path, task.line, message))
        for sub_line, sub_checked in task.subtasks:
            if not sub_checked:
                message = f"{label} is complete but a subtask of {task_label} is not checked"
                diags.append(error("ROADMAP-COMPLETE-OPEN", path, sub_line, message))
    if phase.status == "complete" and not phase.tasks:
        message = f"{label} is complete but lists no tasks"
        diags.append(error("ROADMAP-COMPLETE-OPEN", path, where, message))
    return diags


def _check_tracker(path: str, lines: list[str], phases: list[Phase]) -> list[Diagnostic]:
    """Check the Issue Tracker Map against the phase sections.

    Parameters
    ----------
    path : str
        Name of the roadmap file, used in the findings.
    lines : list of str
        The roadmap, one entry per line.
    phases : list of Phase
        The phase records parsed from the same roadmap.

    Returns
    -------
    list of Diagnostic
        Findings for missing, duplicated, or inconsistent tracker rows.
    """
    diags: list[Diagnostic] = []
    rows: dict[int, tuple[list[str], int]] = {}
    for number, cells, lineno in parse_tracker(lines):
        if number in rows:
            diags.append(error("ROADMAP-TRACKER", path, lineno, f"P{number} has two tracker rows"))
        rows[number] = (cells, lineno)
    known = {phase.number for phase in phases}
    for number, (_, lineno) in rows.items():
        if number not in known:
            diags.append(
                error("ROADMAP-TRACKER", path, lineno, f"tracker row P{number} has no phase")
            )
    for phase in phases:
        label = f"P{phase.number}"
        if phase.number not in rows:
            diags.append(error("ROADMAP-TRACKER", path, phase.line, f"{label} has no tracker row"))
            continue
        cells, lineno = rows[phase.number]
        if len(cells) < 4:
            message = f"{label} row needs Phase, Parent issue, Branch, and Status cells"
            diags.append(error("ROADMAP-TRACKER", path, lineno, message))
            continue
        parent_cell, branch_cell, status_cell = cells[1], cells[2], cells[3]
        if parent_cell != (phase.parent or ""):
            message = f"{label} parent issue '{parent_cell}' differs from its phase section"
            diags.append(error("ROADMAP-TRACKER", path, lineno, message))
        if branch_cell.strip("`") != (phase.branch or ""):
            message = f"{label} branch '{branch_cell}' differs from its phase section"
            diags.append(error("ROADMAP-TRACKER", path, lineno, message))
        if status_cell != (phase.status or ""):
            message = f"{label} status '{status_cell}' differs from its phase section"
            diags.append(error("ROADMAP-TRACKER", path, lineno, message))
    return diags


def check_roadmap(path: str, text: str) -> list[Diagnostic]:
    """Check a roadmap's phase grammar, phase statuses, and tracker map.

    Parameters
    ----------
    path : str
        Name of the roadmap file, used in the findings.
    text : str
        The full roadmap text.

    Returns
    -------
    list of Diagnostic
        Every finding for the roadmap. The list is empty when the roadmap is consistent.
    """
    lines = text.split("\n")
    phases = parse_roadmap(lines)
    if not phases:
        return [error("ROADMAP-NO-PHASES", path, 0, "no '## Phase <n>:' headings found")]
    diags: list[Diagnostic] = []
    previous = -1
    for phase in phases:
        if phase.number <= previous:
            message = f"P{phase.number} must be unique and come after P{previous}"
            diags.append(error("ROADMAP-PHASE-ORDER", path, phase.line, message))
        previous = max(previous, phase.number)
        diags.extend(_check_phase(path, phase))
    diags.extend(_check_tracker(path, lines, phases))
    return diags


def check_changelog(path: str, text: str) -> list[Diagnostic]:
    """Check that changelog entries are ISO dates in non-decreasing order.

    Parameters
    ----------
    path : str
        Name of the changelog file, used in the findings.
    text : str
        The full changelog text.

    Returns
    -------
    list of Diagnostic
        Findings for malformed headings, out-of-order entries, and an empty changelog.
    """
    diags: list[Diagnostic] = []
    previous: dt.date | None = None
    dated = 0
    for lineno, raw in enumerate(text.split("\n"), start=1):
        if not raw.startswith("## "):
            continue
        label = raw[3:].strip()
        try:
            day = dt.date.fromisoformat(label)
        except ValueError:
            message = f"heading '{label}' is not a YYYY-MM-DD date"
            diags.append(error("CHANGELOG-DATE", path, lineno, message))
            continue
        dated += 1
        if previous is not None and day < previous:
            message = f"{label} is older than the entry above it; newest entries go last"
            diags.append(error("CHANGELOG-ORDER", path, lineno, message))
        previous = day
    if dated == 0:
        diags.append(error("CHANGELOG-EMPTY", path, 0, "no dated '## YYYY-MM-DD' entries"))
    return diags


def check_required_files(root: Path) -> list[Diagnostic]:
    """Check that the required files exist, and warn about recommended ones.

    Parameters
    ----------
    root : Path
        Repository root to check.

    Returns
    -------
    list of Diagnostic
        ``FILE-MISSING`` errors for absent required files, and ``FILE-RECOMMENDED``
        warnings for absent recommended files.
    """
    diags: list[Diagnostic] = []
    for rel in REQUIRED_FILES:
        if not (root / rel).is_file():
            diags.append(error("FILE-MISSING", rel, 0, "required file is missing"))
    for rel in RECOMMENDED_FILES:
        if not (root / rel).is_file():
            diags.append(warning("FILE-RECOMMENDED", rel, 0, "recommended file is missing"))
    return diags


def check_markdown_links(root: Path) -> list[Diagnostic]:
    """Check that every local link in a Markdown file points at an existing path.

    Parameters
    ----------
    root : Path
        Repository root to check.

    Returns
    -------
    list of Diagnostic
        ``LINK-BROKEN`` errors for each local link whose target does not exist.
    """
    diags: list[Diagnostic] = []
    for path in iter_text_files(root):
        if path.suffix != ".md":
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(_read_lines(path), start=1):
            for match in LINK_RE.finditer(line):
                target = match.group(1)
                if not _is_local_target(target):
                    continue
                file_part = _strip_fragment(target)
                if not (path.parent / file_part).exists():
                    message = f"link target '{file_part}' does not exist"
                    diags.append(error("LINK-BROKEN", rel, lineno, message))
    return diags


def check_llms(root: Path) -> list[Diagnostic]:
    """Check that every link in ``docs/llms.txt`` names an existing docs page.

    A link that ends in ``.html`` maps to the ``.rst`` source with the same stem. Any
    other link is resolved relative to ``docs/``.

    Parameters
    ----------
    root : Path
        Repository root to check.

    Returns
    -------
    list of Diagnostic
        ``LLMS-BROKEN`` errors for each link without a matching page. The list is empty
        when the index is absent, because ``FILE-MISSING`` reports that case.
    """
    index = root / "docs" / "llms.txt"
    if not index.is_file():
        return []
    diags: list[Diagnostic] = []
    for lineno, line in enumerate(_read_lines(index), start=1):
        for match in LINK_RE.finditer(line):
            target = match.group(1)
            if not _is_local_target(target):
                continue
            file_part = _strip_fragment(target)
            if file_part.endswith(".html"):
                source = root / "docs" / (file_part[: -len(".html")] + ".rst")
            else:
                source = root / "docs" / file_part
            if not source.exists():
                message = f"'{file_part}' has no matching page under docs/"
                diags.append(error("LLMS-BROKEN", "docs/llms.txt", lineno, message))
    return diags


def scan_text(rel: str, data: bytes) -> list[Diagnostic]:
    """Scan one file for control characters, byte-order marks, and unsafe strings.

    Parameters
    ----------
    rel : str
        Repository-relative path, used in the findings.
    data : bytes
        The raw file contents, which must be valid UTF-8.

    Returns
    -------
    list of Diagnostic
        One finding for each defect found. A line can produce more than one finding.
    """
    diags: list[Diagnostic] = []
    if b"\r" in data:
        diags.append(error("CONTROL-CHAR", rel, 0, "carriage return found; use LF line endings"))
    text = data.decode("utf-8")
    for lineno, line in enumerate(text.split("\n"), start=1):
        if "\ufeff" in line:
            diags.append(error("BOM", rel, lineno, "byte-order mark (U+FEFF) found"))
        control = CONTROL_RE.search(line)
        if control is not None:
            code_point = ord(control.group())
            message = f"control character U+{code_point:04X} found"
            diags.append(error("CONTROL-CHAR", rel, lineno, message))
        for pattern, label in PERSONAL_PATTERNS:
            if pattern.search(line):
                message = f"{label}; use a placeholder or a repository-relative path"
                diags.append(error("PATH-PERSONAL", rel, lineno, message))
                break
        for pattern, label in SECRET_PATTERNS:
            if pattern.search(line):
                message = f"possible {label}; remove it, and rotate it if it was real"
                diags.append(error("SECRET-SHAPED", rel, lineno, message))
                break
    return diags


def scan_repository(root: Path) -> list[Diagnostic]:
    """Run the text scans over every file that :func:`iter_text_files` yields.

    Parameters
    ----------
    root : Path
        Repository root to scan.

    Returns
    -------
    list of Diagnostic
        All text-hygiene findings for the repository.
    """
    diags: list[Diagnostic] = []
    for path in iter_text_files(root):
        rel = path.relative_to(root).as_posix()
        diags.extend(scan_text(rel, path.read_bytes()))
    return diags


def check_files(paths: list[Path]) -> list[Diagnostic]:
    """Run only the text scans on the given files, such as issue body files.

    Parameters
    ----------
    paths : list of Path
        Files to scan.

    Returns
    -------
    list of Diagnostic
        ``FILE-MISSING`` for absent files, ``ENCODING`` for files that are not valid
        UTF-8, and the text-hygiene findings for every other file.
    """
    diags: list[Diagnostic] = []
    for path in paths:
        name = str(path)
        if not path.is_file():
            diags.append(error("FILE-MISSING", name, 0, "file does not exist"))
            continue
        data = path.read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            diags.append(error("ENCODING", name, 0, "file is not valid UTF-8"))
            continue
        diags.extend(scan_text(name, data))
    return diags


def run(root: Path) -> list[Diagnostic]:
    """Run every repository check against a root and return all findings.

    Parameters
    ----------
    root : Path
        Repository root to check.

    Returns
    -------
    list of Diagnostic
        The findings from every check, in a stable order.
    """
    diags = check_required_files(root)
    roadmap = root / "ROADMAP.md"
    if roadmap.is_file():
        diags.extend(check_roadmap("ROADMAP.md", _read_text(roadmap)))
    changelog = root / "CHANGE_LOG.md"
    if changelog.is_file():
        diags.extend(check_changelog("CHANGE_LOG.md", _read_text(changelog)))
    diags.extend(check_markdown_links(root))
    diags.extend(check_llms(root))
    diags.extend(scan_repository(root))
    return diags


def main(argv: list[str] | None = None) -> int:
    """Run the validator from the command line.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments to parse. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` when no errors were found, and ``1`` otherwise.

    Raises
    ------
    SystemExit
        When the argument parser rejects the arguments, or the root is not a directory.
    """
    parser = argparse.ArgumentParser(description="Check the UBC-FRESH dev workflow surfaces.")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="repository root to check (default: the checkout that contains this script)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="run only the text scans on these files, such as issue body files",
    )
    args = parser.parse_args(argv)
    files: list[Path] | None = args.files
    if files:
        diags = check_files(files)
    else:
        root = Path(args.root).resolve()
        if not root.is_dir():
            parser.error(f"not a directory: {root}")
        diags = run(root)
    for diag in diags:
        print(diag.render())
    errors = sum(1 for diag in diags if diag.severity == "error")
    print(f"{errors} error(s), {len(diags) - errors} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
