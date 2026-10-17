"""Unit tests for the workflow validator.

Every finding code has a test that makes it fire, and the adjacent valid input is shown
not to fire it. Bad fixtures are built at runtime, so this file never contains the
patterns that the validator checks for.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import check_workflow as cw

FORM_FEED: str = chr(12)
TOKEN_OPEN: str = "{" * 2
TOKEN_CLOSE: str = "}" * 2
BOM: str = chr(0xFEFF)
HOME_PREFIX: str = "/" + "home" + "/" + "someone" + "/"
MAC_PREFIX: str = "/" + "Users" + "/" + "someone" + "/"
WINDOWS_PREFIX: str = "C:" + "\\" + "Users" + "\\" + "someone" + "\\"
GITHUB_TOKEN: str = "ghp_" + "A" * 36
FINE_GRAINED_TOKEN: str = "github_pat_" + "A" * 40
PYPI_TOKEN: str = "pypi-" + "A" * 48
AWS_KEY: str = "AKIA" + "A" * 16
PRIVATE_KEY: str = "-----BEGIN " + "RSA " + "PRIVATE KEY-----"
CREDENTIAL: str = "api_key = " + "abcdefgh" + "ijklmnop"

VALID_ROADMAP: str = """# Demo Roadmap

## Issue Tracker Map

| Phase | Parent issue | Branch | Status |
| --- | --- | --- | --- |
| P0 Bootstrap | TBD | `feature/p0-bootstrap` | planned |
| P1 Build | #2 | `feature/p1-build` | active |

## Phase 0: Bootstrap

Parent issue: TBD
Branch: `feature/p0-bootstrap`
Status: planned

- [ ] P0.1 First bootstrap task

## Phase 1: Build

Parent issue: #2
Branch: `feature/p1-build`
Status: active

- [ ] P1.1 First build task (#3)
  - [ ] First subtask

## Detailed Next Steps Notes

1. Start P1.1.
"""

VALID_CHANGELOG: str = """# Change Log

## 2026-10-12

- Earlier entry.

## 2026-10-13

- Later entry.
"""

VALID_LLMS: str = """# Demo

> Agent index.

- [Index](index.html): home.
"""


def write(root: Path, rel: str, text: str) -> None:
    """Write text to a path under a root, creating the parent directories.

    Parameters
    ----------
    root : Path
        Directory that stands in for the repository root.
    rel : str
        Path of the file, relative to the root.
    text : str
        Text to write, encoded as UTF-8.
    """
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def make_repo(root: Path) -> Path:
    """Create a minimal repository that passes every check.

    Parameters
    ----------
    root : Path
        Directory to fill.

    Returns
    -------
    Path
        The same directory, now a valid repository.
    """
    for rel in cw.REQUIRED_FILES:
        write(root, rel, "placeholder\n")
    write(root, "ROADMAP.md", VALID_ROADMAP)
    write(root, "CHANGE_LOG.md", VALID_CHANGELOG)
    write(root, "docs/llms.txt", VALID_LLMS)
    write(root, "docs/index.rst", "Demo\n====\n")
    write(root, "README.md", "# Demo\n")
    return root


def error_codes(root: Path) -> set[str]:
    """Run the full repository check and return the codes of its errors.

    Parameters
    ----------
    root : Path
        Repository root to check.

    Returns
    -------
    set of str
        The finding codes of every error-severity finding.
    """
    return {diag.code for diag in cw.run(root) if diag.severity == "error"}


def test_valid_repository_has_no_errors(tmp_path: Path) -> None:
    """A repository that follows the workflow produces no errors."""
    assert error_codes(make_repo(tmp_path)) == set()


def test_missing_required_file_is_an_error(tmp_path: Path) -> None:
    """A missing required file produces a FILE-MISSING error."""
    root = make_repo(tmp_path)
    (root / "AGENTS.md").unlink()
    assert "FILE-MISSING" in error_codes(root)


def test_recommended_file_is_only_a_warning(tmp_path: Path) -> None:
    """A missing recommended file produces a warning and no error."""
    root = make_repo(tmp_path)
    diags = cw.run(root)
    assert any(d.code == "FILE-RECOMMENDED" and d.severity == "warning" for d in diags)
    assert "FILE-MISSING" not in error_codes(root)


def test_missing_roadmap_and_changelog_skip_their_checks(tmp_path: Path) -> None:
    """Missing roadmap and changelog files are reported once, not parsed as empty."""
    root = make_repo(tmp_path)
    (root / "ROADMAP.md").unlink()
    (root / "CHANGE_LOG.md").unlink()
    codes = error_codes(root)
    assert codes == {"FILE-MISSING"}


def test_phase_numbers_must_increase(tmp_path: Path) -> None:
    """Two phases with the same number produce a ROADMAP-PHASE-ORDER error."""
    root = make_repo(tmp_path)
    write(root, "ROADMAP.md", VALID_ROADMAP.replace("## Phase 1: Build", "## Phase 0: Build"))
    assert "ROADMAP-PHASE-ORDER" in error_codes(root)


def test_roadmap_without_phases_is_an_error(tmp_path: Path) -> None:
    """A roadmap with no phase headings produces a ROADMAP-NO-PHASES error."""
    root = make_repo(tmp_path)
    write(root, "ROADMAP.md", "# Roadmap\n\nNothing planned yet.\n")
    assert "ROADMAP-NO-PHASES" in error_codes(root)


def test_phase_metadata_lines_are_required(tmp_path: Path) -> None:
    """A phase without its Branch line produces a ROADMAP-PHASE-META error."""
    root = make_repo(tmp_path)
    text = VALID_ROADMAP.replace("Branch: `feature/p0-bootstrap`\n", "", 1)
    write(root, "ROADMAP.md", text)
    assert "ROADMAP-PHASE-META" in error_codes(root)


def test_duplicate_parent_line_is_ignored_after_the_first(tmp_path: Path) -> None:
    """A second Parent issue line in a phase does not replace the first."""
    root = make_repo(tmp_path)
    text = VALID_ROADMAP.replace("Parent issue: #2\n", "Parent issue: #2\nParent issue: #9\n")
    write(root, "ROADMAP.md", text)
    assert error_codes(root) == set()


def test_unknown_status_is_rejected(tmp_path: Path) -> None:
    """A status outside the allowed set produces a ROADMAP-STATUS error."""
    root = make_repo(tmp_path)
    write(root, "ROADMAP.md", VALID_ROADMAP.replace("Status: active", "Status: wip"))
    assert "ROADMAP-STATUS" in error_codes(root)


def test_active_phase_needs_a_parent_issue(tmp_path: Path) -> None:
    """An active phase whose parent is TBD produces a ROADMAP-PARENT-REQUIRED error."""
    root = make_repo(tmp_path)
    write(root, "ROADMAP.md", VALID_ROADMAP.replace("Parent issue: #2", "Parent issue: TBD"))
    assert "ROADMAP-PARENT-REQUIRED" in error_codes(root)


def test_planned_phase_may_wait_for_its_parent(tmp_path: Path) -> None:
    """A planned phase with TBD as its parent is allowed."""
    root = make_repo(tmp_path)
    assert "ROADMAP-PARENT-REQUIRED" not in error_codes(root)


def test_complete_phase_needs_checked_tasks(tmp_path: Path) -> None:
    """A complete phase with an unchecked task produces ROADMAP-COMPLETE-OPEN."""
    root = make_repo(tmp_path)
    text = VALID_ROADMAP.replace("Status: planned", "Status: complete")
    write(root, "ROADMAP.md", text.replace("Parent issue: TBD", "Parent issue: #1"))
    assert "ROADMAP-COMPLETE-OPEN" in error_codes(root)


def test_complete_phase_needs_checked_subtasks(tmp_path: Path) -> None:
    """A complete phase with an unchecked subtask produces ROADMAP-COMPLETE-OPEN."""
    root = make_repo(tmp_path)
    text = VALID_ROADMAP.replace("Status: active", "Status: complete")
    text = text.replace("- [ ] P1.1 First build task (#3)", "- [x] P1.1 First build task (#3)")
    write(root, "ROADMAP.md", text)
    diags = [d for d in cw.run(root) if d.code == "ROADMAP-COMPLETE-OPEN"]
    assert any("subtask" in d.message for d in diags)


def test_complete_phase_must_list_tasks(tmp_path: Path) -> None:
    """A complete phase with no tasks produces ROADMAP-COMPLETE-OPEN."""
    root = make_repo(tmp_path)
    text = VALID_ROADMAP.replace("- [ ] P0.1 First bootstrap task\n", "")
    text = text.replace("Parent issue: TBD", "Parent issue: #1").replace(
        "Status: planned", "Status: complete"
    )
    write(root, "ROADMAP.md", text)
    assert "lists no tasks" in " ".join(d.message for d in cw.run(root))


def test_task_must_belong_to_its_phase(tmp_path: Path) -> None:
    """A task whose ID names another phase produces a ROADMAP-TASK-ID error."""
    root = make_repo(tmp_path)
    text = VALID_ROADMAP.replace("- [ ] P0.1 First bootstrap task", "- [ ] P5.1 Misplaced task")
    write(root, "ROADMAP.md", text)
    assert "ROADMAP-TASK-ID" in error_codes(root)


def test_task_numbers_must_be_unique_within_a_phase(tmp_path: Path) -> None:
    """A repeated task number inside one phase produces a ROADMAP-TASK-ID error."""
    root = make_repo(tmp_path)
    text = VALID_ROADMAP.replace(
        "- [ ] P1.1 First build task (#3)\n",
        "- [ ] P1.1 First build task (#3)\n- [ ] P1.1 Duplicate task\n",
    )
    write(root, "ROADMAP.md", text)
    assert "ROADMAP-TASK-ID" in error_codes(root)


@pytest.mark.parametrize(
    ("row", "fragment"),
    [
        ("| P1 Build | #9 | `feature/p1-build` | active |", "parent issue"),
        ("| P1 Build | #2 | `feature/other` | active |", "branch"),
        ("| P1 Build | #2 | `feature/p1-build` | planned |", "status"),
    ],
)
def test_tracker_row_must_match_its_phase(tmp_path: Path, row: str, fragment: str) -> None:
    """Each tracker cell must agree with the phase section it describes."""
    root = make_repo(tmp_path)
    text = VALID_ROADMAP.replace("| P1 Build | #2 | `feature/p1-build` | active |", row)
    write(root, "ROADMAP.md", text)
    messages = [d.message for d in cw.run(root) if d.code == "ROADMAP-TRACKER"]
    assert any(fragment in message for message in messages)


def test_tracker_rows_must_not_repeat(tmp_path: Path) -> None:
    """Two tracker rows for the same phase produce a ROADMAP-TRACKER error."""
    root = make_repo(tmp_path)
    row = "| P1 Build | #2 | `feature/p1-build` | active |"
    write(root, "ROADMAP.md", VALID_ROADMAP.replace(row, row + "\n" + row))
    assert "ROADMAP-TRACKER" in error_codes(root)


def test_tracker_row_needs_a_phase_section(tmp_path: Path) -> None:
    """A tracker row for a phase with no section produces a ROADMAP-TRACKER error."""
    root = make_repo(tmp_path)
    ghost = "| P9 Ghost | TBD | `feature/p9-ghost` | planned |"
    write(root, "ROADMAP.md", VALID_ROADMAP.replace("| P1 Build", ghost + "\n| P1 Build"))
    messages = [d.message for d in cw.run(root) if d.code == "ROADMAP-TRACKER"]
    assert any("has no phase" in message for message in messages)


def test_phase_needs_a_tracker_row(tmp_path: Path) -> None:
    """A phase with no tracker row produces a ROADMAP-TRACKER error."""
    root = make_repo(tmp_path)
    write(
        root,
        "ROADMAP.md",
        VALID_ROADMAP.replace("| P1 Build | #2 | `feature/p1-build` | active |\n", ""),
    )
    messages = [d.message for d in cw.run(root) if d.code == "ROADMAP-TRACKER"]
    assert any("has no tracker row" in message for message in messages)


def test_short_tracker_row_is_rejected(tmp_path: Path) -> None:
    """A tracker row with fewer than four cells produces a ROADMAP-TRACKER error."""
    root = make_repo(tmp_path)
    short = "| P0 Bootstrap | TBD | `feature/p0-bootstrap` |"
    write(
        root,
        "ROADMAP.md",
        VALID_ROADMAP.replace("| P0 Bootstrap | TBD | `feature/p0-bootstrap` | planned |", short),
    )
    messages = [d.message for d in cw.run(root) if d.code == "ROADMAP-TRACKER"]
    assert any("needs Phase, Parent issue, Branch, and Status" in m for m in messages)


def test_changelog_must_be_in_date_order(tmp_path: Path) -> None:
    """A changelog entry older than the one above it produces CHANGELOG-ORDER."""
    root = make_repo(tmp_path)
    text = "# Change Log\n\n## 2026-10-13\n\n- New.\n\n## 2026-10-12\n\n- Old.\n"
    write(root, "CHANGE_LOG.md", text)
    assert "CHANGELOG-ORDER" in error_codes(root)


def test_changelog_headings_must_be_dates(tmp_path: Path) -> None:
    """A changelog heading that is not an ISO date produces CHANGELOG-DATE."""
    root = make_repo(tmp_path)
    write(root, "CHANGE_LOG.md", "# Change Log\n\n## Unreleased\n\n- Pending.\n")
    codes = error_codes(root)
    assert "CHANGELOG-DATE" in codes
    assert "CHANGELOG-EMPTY" in codes


def test_changelog_needs_a_dated_entry(tmp_path: Path) -> None:
    """A changelog with no dated headings produces CHANGELOG-EMPTY."""
    root = make_repo(tmp_path)
    write(root, "CHANGE_LOG.md", "# Change Log\n\nNo entries yet.\n")
    assert "CHANGELOG-EMPTY" in error_codes(root)


def test_broken_local_link_is_an_error(tmp_path: Path) -> None:
    """A local Markdown link to a missing file produces LINK-BROKEN."""
    root = make_repo(tmp_path)
    write(root, "README.md", "# Demo\n\nSee [the guide](docs/missing.rst).\n")
    assert "LINK-BROKEN" in error_codes(root)


def test_local_link_to_an_existing_file_passes(tmp_path: Path) -> None:
    """A local link to an existing file, with a fragment, produces no error."""
    root = make_repo(tmp_path)
    write(root, "README.md", "# Demo\n\nSee [the index](docs/index.rst#top).\n")
    assert error_codes(root) == set()


@pytest.mark.parametrize(
    "link",
    [
        "[site](https://example.org/x)",
        "[mail](mailto:someone@example.org)",
        "[top](#demo)",
        "[template](<placeholder>)",
        "[token](" + TOKEN_OPEN + "PROJECT_NAME" + TOKEN_CLOSE + ")",
    ],
)
def test_external_anchor_and_placeholder_links_are_ignored(tmp_path: Path, link: str) -> None:
    """Links that are not local paths are not resolved."""
    root = make_repo(tmp_path)
    write(root, "README.md", f"# Demo\n\n{link}\n")
    assert error_codes(root) == set()


def test_llms_index_must_point_at_existing_pages(tmp_path: Path) -> None:
    """An llms.txt link with no matching page produces LLMS-BROKEN."""
    root = make_repo(tmp_path)
    write(root, "docs/llms.txt", "# Demo\n\n- [Ghost](ghost.html): no page.\n")
    assert "LLMS-BROKEN" in error_codes(root)


def test_llms_index_skips_external_links(tmp_path: Path) -> None:
    """An external link in llms.txt is not resolved against the docs tree."""
    root = make_repo(tmp_path)
    write(root, "docs/llms.txt", "# Demo\n\n- [Site](https://example.org/docs): elsewhere.\n")
    assert error_codes(root) == set()


def test_llms_index_accepts_non_html_targets(tmp_path: Path) -> None:
    """An llms.txt link to a non-HTML file resolves against docs/ directly."""
    root = make_repo(tmp_path)
    write(root, "docs/guides/notes.md", "# Notes\n")
    write(root, "docs/llms.txt", "# Demo\n\n- [Notes](guides/notes.md): a page.\n")
    assert error_codes(root) == set()


def test_llms_index_missing_non_html_target_is_reported(tmp_path: Path) -> None:
    """A non-HTML llms.txt link to a missing file produces LLMS-BROKEN."""
    root = make_repo(tmp_path)
    write(root, "docs/llms.txt", "# Demo\n\n- [Notes](guides/absent.md): a page.\n")
    assert "LLMS-BROKEN" in error_codes(root)


def test_missing_llms_index_is_only_a_file_error(tmp_path: Path) -> None:
    """Without docs/llms.txt, only FILE-MISSING is reported for it."""
    root = make_repo(tmp_path)
    (root / "docs" / "llms.txt").unlink()
    assert error_codes(root) == {"FILE-MISSING"}


def test_control_characters_are_rejected(tmp_path: Path) -> None:
    """A form feed inside a line produces CONTROL-CHAR."""
    root = make_repo(tmp_path)
    write(root, "README.md", f"# Demo\n\nBranch: eature{FORM_FEED}/p0\n")
    assert "CONTROL-CHAR" in error_codes(root)


def test_byte_order_mark_is_rejected(tmp_path: Path) -> None:
    """A byte-order mark at the start of a file produces BOM."""
    root = make_repo(tmp_path)
    write(root, "README.md", f"{BOM}# Demo\n")
    assert "BOM" in error_codes(root)


def test_carriage_returns_are_rejected(tmp_path: Path) -> None:
    """A CRLF line ending produces a CONTROL-CHAR finding for the file."""
    root = make_repo(tmp_path)
    (root / "README.md").write_bytes(b"# Demo\r\n")
    diags = [d for d in cw.run(root) if d.code == "CONTROL-CHAR"]
    assert diags
    assert diags[0].line == 0


@pytest.mark.parametrize(
    "prefix",
    [HOME_PREFIX, MAC_PREFIX, WINDOWS_PREFIX],
)
def test_personal_paths_are_rejected(tmp_path: Path, prefix: str) -> None:
    """Home-directory paths for Linux, macOS, and Windows produce PATH-PERSONAL."""
    root = make_repo(tmp_path)
    write(root, "README.md", f"Clone into {prefix}projects/demo\n")
    assert "PATH-PERSONAL" in error_codes(root)


def test_placeholder_paths_pass(tmp_path: Path) -> None:
    """A placeholder in place of a home directory is allowed."""
    root = make_repo(tmp_path)
    write(root, "README.md", "Clone into /home/<user>/projects/demo\n")
    assert "PATH-PERSONAL" not in error_codes(root)


@pytest.mark.parametrize(
    "secret",
    [GITHUB_TOKEN, FINE_GRAINED_TOKEN, PYPI_TOKEN, AWS_KEY, PRIVATE_KEY, CREDENTIAL],
)
def test_secret_shapes_are_rejected(tmp_path: Path, secret: str) -> None:
    """Each credential shape produces SECRET-SHAPED."""
    root = make_repo(tmp_path)
    write(root, "notes/leak.md", f"token {secret}\n")
    assert "SECRET-SHAPED" in error_codes(root)


def test_control_characters_in_a_symlink_are_not_scanned(tmp_path: Path) -> None:
    """Symbolic links are skipped by the repository scan."""
    root = make_repo(tmp_path / "repo")
    outside = tmp_path / "outside.md"
    outside.write_text(f"Bad{FORM_FEED}\n", encoding="utf-8")
    (root / "linked.md").symlink_to(outside)
    assert "CONTROL-CHAR" not in error_codes(root)


def test_present_recommended_file_is_not_warned(tmp_path: Path) -> None:
    """A recommended file that exists produces no FILE-RECOMMENDED warning for it."""
    root = make_repo(tmp_path)
    write(root, "LICENSE", "MIT\n")
    warned = {d.path for d in cw.run(root) if d.code == "FILE-RECOMMENDED"}
    assert "LICENSE" not in warned
    assert "CONTRIBUTING.md" in warned


def test_oversized_files_are_not_scanned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Files above the scan size limit are skipped."""
    root = make_repo(tmp_path)
    write(root, "notes/large.md", f"Bad{FORM_FEED}\n")
    monkeypatch.setattr(cw, "MAX_SCAN_BYTES", 4)
    assert "CONTROL-CHAR" not in error_codes(root)


def test_non_utf8_files_are_not_scanned(tmp_path: Path) -> None:
    """Binary files that are not valid UTF-8 are skipped by the repository scan."""
    root = make_repo(tmp_path)
    (root / "data.bin").write_bytes(b"\xff\xfe" + GITHUB_TOKEN.encode("ascii"))
    assert error_codes(root) == set()


def test_check_files_reports_missing_and_undecodable_files(tmp_path: Path) -> None:
    """The file mode reports absent files and files that are not UTF-8."""
    missing = tmp_path / "absent.md"
    binary = tmp_path / "binary.md"
    binary.write_bytes(b"\xff\xfe")
    codes = {d.code for d in cw.check_files([missing, binary])}
    assert codes == {"FILE-MISSING", "ENCODING"}


def test_files_mode_checks_a_single_body(tmp_path: Path) -> None:
    """The file mode fails a body with a control character, and passes a clean one."""
    body = tmp_path / "body.md"
    body.write_text(f"Branch: eature{FORM_FEED}/p0\n", encoding="utf-8")
    assert cw.main(["--files", str(body)]) == 1
    clean = tmp_path / "clean.md"
    clean.write_text("Branch: `feature/p0`\n", encoding="utf-8")
    assert cw.main(["--files", str(clean)]) == 0


def test_main_reports_a_clean_repository(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command prints a summary line and returns 0 for a clean repository."""
    assert cw.main([str(make_repo(tmp_path))]) == 0
    assert "0 error(s), 4 warning(s)" in capsys.readouterr().out


def test_main_reports_errors_with_exit_status_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command prints each finding and returns 1 when any error is present."""
    root = make_repo(tmp_path)
    (root / "AGENTS.md").unlink()
    assert cw.main([str(root)]) == 1
    output = capsys.readouterr().out
    assert "FILE-MISSING" in output
    assert "error(s)" in output


def test_main_rejects_a_root_that_is_not_a_directory(tmp_path: Path) -> None:
    """A root path that does not exist makes the command exit with status 2."""
    with pytest.raises(SystemExit) as excinfo:
        cw.main([str(tmp_path / "absent")])
    assert excinfo.value.code == 2


def test_render_shows_the_location_only_when_a_line_is_known() -> None:
    """A finding with a line number shows it, and a whole-file finding shows the path."""
    with_line = cw.Diagnostic("error", "BOM", "README.md", 3, "message")
    whole_file = cw.Diagnostic("error", "CONTROL-CHAR", "README.md", 0, "message")
    assert "README.md:3" in with_line.render()
    assert "README.md:0" not in whole_file.render()
    assert whole_file.render().split()[1] == "CONTROL-CHAR"


def test_python_version_supports_the_documented_grammar() -> None:
    """The validator needs Python 3.11 or later, which the CI matrix covers."""
    assert sys.version_info >= (3, 11)


GIT_REQUIRED = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(root: Path, *args: str) -> None:
    """Run a git command in a directory, for the file-set tests.

    Parameters
    ----------
    root : Path
        Directory to run the command in.
    *args : str
        Arguments passed to git after ``-C``.
    """
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@GIT_REQUIRED
def test_git_file_set_ignores_caches_that_git_ignores(tmp_path: Path) -> None:
    """A file under an ignored directory is not scanned, even if it holds a personal path."""
    root = make_repo(tmp_path)
    git(root, "init", "-q")
    write(root, ".gitignore", "cache/\n")
    write(root, "cache/leak.txt", f"Clone into {HOME_PREFIX}runner/work\n")
    assert "PATH-PERSONAL" not in error_codes(root)


@GIT_REQUIRED
def test_git_file_set_scans_untracked_files_that_are_not_ignored(tmp_path: Path) -> None:
    """A new file that a commit would include is scanned before it is staged."""
    root = make_repo(tmp_path)
    git(root, "init", "-q")
    write(root, "notes/new.md", f"Bad{FORM_FEED}\n")
    paths = {d.path for d in cw.run(root) if d.code == "CONTROL-CHAR"}
    assert "notes/new.md" in paths


@GIT_REQUIRED
def test_a_directory_inside_another_repository_is_walked(tmp_path: Path) -> None:
    """A generated project inside another repository is walked, not listed by the outer git."""
    outer = tmp_path / "outer"
    outer.mkdir()
    git(outer, "init", "-q")
    inner = make_repo(outer / "tmp" / "inner")
    write(inner, "notes/bad.md", f"Bad{FORM_FEED}\n")
    assert "CONTROL-CHAR" in error_codes(inner)


def test_missing_git_falls_back_to_walking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without git on the path, the scan walks the tree and still finds defects."""
    root = make_repo(tmp_path)
    write(root, "notes/bad.md", f"Bad{FORM_FEED}\n")
    monkeypatch.setenv("PATH", "")
    assert "CONTROL-CHAR" in error_codes(root)


def test_failed_listing_falls_back_to_walking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If git fails to list files, the scan walks the tree instead of trusting an empty list."""
    root = make_repo(tmp_path)
    write(root, "notes/bad.md", f"Bad{FORM_FEED}\n")

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, stdout=f"{root}\n", stderr="")
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="fatal\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert "CONTROL-CHAR" in error_codes(root)


def test_walked_trees_skip_generated_caches(tmp_path: Path) -> None:
    """A walked tree skips generated caches, such as a Hypothesis cache."""
    root = make_repo(tmp_path)
    write(root, ".hypothesis/constants/cache", f"Runner path {HOME_PREFIX}runner/work\n")
    write(root, "runtime/dashboard.md", f"Runner path {HOME_PREFIX}runner/work\n")
    assert error_codes(root) == set()


GITIGNORE_FORMS: str = """# Comments and blank lines are not rules.

__pycache__/
*.py[cod]
.venv/
_build/
docs/_build/
data/private/
tmp/
runtime/
.DS_Store
Thumbs.db
.coverage
.coverage.*
coverage.xml
.hypothesis/
"""


def test_walk_mode_skips_ignored_files_even_when_they_are_utf8_text(tmp_path: Path) -> None:
    """A file that the root .gitignore ignores is not scanned, even though it is valid UTF-8."""
    root = make_repo(tmp_path)
    write(root, ".gitignore", "local-notes.log\n")
    write(root, "local-notes.log", f"Clone into {HOME_PREFIX}someone/projects\n")
    assert "PATH-PERSONAL" not in error_codes(root)


def test_walk_mode_still_scans_files_the_gitignore_does_not_ignore(tmp_path: Path) -> None:
    """A file that the root .gitignore does not match is still scanned."""
    root = make_repo(tmp_path)
    write(root, ".gitignore", "local-notes.log\n")
    write(root, "notes/open.log", f"Clone into {HOME_PREFIX}someone/projects\n")
    assert "PATH-PERSONAL" in error_codes(root)


def test_walk_mode_prunes_ignored_directories_and_generated_ones(tmp_path: Path) -> None:
    """Ignored directories are not walked, and neither are the generated ones in SKIP_DIRS."""
    root = make_repo(tmp_path)
    write(root, ".gitignore", "cache/\n")
    write(root, "cache/leak.md", f"Clone into {HOME_PREFIX}someone/projects\n")
    write(root, ".venv/lib/leak.md", f"Clone into {HOME_PREFIX}someone/projects\n")
    assert "PATH-PERSONAL" not in error_codes(root)


def test_a_trailing_slash_rule_leaves_a_same_named_file_scanned(tmp_path: Path) -> None:
    """A trailing-slash rule applies to directories, so a file with the same name is scanned."""
    root = make_repo(tmp_path)
    write(root, ".gitignore", "cache/\n")
    write(root, "docs/cache", f"Clone into {HOME_PREFIX}someone/projects\n")
    assert "PATH-PERSONAL" in error_codes(root)


def test_an_anchored_rule_matches_only_at_the_root(tmp_path: Path) -> None:
    """A rule with a leading slash applies at the root only, so a nested file is still scanned."""
    root = make_repo(tmp_path)
    write(root, ".gitignore", "/notes.md\n")
    write(root, "notes.md", f"Clone into {HOME_PREFIX}someone/projects\n")
    write(root, "docs/notes.md", f"Clone into {HOME_PREFIX}someone/projects\n")
    flagged = [diag.path for diag in cw.run(root) if diag.code == "PATH-PERSONAL"]
    assert flagged == ["docs/notes.md"]


def test_a_later_negation_re_includes_a_file(tmp_path: Path) -> None:
    """A later ! rule re-includes a file that an earlier rule ignores."""
    root = make_repo(tmp_path)
    write(root, ".gitignore", "*.log\n!keep.log\n")
    write(root, "keep.log", f"Clone into {HOME_PREFIX}someone/projects\n")
    write(root, "drop.log", f"Clone into {HOME_PREFIX}someone/projects\n")
    flagged = [diag.path for diag in cw.run(root) if diag.code == "PATH-PERSONAL"]
    assert flagged == ["keep.log"]


def test_a_negation_cannot_reinclude_a_file_in_an_ignored_directory(tmp_path: Path) -> None:
    """Git does not re-include a file whose directory is ignored, and the walk follows that."""
    root = make_repo(tmp_path)
    write(root, ".gitignore", "build/\n!build/keep.md\n")
    write(root, "build/keep.md", f"Clone into {HOME_PREFIX}someone/projects\n")
    assert "PATH-PERSONAL" not in error_codes(root)


def test_only_the_root_gitignore_is_read(tmp_path: Path) -> None:
    """A .gitignore in a subdirectory is not read, so its rules do not apply."""
    root = make_repo(tmp_path)
    write(root, "docs/.gitignore", "private.md\n")
    write(root, "docs/private.md", f"Clone into {HOME_PREFIX}someone/projects\n")
    assert "PATH-PERSONAL" in error_codes(root)


def test_a_root_without_a_gitignore_has_no_rules(tmp_path: Path) -> None:
    """A root with no .gitignore has no rules."""
    assert cw.load_ignore_rules(tmp_path) == []


def test_gitignore_forms_used_by_this_repository_are_applied(tmp_path: Path) -> None:
    """Each form in the repository's ignore file matches the paths it should, and no others."""
    write(tmp_path, ".gitignore", GITIGNORE_FORMS)
    rules = cw.load_ignore_rules(tmp_path)
    cases: list[tuple[str, bool, bool]] = [
        ("pkg/__pycache__/mod.py", False, True),
        ("pkg/mod.pyc", False, True),
        ("pkg/mod.py", False, False),
        ("docs/_build/index.html", False, True),
        ("other/_build/index.html", False, True),
        ("data/private/row.csv", False, True),
        ("data/public/row.csv", False, False),
        (".venv/bin/python", False, True),
        ("tmp", True, True),
        ("notes/tmp.md", False, False),
        ("runtime", True, True),
        ("runtime", False, False),
        ("docs/Thumbs.db", False, True),
        (".DS_Store", False, True),
        (".coverage.worker", False, True),
        ("coverage.xml", False, True),
        ("docs/index.rst", False, False),
        ("scripts/check_workflow.py", False, False),
    ]
    for rel, is_dir, expected in cases:
        assert cw.is_ignored(rel, is_dir=is_dir, rules=rules) is expected, rel


def test_character_classes_and_single_character_wildcards(tmp_path: Path) -> None:
    """A ! class negates, ? matches one character, and an unclosed bracket is literal."""
    write(tmp_path, ".gitignore", "file[!0-9].txt\nnote?.md\ndocs/*.rst\nbad[name\nx[]\n")
    rules = cw.load_ignore_rules(tmp_path)
    assert cw.is_ignored("fileX.txt", is_dir=False, rules=rules)
    assert not cw.is_ignored("file1.txt", is_dir=False, rules=rules)
    assert cw.is_ignored("note1.md", is_dir=False, rules=rules)
    assert not cw.is_ignored("note12.md", is_dir=False, rules=rules)
    assert cw.is_ignored("docs/index.rst", is_dir=False, rules=rules)
    assert not cw.is_ignored("docs/sub/index.rst", is_dir=False, rules=rules)
    assert cw.is_ignored("bad[name", is_dir=False, rules=rules)
    assert cw.is_ignored("x[]", is_dir=False, rules=rules)


def test_lines_the_reader_cannot_apply_are_skipped(tmp_path: Path) -> None:
    """Escapes, ** patterns, bare slashes, bad classes, comments, and blank lines give no rule."""
    write(tmp_path, ".gitignore", "**/cache.md\n\\#literal\n/\n[z-a]\n# a comment\n\n")
    assert cw.load_ignore_rules(tmp_path) == []


def test_the_last_matching_rule_decides(tmp_path: Path) -> None:
    """The last matching rule decides, so a rule after a negation can ignore a file again."""
    write(tmp_path, ".gitignore", "*.log\n!keep.log\n*.log\n")
    rules = cw.load_ignore_rules(tmp_path)
    assert cw.is_ignored("keep.log", is_dir=False, rules=rules)


def test_a_path_under_an_ignored_directory_is_ignored(tmp_path: Path) -> None:
    """A path under an ignored directory stays ignored, even when a later rule names it."""
    write(tmp_path, ".gitignore", "build/\n!build/keep.md\n")
    rules = cw.load_ignore_rules(tmp_path)
    assert cw.is_ignored("build/keep.md", is_dir=False, rules=rules)
    assert not cw.is_ignored("keep.md", is_dir=False, rules=rules)


def test_an_unsupported_negation_cannot_hide_a_file(tmp_path: Path) -> None:
    """A negation the reader cannot apply cannot leave a file hidden by an earlier rule."""
    root = make_repo(tmp_path)
    write(root, ".gitignore", "*.log\n!important/**/audit.log\n")
    write(root, "important/x/audit.log", f"Clone into {HOME_PREFIX}someone/projects\n")
    assert "PATH-PERSONAL" in error_codes(root)


def test_an_unsupported_negation_reincludes_every_path(tmp_path: Path) -> None:
    """A negated line the reader cannot apply becomes one rule that re-includes every path."""
    write(tmp_path, ".gitignore", "*.log\n!**/keep.log\n")
    rules = cw.load_ignore_rules(tmp_path)
    assert [rule.negated for rule in rules] == [False, True]
    assert not cw.is_ignored("docs/keep.log", is_dir=False, rules=rules)
