# Contributing to Microsite Annotation Flywheel

Read `AGENTS.md` and `ROADMAP.md` before non-trivial work.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
pre-commit install
```

The pre-commit hooks call `python` from `PATH`, so commit from a shell where the environment is
active. A commit from a shell without it runs the hooks in the system Python, which lacks the dev
tools, and the hooks fail.

## The green dashboard

Run `python scripts/dashboard.py` before each commit. It must be green. Any change from green is
investigated and fixed immediately, with no exceptions. The pre-commit hooks run the same checks on
each commit.

## Workflow

- Work on the branch for the active phase, named `feature/p<n>-<slug>`.
- Start each commit message with the task ID, for example `P0.1 add license`.
- Write every function with full type annotations and a numpy-style docstring, and add unit tests for it
  in the same change set.
- Keep `ROADMAP.md`, `CHANGE_LOG.md`, `notes/`, and the matching GitHub issues in step with the work.
- Open pull requests with `Part of #<parent issue>` in the body. CI must be green. A maintainer merges human-authored pull requests. An agent-run phase closes out as described in `AGENTS.md`.

## Control characters and public safety

- Write issue and pull request bodies to files, and post them with `--body-file`. Do not build them from
  escaped strings.
- Use LF line endings.
- Run `python scripts/check_workflow.py` before you push. To check one body file, run
  `python scripts/check_workflow.py --files path/to/body.md`.
- Keep private data, credentials, raw transcripts, and machine-specific paths out of the repository.
