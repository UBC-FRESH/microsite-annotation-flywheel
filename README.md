# Microsite Annotation Flywheel

Provide a scriptable human-in-the-loop annotation flywheel for microsite classification from drone imagery: sync Roboflow annotations, train the pre-labeller, upload editable pre-labels to unannotated images, and track per-round convergence in support of precision aerial reforestation.

## Status

Phases P0 to P4 are built: the flywheel code, its tests, and the planning surfaces pass the green dashboard. Phase P5 is the runbook and the acceptance run. `ROADMAP.md` lists the phases; the GitHub issues, once created, carry the live status.

## How this project works

This project follows the UBC-FRESH dev workflow. Work is planned as phases, tasks, and subtasks in `ROADMAP.md`, mirrored in GitHub issues, and recorded in `CHANGE_LOG.md`. The agent operating contract is in `AGENTS.md`. Read `AGENTS.md`, then `ROADMAP.md`, before you change anything.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest
sphinx-build -b html docs _build/html -W
python scripts/check_workflow.py
```

## Operating the loop

The operator runs four commands, written up for a non-coder in [`RUNBOOK.md`](RUNBOOK.md):

```bash
python scripts/loop.py sync       # download the latest annotations
python scripts/loop.py train      # train the pre-labeller
python scripts/loop.py prelabel   # upload editable pre-labels
python scripts/loop.py status     # show the per-round trend
```

`config.yaml` names the workspace, the projects, the class, the paths, and the round. The
Roboflow key is read from a gitignored `.env` as `ROBOFLOW_API_KEY`.

## Repository map

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Operating contract for coding agents |
| `ROADMAP.md` | Plan of record and issue tracker map |
| `CHANGE_LOG.md` | Dated narrative of changes |
| `RUNBOOK.md` | How the operator runs the loop |
| `config.yaml` | Workspace, projects, class, paths, and round |
| `scripts/loop.py` | The operator CLI: sync, train, prelabel, compare, status |
| `docs/` | Documentation site |
| `notes/` | Decisions (ADRs), reference notes, and focused plans |
| `scripts/check_workflow.py` | Workflow validator |
| `tests/` | pytest suite, including a check that the planning surfaces pass the validator |

## License

Choose and add a license before the first public release.
