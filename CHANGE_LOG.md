# Change Log

Newest entries last. Each milestone gets a dated entry that matches the roadmap status and the GitHub issue comments.

## 2026-10-17

- Created Microsite Annotation Flywheel from the UBC-FRESH dev workflow seed. The agent contract, roadmap, docs skeleton, and validator came from the seed. Every phase is planned, and no parent issues exist yet.
- P0.1 adapted the roadmap to the six flywheel phases (P0 scaffold, P1 sync, P2 train, P3 prelabel and upload, P4 loop bookkeeping and convergence, P5 runbook and acceptance) and added the project-specific rules to `AGENTS.md`.
- P0.2 added credential and data-artifact guardrails to `.gitignore`, recorded the Roboflow evidence in `notes/reference/roboflow-canonical-projects.md`, and recorded the design in `notes/adr/0001-flywheel-design.md`.
- Verified against the live Roboflow API: `a58-sandbox` now holds 18 annotated images (97 `a58-sandbox-microsite` instances) and 732 unannotated images, so round 1 self-hosts in that project.
- P1–P4 built the flywheel under `scripts/`: `config.py`, `geometry.py`, `roboflow_api.py`, `sync.py`, `model.py`, `train.py`, `prelabel.py`, `state.py`, `convergence.py`, and the `loop.py` CLI (`sync`, `train`, `prelabel`, `compare`, `status`). `config.yaml` names the workspace, projects, class, paths, and round.
- The upload guard is enforced in code: `prelabel` re-reads each target image and skips any that already carries an annotation, and every write uses `annotation_overwrite=False` and `is_prediction=True`.
- Dashboard result on this change set: GREEN (lint, format, types, 267 tests at 98% branch coverage, validator, docs).
- Gate and tool configuration change: added `requirements.txt` (numpy, opencv-headless, PyYAML, requests, roboflow) and `requirements-train.txt` (torch stack), wired both into `requirements-dev.txt`; added mypy import-ignores for the untyped third-party modules and waived ANN401 for `scripts/roboflow_api.py` and `tests/`. Reason: the flywheel legitimately depends on those libraries, and the SDK/JSON seam is honestly dynamically typed.
- Deviation recorded: the code landed on `main` before its GitHub issues because `gh` is not yet authenticated on this container. Phase and task issues are to be created and back-linked once `gh auth login` is done.
