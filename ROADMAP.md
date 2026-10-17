# Microsite Annotation Flywheel Roadmap

This roadmap is the plan of record and the issue tracker map for the Microsite Annotation
Flywheel. Keep it in step with the GitHub issues, `CHANGE_LOG.md`, and `notes/`.
`scripts/check_workflow.py` parses the phase headers, status lines, and checkboxes below,
so keep their format.

Every phase starts as `planned` with `Parent issue: TBD`. Activate a phase only after its
parent issue exists and the maintainer approves it.

## Issue Tracker Map

| Phase | Parent issue | Branch | Status |
| --- | --- | --- | --- |
| P0 Scaffold | #1 | `feature/p0-scaffold` | active |
| P1 Sync | #5 | `feature/p1-sync` | planned |
| P2 Train | #10 | `feature/p2-train` | planned |
| P3 Prelabel and upload | #14 | `feature/p3-prelabel-upload` | planned |
| P4 Loop bookkeeping and convergence | #19 | `feature/p4-loop-convergence` | planned |
| P5 Runbook and acceptance | #24 | `feature/p5-runbook-acceptance` | planned |

## Lifecycle Map

| Stage | Phases | Exit criteria |
| --- | --- | --- |
| Scope and design | P0 | The seed is adapted, the guardrails hold, and the dashboard is green |
| Build | P1, P2, P3 | Each task committed with its task ID; each commit leaves the dashboard green; docs updated in the same change set |
| Verify | P4, P5 | The dry loop runs end to end; the acceptance evidence is in the close-out report |
| Release and deploy | P5 (procedure) | A tag and release notes, published from an approved close-out |
| Operate and maintain | P5 (procedure) | The loop reruns for a new round or a new microsite definition without code changes |

## Conventions

- A phase is `P<n>`. A task is `P<n>.<m>`. A subtask is a checklist item inside the task's issue body. Three levels at most.
- Status values are `planned`, `active`, `pr-open`, `complete`, and `deferred`.
- A phase becomes `complete` only after its pull request merges to `main` and its parent issue closes.
- The operator-facing surface is four commands: `sync`, `train`, `prelabel`, `status`. The CLIs stay thin wrappers over the importable functions in `scripts/`.

## Phase 0: Scaffold

Parent issue: #1
Branch: `feature/p0-scaffold`
Status: active

Goal: adapt the seed contract to this project, install the credential and data guardrails, and
make the first pull request reviewable.

- [x] P0.1 Adapt `AGENTS.md`, `README.md`, and this roadmap to the flywheel (#2)
- [x] P0.2 Add the credential and data-artifact guardrails, and record the Roboflow API evidence in `notes/` (#3)
- [x] P0.3 Confirm the validator and the full dashboard pass on the scaffold commit (#4)

Acceptance criteria:

- The dashboard is green and the validator reports zero errors on the scaffold commit.

## Phase 1: Sync

Parent issue: #5
Branch: `feature/p1-sync`
Status: planned

Goal: pull a project's annotations and image lists into a local dataset directory with clear
provenance, so `train` has a reproducible input.

- [x] P1.1 Define `config.yaml` and the loader that validates it (#6)
- [x] P1.2 Implement the Roboflow client that lists images, reads annotation status, and downloads annotations (#7)
- [x] P1.3 Implement `sync` with provenance (project, round, timestamp, counts)
- [x] P1.4 Unit and contract tests for the configuration loader and the client (#9)

Acceptance criteria:

- `sync` writes a local dataset plus a provenance record, and reports annotated and unannotated counts.

## Phase 2: Train

Parent issue: #10
Branch: `feature/p2-train`
Status: planned

Goal: train the pre-labeller from the synced labels with the proven recipe, and record the
checkpoint's provenance.

- [x] P2.1 Implement the U-Net ResNet34 + focal/dice training recipe behind `train` (#11)
- [x] P2.2 Record checkpoint provenance: dataset round, recipe, and metrics (#12)
- [x] P2.3 Unit tests for the training configuration, dataset, and metrics (#13)

Acceptance criteria:

- `train` writes a checkpoint plus a metrics-and-provenance record, and prints validation IoU per epoch.

## Phase 3: Prelabel and upload

Parent issue: #14
Branch: `feature/p3-prelabel-upload`
Status: planned

Goal: run the pre-labeller over a project's unannotated images and upload predictions as
editable pre-labels, without ever touching an existing annotation.

- [x] P3.1 Implement the binary-mask to polygon conversion (#15)
- [x] P3.2 Implement the annotation-free verification that guards every upload (#16)
- [x] P3.3 Implement `prelabel` to predict on unannotated images and upload pre-labels (#17)
- [x] P3.4 Unit and contract tests for the conversion, the guard, and the uploader (#18)

Acceptance criteria:

- `prelabel` uploads to annotation-free images only, and reports exactly which images it wrote and which it skipped.

## Phase 4: Loop bookkeeping and convergence

Parent issue: #19
Branch: `feature/p4-loop-convergence`
Status: planned

Goal: track rounds so the loop can run repeatedly, and compute how much the human edited the
pre-labels so the operator can decide when to stop.

- [x] P4.1 Implement the `loop_state.json` per-round record (#20)
- [x] P4.2 Implement the object-level and pixel-level convergence metrics (#21)
- [x] P4.3 Implement `status` and the per-round trend table (#22)
- [x] P4.4 Unit tests for the state store and the convergence metrics (#23)

Acceptance criteria:

- `status` prints a per-round trend table with object-level and pixel-level convergence.

## Phase 5: Runbook and acceptance

Parent issue: #24
Branch: `feature/p5-runbook-acceptance`
Status: planned

Goal: hand the operator a runbook, run one full dry loop small-scale, and record the evidence.

- [ ] P5.1 Write `RUNBOOK.md` for a non-coder operator (#25)
- [ ] P5.2 Run the dry loop on a handful of images and record the acceptance evidence (#26)
- [ ] P5.3 Publish the docs and prepare the phase close-out (#27)

Acceptance criteria:

- A dry loop syncs annotated images, trains briefly, uploads pre-labels to unannotated images only, and shows the status table.

## Detailed Next Steps Notes

1. Create the GitHub repository and the Phase 0 parent issue. This needs maintainer approval.
2. Activate Phase 0 by setting its parent issue number and its status to `active`, here and in the tracker map.
3. The operator-facing entry point is `scripts/loop.py`; the runbook in `docs/guides/runbook.md` targets a non-coder.
