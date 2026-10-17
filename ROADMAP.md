# Microsite Annotation Flywheel Roadmap

This roadmap is the plan of record and the issue tracker map for Microsite Annotation Flywheel. Keep it in step with the GitHub issues, `CHANGE_LOG.md`, and `notes/`. `scripts/check_workflow.py` parses the phase headers, status lines, and checkboxes below, so keep their format.

Every phase starts as `planned` with `Parent issue: TBD`. Activate a phase only after its parent issue exists and the maintainer approves it.

## Issue Tracker Map

| Phase | Parent issue | Branch | Status |
| --- | --- | --- | --- |
| P0 Bootstrap | TBD | `feature/p0-bootstrap` | planned |
| P1 Scope and design | TBD | `feature/p1-scope-design` | planned |
| P2 Contracts and prototype | TBD | `feature/p2-contracts-prototype` | planned |
| P3 Build | TBD | `feature/p3-build` | planned |
| P4 Verify | TBD | `feature/p4-verify` | planned |
| P5 Release and deploy | TBD | `feature/p5-release` | planned |
| P6 Operate and maintain | TBD | `feature/p6-operate` | planned |

## Lifecycle Map

| Stage | Phases | Exit criteria |
| --- | --- | --- |
| Scope and design | P0, P1 | Goal, scope, and out-of-scope items recorded; decisions in ADRs, each accepted or deferred |
| Build | P2, P3 | Each task committed with its task ID; each commit leaves the dashboard green; docs updated in the same change set |
| Verify | P4 | The dashboard and CI pass on the head commit; each acceptance criterion has evidence in the close-out report |
| Release and deploy | P5 | Tag, release notes, and Pages live; a clean checkout verifies them; each release action approved on its own |
| Operate and maintain | P6 | Review cadence held; incoming ideas triaged; retirement decided and recorded as an ADR |
| Retire | P6 (procedure) | Archive notice, final changelog entry, and issues closed |

## Conventions

- A phase is `P<n>`. A task is `P<n>.<m>`. A subtask is a checklist item inside the task's issue body. Three levels at most.
- Status values are `planned`, `active`, `pr-open`, `complete`, and `deferred`.
- A phase becomes `complete` only after its pull request merges to `main` and its parent issue closes.

## Phase 0: Bootstrap

Parent issue: TBD
Branch: `feature/p0-bootstrap`
Status: planned

Goal: adapt the contract to this project, and make the first pull request reviewable.

- [ ] P0.1 Adapt `AGENTS.md` and `README.md` to this project
- [ ] P0.2 Confirm the validator and the docs build pass

Acceptance criteria:

- <State the criterion that shows this phase met its goal.>

## Phase 1: Scope and design

Parent issue: TBD
Branch: `feature/p1-scope-design`
Status: planned

Goal: state the question, the users, the scope, and the out-of-scope items. Record the first decisions as ADRs.

- [ ] P1.1 Write the problem statement and scope in `notes/`
- [ ] P1.2 Record the first design decisions as ADRs in `notes/adr/`

Acceptance criteria:

- <State the criterion that shows this phase met its goal.>

## Phase 2: Contracts and prototype

Parent issue: TBD
Branch: `feature/p2-contracts-prototype`
Status: planned

Goal: define the interfaces and data contracts, then build the smallest prototype that exercises them.

- [ ] P2.1 Define the interface and data contracts in `docs/`
- [ ] P2.2 Build a prototype that exercises the contracts on synthetic data

Acceptance criteria:

- <State the criterion that shows this phase met its goal.>

## Phase 3: Build

Parent issue: TBD
Branch: `feature/p3-build`
Status: planned

Goal: implement the agreed scope, one task at a time, on the phase branch.

- [ ] P3.1 Implement the first build task

Acceptance criteria:

- <State the criterion that shows this phase met its goal.>

## Phase 4: Verify

Parent issue: TBD
Branch: `feature/p4-verify`
Status: planned

Goal: show that the build meets its acceptance criteria, and record the evidence in the changelog.

- [ ] P4.1 Run the acceptance checks and record the evidence

Acceptance criteria:

- <State the criterion that shows this phase met its goal.>

## Phase 5: Release and deploy

Parent issue: TBD
Branch: `feature/p5-release`
Status: planned

Goal: tag a release, publish the docs, and verify the published artifacts from a clean checkout.

- [ ] P5.1 Write the release notes and the known-gaps list
- [ ] P5.2 Tag, publish, and verify from a clean checkout (maintainer approval required)

Acceptance criteria:

- <State the criterion that shows this phase met its goal.>

## Phase 6: Operate and maintain

Parent issue: TBD
Branch: `feature/p6-operate`
Status: planned

Goal: keep the project current: triage incoming ideas, review on a cadence, and close out cleanly.

- [ ] P6.1 Set a review cadence and record the first review

Acceptance criteria:

- <State the criterion that shows this phase met its goal.>

## Detailed Next Steps Notes

1. Create the GitHub repository and the Phase 0 parent issue. This needs maintainer approval.
2. Activate Phase 0 by setting its parent issue number and its status to `active`, here and in the tracker map.
