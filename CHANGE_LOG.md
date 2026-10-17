# Change Log

Newest entries last. Each milestone gets a dated entry that matches the roadmap status and the GitHub issue comments.

## 2026-10-17

- Created Microsite Annotation Flywheel from the UBC-FRESH dev workflow seed. The agent contract, roadmap, docs skeleton, and validator came from the seed. Every phase is planned, and no parent issues exist yet.
- P0.1 adapted the roadmap to the six flywheel phases (P0 scaffold, P1 sync, P2 train, P3 prelabel and upload, P4 loop bookkeeping and convergence, P5 runbook and acceptance) and added the project-specific rules to `AGENTS.md`.
- P0.2 added credential and data-artifact guardrails to `.gitignore`, recorded the Roboflow evidence in `notes/reference/roboflow-canonical-projects.md`, and recorded the design in `notes/adr/0001-flywheel-design.md`.
- Verified against the live Roboflow API: `a58-sandbox` now holds 18 annotated images (97 `a58-sandbox-microsite` instances) and 732 unannotated images, so round 1 self-hosts in that project.
