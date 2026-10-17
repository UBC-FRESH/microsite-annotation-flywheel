# ADR 0001: Flywheel design for the first annotation round

Status: accepted
Date: 2026-10-17

## Context

The flywheel must let a non-coder operator run a repeated
sync → train → pre-label → correct loop against Roboflow, without ever risking the human
annotations. The Robotflow Workspace holds three relevant projects: `a58-sandbox`
(instance-segmentation, a starter batch of 18 annotated images plus 732 unannotated),
`a58-microsites` (semantic-segmentation, 549 annotated), and `a58-features` (instance, 6
annotated). The proven model is U-Net ResNet34 with a focal/dice loss and TTA.

## Decisions

1. **Round 1 is self-hosted in `a58-sandbox`.** The loop trains on the project's own hand
   annotations and pre-labels the rest of that project. The operator chose to bootstrap the
   sandbox by hand rather than train on the `a58-microsites` taxonomy. Consequence: the
   configuration separates a *train source* from a *pre-label target*, because later rounds
   may train on one project and pre-label another.

2. **Pre-labels are polygons labelled `a58-sandbox-microsite`,** converted from the model's
   binary mask (one polygon per connected component). `a58-sandbox` is instance-segmentation,
   so a raw mask cannot be uploaded. Consequence: the tooling needs a mask-to-polygon step and
   a component-size floor to drop noise blobs.

3. **Uploads are annotation-safe by construction.** Every pre-label write uses
   `annotation_overwrite=False` and `is_prediction=True`, and the tool queries each target
   image's annotation status immediately before writing, skipping any image that already has
   annotations. Consequence: a human annotation cannot be clobbered even by a bug; the only
   failure mode is skipping an image, which is safe and reported.

4. **The convergence signal is object-level and pixel-level, reported side by side.** After a
   round of human correction, `sync` compares the annotations now in the project against the
   pre-labels that were uploaded, matching objects at IoU ≥ 0.5 and counting added, deleted,
   and edited objects, and computing the fraction of positive pixels changed and the IoU
   between the uploaded and corrected masks. Consequence: the operator watches the trend
   shrink toward stability and stops the loop when it flattens.

5. **Checkpoints, datasets, predictions, and galleries live in the DataLad data repo.** Only
   small JSON state and metrics live in the code repo, so the code stays cloneable and every
   round stays reproducible from a declared input.

## Consequences

- The loop is re-runnable for a different microsite definition or a new project by editing
  `config.yaml` and bumping the round, with no code change.
- The safe-upload rule means the tool can never *destroy* data, only decline to write, so the
  operator's errors are recoverable.
- A dry run trained on 18 images will produce a weak model; that is expected and is a
  pipeline smoke test, not a scientific result.
