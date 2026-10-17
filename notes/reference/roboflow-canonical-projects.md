# Reference: Roboflow canonical projects and API behaviour

Evidence captured 2026-10-17 from the live REST API and the `roboflow` Python SDK (1.7.1),
against the FRESH Workspace `gregory-paradis-ubc-ca`
(id `w4vCTstXHBODmnHXKH2vY9dhMbk1`). No credential values are recorded here.

## How it was read

- REST: `GET https://api.roboflow.com/{workspace}/{project}?api_key=...` returns the project
  object under `.project`.
- SDK: `roboflow.Roboflow(api_key=...).workspace(ws).project(slug)`.
- Image listing uses `project.search(limit=..., offset=..., fields=["id","name","annotations","split"])`.
  It returns a list of image records; `annotations` is `[]` for an annotation-free image and a
  `{"count": n, "classes": {...}}` object otherwise. This is the authoritative per-image check.

## Project facts

| Project | Type | Class(es) | Annotated | Unannotated | Split |
| --- | --- | --- | --- | --- | --- |
| `a58-sandbox` | instance-segmentation | `a58-sandbox-microsite`, `a58-sandbox-road`, `a58-sandbox` | 18 | 732 | train 18 |
| `a58-microsites` | semantic-segmentation | `microsite-quality` | 549 | 0 | train 385, valid 110, test 54 |
| `a58-features` | instance-segmentation | rock, log, slash, stump, vegetation, not-microsite, ... | 6 | 750 | train 6 |

- `a58-sandbox` is public with license `BY-NC-SA 4.0`; the other two are private.
- The starter batch in `a58-sandbox` is 18 `train` images carrying 97 `a58-sandbox-microsite`
  instances. The operator added these 2026-10-17; before that the project was a clean slate.

## Fields that mean something different from their names

- `project.images` is the number of images that carry annotations, not the image total.
- `project.unannotated` under-reported the unannotated total relative to the image listing
  (750 vs 732). Treat `search` as authoritative and compute the counts yourself.
- A project's `classes` counter includes classes registered historically; it is not a reliable
  annotation count. Use the per-image `annotations.count` instead.

## Upload behaviour used by this project

- `project.save_annotation(image_id=..., annotation_path=..., annotation_labelmap=...,
  is_prediction=True, annotation_overwrite=False)` attaches an annotation to an existing image.
- `annotation_overwrite=False` and `is_prediction=True` are the safe defaults: they refuse to
  replace an existing annotation and mark the result as a model prediction awaiting review.
- The REST equivalent is
  `POST https://api.roboflow.com/dataset/{project}/annotate/{image_id}?api_key=...&name=...`
  with a JSON body `{"annotationFile": "...", "labelmap": {...}}`.

## Uncertainty

- The `unannotated` discrepancy above is unexplained; it may be a cached field. The tooling
  therefore never trusts it, and recomputes counts from the image listing on every run.
