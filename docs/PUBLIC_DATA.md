# Public labeled detector pilot

Prepared separately from the user's SQLite review data. Nothing is imported into
the local Store and no local human annotation is replaced.

## Source and reproducibility

- Publisher source: [Eagle Eye Basketball v1](https://universe.roboflow.com/eagle-eye/basketball-1zhpe/dataset/1).
- [Pinned mirror](https://github.com/ahmedsilat44/Swish-Vision/tree/b22265d1a62bbfd5cc1c0d3aea63be38950869a4/Basketball-1).
- Publisher license: CC BY 4.0, retained with attribution in the prepared bundle.
- Fixed mirror revision: `b22265d1a62bbfd5cc1c0d3aea63be38950869a4`.
- Dataset hash: `0f4b75e969a850a9a17896fdd4fa87edae713ad50c1a90336205d050fadb81fe`.

```sh
uv run --extra vision python scripts/prepare_public_detector.py \
  artifacts/datasets/public-ball-rim-v1 --train-count 300 --val-count 130
```

Choose a new output name if it already exists. Downloads are cached outside Git
under `artifacts/external/swish-vision`; `--offline` converts an already downloaded
selection. Source Git blob IDs are checked during download; converted files have
an immutable SHA-256 manifest compatible with the existing detector trainer.

## Actual contents

| Source partition | Prepared images | Ball boxes | Rim boxes |
| --- | ---: | ---: | ---: |
| Training | 296 | 311 | 33 |
| Validation | 124 | 131 | 0 |

The deterministic download selected 300 training and all 130 source validation
images. Ten empty-label images were excluded, rather than inventing verified
negatives. `basketball` maps to ball; rim maps to rim. Images with ambiguous
`sports ball` labels are excluded (none occurred in this selection). Original
filenames are kept disjoint across source splits, and exact decoded duplicate
images are removed. No external test images or local locked-test frames were
downloaded, decoded or used.

## Limits and next experiment

These are **publisher-provided labels**, not locally verified human gold. The
bundle says `external_labeled_detection`; it does not invent games, possessions,
shot outcomes or defender identities. Source images were resized to 416×416 and
include varied camera views, close-ups and illustrations. They may transfer
poorly to tiny balls in the user's wide-angle game. Rim label extent also varies.

**There are no rim reference boxes in the publisher's validation set.** This
package can support a first detector pretraining experiment and external ball
validation, but cannot establish rim recall or select reliable rim thresholds.
Another annotated rim-validation source is needed before those claims. Source
game/replay identities are unavailable, so filename/pixel checks do not prove
independence of all visually related examples.

No model has been trained from this package in the local preparation session.
Training remains a separate bounded cloud-GPU experiment; do not claim measured
improvement on the user's game from public labels or software checks. Preserve
the local train/validation/test policy and keep user-reviewed labels separately
for later adaptation and checks. Public detection labels do not replace shot
made/missed labels or full-possession verification.
