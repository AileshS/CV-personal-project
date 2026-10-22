# Bounded training and honest evaluation

Use free resources only. The project does not provision paid compute. No GPU is
present in this cloud machine. Real reviewed train/validation labels are still
required before basketball training; manufactured smoke data proves execution only.

## CPU checks and a real checkpoint smoke

```sh
uv sync --locked
uv run pytest -q
uv run ruff check .
uv sync --locked --extra vision
uv run --extra vision python scripts/smoke_training.py artifacts/smoke/training-new
```

The smoke uses four artificial training images, two validation images and three
96-pixel CPU epochs. It pauses after the first checkpoint, verifies optimizer state,
resumes the same candidate and checks RNG/epoch progress. Its separate fabricated
shot table exercises sklearn. Outputs are explicitly synthetic and stay outside
Git. A successful CPU smoke does not prove GPU, Colab, Drive persistence or exact
numerical equivalence across interruptions.

## Reviewed data and model selection

```sh
uv run --extra vision playlab export artifacts/datasets/reviewed-v1
uv run --extra vision playlab train-classifier artifacts/datasets/reviewed-v1 artifacts/runs/classifier-1
uv run --extra vision playlab train-detector artifacts/datasets/reviewed-v1 \
  artifacts/runs/ball-rim-1 --device 0 --epochs 40 --max-hours 1 --image-size 960
```

`--device 0` requires an available free GPU. Use `--device cpu` for a bounded CPU
run. The detector uses exact manifest-listed train/validation images, never the
test split. The classifier fits its imputer/scaler on train only and chooses among
regularized classifiers and abstention thresholds on validation; geometry wins ties.
Unreviewed or unknown outcomes are never made/missed training examples.
After installing vision, keep `--extra vision` on viewer, worker, export and training
commands. A plain `uv run` can remove optional dependencies; changing the detector
runtime invalidates analysis features and requires reanalysis. For CPU-only work,
omit the extra consistently throughout that separate environment.

At most three detector candidates per dataset are recorded in sibling run
directories. Each candidate has at most two hours of configured training time;
setup and final validation add overhead. Do not move runs to bypass this budget.
Resume requires the original dataset, code, runtime versions, epoch target, image
size, checkpoint and RNG sidecar. Completed checkpoints are finalized; interrupted
or deliberately paused runs can resume:

```sh
uv run --extra vision playlab train-detector artifacts/datasets/reviewed-v1 \
  artifacts/runs/ball-rim-1 --device 0 --epochs 40 --max-hours 1 --image-size 960 \
  --resume artifacts/runs/ball-rim-1/detector/weights/last.pt
```

The notebook `notebooks/train_colab.ipynb` verifies source/dataset zip SHA-256,
requires a Python 3.12 GPU runtime and a real checkpoint smoke, and saves runs to
Drive. Use the same archive hashes when reconnecting. Its Drive/GPU cells have not
been executed in this cloud session. Free Colab can expire; no keep-alive bypass or
uninterrupted-training guarantee is provided.

## Validation before the held-out test

Run equal-image detector comparisons on validation first:

```sh
uv run --extra vision python scripts/benchmark_detector.py \
  artifacts/datasets/reviewed-v1 artifacts/runs/ball-rim-1/detector/weights/best.pt \
  artifacts/reports/detector-validation.json --split val --device cpu
```

The COCO baseline supports ball, not rim. Reports contain fixed-threshold
precision/recall counts, not mAP. Negative-frame predictions count as false
positives. Do not call synthetic-image counts basketball accuracy.

## Freeze and evaluate once

Finalize code, weights and inference settings using development/validation only.
Make a durable commit before freezing: changing code or the Git revision invalidates
the freeze. Use automatic inference for end-to-end automatic results. It ignores
all human shot anchors; assisted mode is a separate capability and report.

```sh
uv run --extra vision playlab analyze TEST_POSSESSION_ID --mode automatic \
  --weights artifacts/runs/ball-rim-1/detector/weights/best.pt \
  --model-bundle artifacts/runs/classifier-1/classifier.joblib
uv run --extra vision python scripts/predict_dataset.py artifacts/datasets/reviewed-v1 \
  artifacts/reports/test-predictions.json --data data --split test --mode automatic
```

Exported predictions carry dataset/split identity, analysis input fingerprints,
effective settings and every inference artifact hash. Freeze the prediction JSON
and **every path in its `inference_artifacts` map**, including the baseline
`artifacts/models/yolo11n.pt`, custom weights and classifier bundle when used:

```sh
uv run --extra vision playlab freeze artifacts/datasets/reviewed-v1 artifacts/frozen/pilot-1 \
  artifacts/reports/test-predictions.json artifacts/models/yolo11n.pt \
  artifacts/runs/ball-rim-1/detector/weights/best.pt \
  artifacts/runs/classifier-1/classifier.joblib
uv run --extra vision playlab evaluate artifacts/datasets/reviewed-v1 \
  artifacts/reports/test-predictions.json artifacts/reports/test-1 \
  --freeze artifacts/frozen/pilot-1/freeze.json
```

No inference computation needs human test outcomes. Reading source frames for
locked inference is separate from opening the final outcome report. Raw prediction
lists without provenance are rejected. Missing analyses, missed events and abstained
outcomes remain in coverage denominators; false-positive events are reported.
Automatic and release-assisted results are distinguished. ID-based defender counts
are meaningful only against the identical frozen tracking run.

Each freeze has a single-use outcome-test fence. Do not erase it or create fresh
freezes to retune against the same test. Do not tune thresholds after seeing results.
The detector test has its own single-use fence and additionally requires frozen
baseline/custom weights and a JSON settings file passed as `--config` with `--freeze`.

Videos, native review frames, labels, model weights, datasets and generated reports
stay outside Git. The source repository and local data/artifact bundles are separate
handoffs; download/restore required bundles before using their cloud paths locally.
