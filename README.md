# Basketball Play Lab

A fresh, independent basketball video project. It proposes editable whole
possessions, analyzes selected possessions for made/missed shots, and suggests
the closest visible opposing player at release using approximate image distance.
Foul and turnover labels are manual. Predictions remain review-required.

## Start locally

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --locked
uv run streamlit run app.py
```

Run the processing worker in a second terminal:

```sh
uv run playlab worker
```

Install actual neural detection separately:

```sh
uv sync --locked --extra vision
uv run --extra vision playlab worker
```

Without vision dependencies, analysis can show diagnostic color candidates but
does not claim those are trained basketball detections or classify their shots.
Use `uv run --extra vision` consistently after installing that extra, because a
plain `uv run` can remove optional dependencies when syncing the environment.

Full sources are imported by local path or YouTube URL. The viewer queues long
operations; it does not analyze the whole game on a widget rerun. Games,
annotations, results and job checkpoints persist in `data/playlab.sqlite`.

## Training and testing

```sh
uv run pytest -q
uv run ruff check .
uv run playlab --help
```

Assign games to training, validation and final test before labeling. The
same-game fallback requires declared continuous blocks with ten-second guards.
Mark complete frames as reviewed, including every visible ball/rim, or explicitly
mark a negative frame. Exporting partial labels is rejected.

```sh
uv run playlab export artifacts/datasets/v1
uv run --extra vision playlab train-detector artifacts/datasets/v1 artifacts/runs/detector-1 --device 0
uv run playlab train-classifier artifacts/datasets/v1 artifacts/runs/classifier-1
```

Detector training is capped at two hours per candidate and three candidates per
dataset in sibling run directories. Shot classification uses train-only
imputation/scaling and validation model selection. The simpler geometric baseline
is retained when validation does not favor the learned classifier.

The Colab notebook under `notebooks/` prepares SHA-pinned code/dataset packages,
saves outputs to Google Drive, and gates longer training on checkpoint recovery.
No actual GPU training or real-game accuracy evaluation has been completed.

## Current handoff

Read [HANDOFF.md](HANDOFF.md) before continuing. This is a paused implementation
snapshot, not a completed or accurate basketball analytics product.

Videos, labels, weights, local environments and secrets are excluded from Git.
The cloud session must obtain its own source assets. A local worker stops when
this computer is off. A published Codex Cloud task can continue coding remotely;
GPU training uses a separate resource and free notebook availability varies.

Optional phone notification delivery uses `PLAYLAB_NTFY_URL` and
`PLAYLAB_NTFY_TOKEN` on the machine actually running the worker. Neither a remote
endpoint nor phone receipt has been configured or tested. The viewer always
retains unresolved human-action requests even when delivery is unavailable.

Ultralytics software/model licensing applies to that optional dependency:
[licensing](https://www.ultralytics.com/license). Source footage is not included
or redistributed with this code.
