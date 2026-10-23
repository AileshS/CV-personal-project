# Basketball Play Lab

A fresh, independent basketball video project. It proposes editable whole
possessions, analyzes selected possessions for made/missed shots, and suggests
the closest visible opposing player at release using approximate image distance.
Foul and turnover labels are manual. Predictions remain review-required.

The website (`playlab web`, then open http://localhost:8600) is the shot dashboard:
pick a video (or add one from YouTube), set a start time and length, press **Analyze**,
and keep working while it runs in the background. Each analyzed clip shows the number
of shots made and missed, a frame-scrubbable timeline with a marker per shot, and every
shot's time in hh:mm:ss.

The earlier Streamlit page (`app.py`) and the labeling tools (`labeling_app.py`,
including **Tracking video**, see [the tracking-video guide](docs/TRACKED_VIDEO.md))
are still available. Made/missed is checked only on a small set
of hand-labeled shots so far; treat results as review-required.

For the review-batch workspace and safe local integration steps, read
[docs/REVIEW_WORKSPACE.md](docs/REVIEW_WORKSPACE.md). The preceding local results
are in [LOCAL_REPORT.md](LOCAL_REPORT.md). Code fetches do not transfer local videos,
SQLite data, weights or the unreviewed frame archive.

## Start locally

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --locked --extra vision
uv run --extra vision playlab web
```

Then open http://localhost:8600.

The tracking screen starts its worker when you select Analyze. For the advanced
workspaces, a worker can also be started manually in a second terminal:

```sh
uv run --extra vision playlab worker
```

For CPU development without neural inference, use `uv sync --locked` and omit
`--extra vision` consistently. To restore the pretrained runtime:

```sh
uv sync --locked --extra vision
uv run --extra vision playlab worker
```

Without vision dependencies, analysis can show diagnostic color candidates but
does not claim those are trained basketball detections or classify their shots.
Use `uv run --extra vision` consistently after installing that extra, because a
plain `uv run` can remove optional dependencies when syncing the environment.

Choose release assistance explicitly in the viewer or CLI:

```sh
uv run --extra vision playlab analyze POSSESSION_ID --mode automatic --queue
uv run --extra vision playlab analyze POSSESSION_ID --mode assisted --queue
```

Automatic mode ignores human shot anchors entirely. Assisted mode uses saved
release anchors. Their cached predictions and evaluation reports stay separate.

Full sources are imported by local path or YouTube URL. The viewer queues long
operations; it does not analyze the whole game on a widget rerun. Games,
annotations, results and job checkpoints persist in `data/playlab.sqlite`.

Choose **Review batches** to navigate a versioned review manifest, inspect
unreviewed suggestions and save human boundary corrections before opening frame
labels. Thumbnail-only previews support browsing; labeling requires matching local
source assets. Loading a batch never imports a game or accepts its suggestions.

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
uv run --extra vision playlab export artifacts/datasets/v1
uv run --extra vision playlab train-detector artifacts/datasets/v1 artifacts/runs/detector-1 --device 0
uv run --extra vision playlab train-classifier artifacts/datasets/v1 artifacts/runs/classifier-1
```

Detector training is capped at two hours per candidate and three candidates per
dataset in sibling run directories. Shot classification uses train-only
imputation/scaling and validation model selection. The simpler geometric baseline
is retained when validation does not favor the learned classifier.

The Colab notebook under `notebooks/` prepares SHA-pinned code/dataset packages,
saves outputs to Google Drive, and gates longer training on checkpoint recovery.
No actual GPU training or real-game accuracy evaluation has been completed.

See [annotation and batch review](docs/ANNOTATION.md) and
[training, checkpoint recovery and evaluation](docs/TRAINING.md) for the complete
workflow. A real CPU checkpoint-resume smoke on artificial images has completed;
it is software evidence only. See [cloud continuation results](CLOUD_REPORT.md)
for measured checks, artifact hashes and remaining blockers.

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
