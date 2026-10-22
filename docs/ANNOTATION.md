# Review before training

The possession baseline proposes editable activity/control spans. Long spans,
camera cuts and fixed review windows are not verified whole possessions. Review
the start/end of the offensive possession; retain offensive rebounds. Fouls and
turnovers are manual labels. Keep unknown outcomes and unresolved players explicit.

## Import and partition

Run the viewer and worker in separate terminals from the project root:

```sh
uv sync --locked
uv run streamlit run labeling_app.py
uv run playlab worker
```

Local video files must exist on the machine running the worker. For a source on
this machine, or a supported YouTube URL:

```sh
uv run playlab import /absolute/path/game.mp4 --queue
uv run playlab import 'https://www.youtube.com/watch?v=B-c3Ybc8HYE' --queue
uv run playlab status
```

Copy the imported game ID from status. Assign complete games to train/validation/
test before sampling. Keep a separate locked game untouched during selection.

```sh
uv run playlab split-game GAME_ID train
uv run playlab index GAME_ID --queue
```

If only one game is available, first declare contiguous blocks with `split-blocks`
or Game setup. Each internal boundary has a ten-second exclusion guard. Keep shot
context, possessions, crops and replays within their partition. Replays and repeated
camera views need human grouping; the software cannot identify every replay.

## Prepare model-assisted batches

Install the optional runtime once; keep using `--extra vision` in that environment:

```sh
uv sync --locked --extra vision
uv run --extra vision playlab prepare-label-batch artifacts/review/batch-01 \
  --frames 300 --possessions 50 --split train --split val --model-assisted
```

The command requires enough split-safe proposals and refuses to overwrite an
existing batch. `--weights /path/ball-rim.pt` supplies a separately trained detector.
The default COCO baseline has no rim class. It can miss balls; missing suggestions
are **not** negative labels. All suggestions are `reviewed: false` and are written
only to the batch. Human annotations in SQLite are unchanged.
Keep `--extra vision` on subsequent viewer, worker, analysis and export commands in
that environment; for example `uv run --extra vision streamlit run labeling_app.py` and
`uv run --extra vision playlab worker`. A plain `uv run` can remove vision and make
previous neural analysis features stale.

Open the generated `index.html` locally to inspect native-source-time samples and
suggested boxes. `manifest.json` contains each image's timestamp, partition, source
identity, hash and normalized proposed coordinates. Sampling is deterministic over
time; it does not establish camera, game or shot-type diversity.

For a short source with an unsplit activity span, explicitly request review windows:

```sh
uv run --extra vision playlab prepare-label-batch artifacts/review/practice-01 \
  --game GAME_ID --frames 60 --possessions 4 --split train --model-assisted \
  --allow-review-windows --window-seconds 30
```

These are boundary-review aids, not four claimed possessions. Where the manifest
says `requires_parent_split_edit`, edit/split the source possession before recording
labels. Batch preparation never changes possession boundaries for you.

## Human review in the viewer

Use the **Review batches** workspace to load the batch manifest, navigate source
timestamps, keep review notes and explicitly link a frame to a corrected whole
possession before opening **Frame labels**. See [the review workspace
guide](REVIEW_WORKSPACE.md) for local data preservation, preview limitations and
restart behavior. Multiple possessions may occur within a sampling window; do not
turn the window into a verified possession automatically.

1. Correct whole-possession boundaries and confirm the source partition.
2. In Game setup, check the basket view and both team uniform colors. Revisit them
   after a camera change. A solo practice clip cannot validate defender behavior.
3. For each sampled elapsed timestamp in Frame labels, inspect the full frame.
   Correct/add every visible ball and rim, including objects the model missed.
   Suggested coordinates are reference material; there is no bulk acceptance of gold.
4. Mark the complete frame reviewed only after all its boxes are corrected. Mark a
   negative frame only when you checked that no ball/rim needs a box. Keep uncertain
   or incomplete frames unreviewed; never turn an empty model output into gold.
5. Save shot release, made/missed/unknown, shooter and closest visible opponent
   separately. Track IDs belong to a particular tracking run, not jersey numbers.
   Compare IDs only against that same frozen run; cross-model defender accuracy
   needs independently reviewed boxes/points. Do not infer court distance.
6. Review requested actions in Jobs and export, or `uv run --extra vision playlab attention`.

The batch is a review aid; export includes only manually reviewed annotations:

```sh
uv run --extra vision playlab export artifacts/datasets/reviewed-v1
```

Export rejects partial frames, split leakage, guarded contexts and duplicate decoded
images across splits. Stale prediction features are stripped; reanalyze matched
possessions before training a result classifier. Never edit an exported dataset in
place; create a new version after label corrections.
