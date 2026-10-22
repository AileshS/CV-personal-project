# A 10-second video with ball and rim boxes

From the project root, with Python 3.12 and uv:

```sh
uv sync --locked --extra vision
uv run --extra vision streamlit run labeling_app.py
```

Keep `--extra vision` on subsequent uv commands in this environment. A plain
`uv run` can remove the model runtime. No GPU or training is required.

## Use the screen

1. Choose an existing video, upload a video, or open its path on this machine.
2. Set the start time. The selected interval is ten seconds, shortened only at the
   end of the source. The optional preview shows its starting frame.
3. Select **Analyze**. The worker starts automatically; progress updates while it
   prepares the result. The first run downloads the basketball model, about 6 MB.
4. Play **Your tracked clip**, or select **Download tracked video**.

The output is an H.264 MP4 with source audio when present, colored ball and hoop
boxes, object tracking and detector scores. Original source timestamps remain on
the video, including an existing excerpt's saved source offset. Processing runs on
every output frame, capped at 30 fps and a 1280-pixel input width. No possessions,
human labels or training examples are created.

Open it from the sidebar: turn on **Advanced**, then choose the **Tracking video**
workspace. `app.py` itself shows only Quick analysis.

## Detector and scores

The product uses a two-class pretrained basketball detector, rather than COCO's
ball-only class set. Weights come from Avi Shah's public
[basketball detector repository](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker),
revision `4eb004437919f3667dce3d0a95575a510d763485`, file `best.pt`.

SHA-256: `72c1e7c64cc6054db3dc5429385630f81c2136f5935fed546e6887bfe158d306`.

The default local cache is `artifacts/models/basketball-ball-rim.pt`. Download uses
HTTPS and checks the pinned content hash before publishing the asset. It preserves
a different existing model file. `PLAYLAB_MODEL_DIR` can relocate the cache.
Only the pretrained weights are used; no implementation is copied from that
project. Its source does not supply a model license file, so this is not described
as a model with verified open licensing. Ultralytics' own licensing also applies.

Labels map **Basketball** to ball and **Basketball Hoop** to rim/hoop. The latter
can include the rim and hanging net; it is not a measured rim centerline. A number
such as **Ball 81%** is the raw detector score, not a statistical confidence
interval, a calibrated probability, or an accuracy estimate. Tracking IDs may
change after occlusion or camera cuts. Frames without a current detection do not
show an invented box or a carried-forward score.

Detection results do not establish made/missed shots, reliable full-game tracking,
or defender accuracy. Those tasks remain separate. Fine-tuning and bulk labeling
are deferred while the usable detection-video flow is prioritized.

## Files, failures and preservation

Requests/results use separate `tracking_clip` and `tracking_result` records in the
existing Store. The renderer writes an annotated MP4, framewise raw boxes/scores
and a report under `data/tracking_video/`. Files appear as a completed result only
after rendering succeeds. Cancellation/failure cannot publish a partial video.
Retry restarts the short clip; existing user media and human annotations remain
untouched. Source, model, code or output changes invalidate an old result.

Known locked-test intervals are rejected before decoding or source-content reads,
including preview. A declared test interval is not a demo source.

If a worker cannot start or stops, use the screen's explicit Retry action. It does
not remove the lock of a live or unverifiable worker. Startup logs are retained in
the Store's data directory. Do not delete worker locks by hand. A managed worker
can outlive the viewer process on the same machine; it stops when that machine
stops. This feature does not promise uninterrupted cloud training.

For a reproducible model download without opening the viewer:

```sh
uv run --extra vision python -c "from playlab.model_assets import ensure_ball_rim_model; print(ensure_ball_rim_model())"
```

Videos, weights, SQLite data and generated results remain outside Git. Fetching
code does not retrieve the user's local full game or any cloud demonstration clip.
