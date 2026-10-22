# Ball/rim tracking video: product result

Starting branch: `local/basketball-pilot-20261020`.
Verified starting commit: `c1d5e027a73f785561491f54a9a8e434b170b6a1`.
Implementation branch: `cloud/ball-rim-video-product`.

## Delivered flow

The default screen now has video selection/upload, a start time and **Analyze**.
It produces a ten-second playable/downloadable video with scored ball/rim boxes.
The worker starts automatically. Detailed review, labels, possession cutting,
basket hints and shot-outcome classification are not prerequisites.

The compatible pretrained two-class model supplies **Basketball** and **Basketball
Hoop** detections. ByteTrack associates current detections between frames. Every
output frame is inferred, capped at 30 fps/1280px width. H.264/AAC output retains
bounded source audio and original elapsed timestamps. Missing detections remain
gaps; no prior box/score is presented as a new model observation.

Detailed workspaces remain behind **Advanced tools**. Existing media, human data
and advanced workflows are preserved. New request/result kinds and output files
are separate from annotations, training exports and the older Quick analysis.

## Actual browser acceptance

Chromium/Playwright opened the real source through the local-path form, selected
the default interval, clicked Analyze, waited for the automatically started worker,
played/searched the MP4 in Chromium and downloaded it. No browser errors occurred.

| Check | Observed result |
| --- | --- |
| Selected source interval | 0–10 seconds of the public solo-practice video |
| Complete browser-to-download time | 46.71 seconds, CPU |
| Output video | H.264, 960×540, 300 frames, 29.9700 fps, 10.01 s |
| Output audio | AAC, 10.00 s |
| Browser player | `readyState=4`, no media error; playback/seek/download passed |
| Frames with model rim boxes | 297/300 |
| Frames with model ball boxes | 49/300 |
| Visual check | Simultaneous readable Ball/Rim boxes, track IDs and detector percentages |
| Human annotations written | None; only game metadata and tracking request/result records |

These counts describe detector output presence, **not recall, accuracy, true ball
visibility, or a basketball outcome metric**. Tracking can break at occlusions;
the video deliberately exposes missed detections. Scores are raw detector scores,
not statistical confidence intervals or calibrated probabilities. No made/missed
recognition is claimed.

Final retained artifacts (outside Git):

```text
artifacts/product-demo/ball-rim-video-v1/basketball-tracked-10s.mp4
artifacts/product-demo/ball-rim-video-v1/browser-report.json
artifacts/product-demo/ball-rim-video-v1/choose-clip.png
artifacts/product-demo/ball-rim-video-v1/tracked-video-browser.png
artifacts/product-demo/ball-rim-video-v1/tracked-frame-1.4s.png
artifacts/product-demo/ball-rim-video-v1/tracking-video-demo.zip
artifacts/product-demo/ball-rim-video-v1/artifact-index.json
```

MP4 SHA-256:
`3d5ed16c4902d2e16e6fa6064226733e1f20d4ae93d35636dc72ab86a1a3828a`.
Framewise samples SHA-256:
`efaebbec097aec37582dd71443689260027733011f6043008231ba004db6f812`.

The isolated demo Store is under the same artifact folder's `data/`; its result
points to `tracking_video/<request-id>/<attempt-id>/tracked.mp4`, `samples.json`
and `report.json`. Do not copy this demo database over the user's local Store.

## Verification and repairs

- Reproduced the baseline product suite: **189 passed, 1 deselected**.
- Final product suite: **249 passed, 1 deselected in 65.73 s**; Ruff and whitespace
  checks passed. Exact command:

  ```sh
  uv run --locked pytest -q --ignore=tests/test_training.py --ignore=tests/test_evaluation.py --ignore=tests/test_benchmark.py -k 'not missing_optional_weights_allow_honest_fallback_export_and_freeze'
  ```

  Training/evaluation/benchmark suites and the named synthetic freeze regression
  were outside this product milestone; no new test was disabled to obtain success.
- Independent review: **69 focused checks passed**, followed by independent MP4
  hash/codec/frame/visual/provenance inspection. No required repairs remain.
- One review repair cycle covered protected previews, model installation races,
  delayed/stopped worker handling and one-click interrupted-job recovery. Renderer
  cancellation tests also found and fixed blocked FFmpeg shutdown before acceptance;
  only owned child processes are terminated, and incomplete files are not published.
- Existing UI tests explicitly enter Advanced tools, retaining their meaningful
  annotation/review/Quick analysis checks. New tests cover the simpler default flow.
- Actual pinned-model HTTPS download/checksum/cache reuse passed. The complete
  saved cloud install script also executed successfully.
- Browser request's source/model/code hashes still matched the completed output
  after the final tests. Source and dependency declarations/lockfile were preserved.

## Model, footage and limits

Model: Avi Shah's public basketball weights, revision
`4eb004437919f3667dce3d0a95575a510d763485`, `best.pt` from
`avishah3/AI-Basketball-Shot-Detection-Tracker`.
SHA-256: `72c1e7c64cc6054db3dc5429385630f81c2136f5935fed546e6887bfe158d306`.
The downloader caches `artifacts/models/basketball-ball-rim.pt`; no weights or
third-party implementation are committed. The upstream supplies no model license
file, so verified open model licensing is not claimed. See the attribution and
score/class definitions in [docs/TRACKED_VIDEO.md](docs/TRACKED_VIDEO.md).

Demonstration source: that repository's `video_test_5.mp4` at the same fixed
revision, 112.712604 seconds, 960×540, audio. Full source SHA-256:
`33c8337dca309217b8ec3608ed5baa4a204a13e1a9c0f7edbfa8612fb14ae3e5`.
It is practice footage from the model author's repository, not an independent
accuracy benchmark. No training or test-set tuning was performed.

The user's original full game and uploaded 50-second demo were not available on
this cloud machine. A fresh attempt to retrieve the original YouTube source still
returned proxy CONNECT 403. The new product must therefore be checked on that
local footage after integration; transfer accuracy is unmeasured.

Bulk labeling, fine-tuning, locked evaluation, GPU/Colab recovery and phone receipt
remain deferred/unverified. No paid resource was created. The product itself
requires neither labels nor GPU to render tracking video.

## Startup and preservation

```sh
uv sync --locked --extra vision
uv run --extra vision streamlit run app.py
```

Analyze starts the worker; retain the extra consistently. First use downloads the
six-MB pinned detector. Read `docs/TRACKED_VIDEO.md`. Updated cloud install/start
instructions were saved in the environment draft; review/save/publish is separate
from current-instance/browser validation. Fresh-task restoration was not tested.

A consistent pre-change cloud SQLite backup passed integrity checking at
`/workspace/cloud-preservation/before-ball-rim-video-c1d5e02.sqlite`. No original
local database was copied or changed. Preserve local files/commits/SQLite when
integrating. **All task-owned viewer, worker and browser processes are stopped
after validation; no training process was started.** Use the exact commit in the
final handoff; do not reset the user's branch or touch older projects.
