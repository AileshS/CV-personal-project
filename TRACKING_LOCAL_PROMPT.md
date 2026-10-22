# Copy into LOCAL Codex

```text
Continue the fresh Basketball Play Lab project in my existing local checkout.
Preserve all local source changes, commits, videos, weights and SQLite data. Leave
ArcLab and every older project untouched. No paid resources; deadline October23.
Bulk labeling/fine-tuning remain deferred. Prioritize useful ten-second tracking.

Cloud branch: cloud/ball-rim-video-product
Exact tested implementation commit:
b8b0059e6aa610875d67ef7169c3986b2ba8404b
Base was local/basketball-pilot-20261020 at
c1d5e027a73f785561491f54a9a8e434b170b6a1.
The branch also contains a subsequent documentation-only handoff commit.

Read AGENTS.md, PRODUCT_HANDOFF.md, TRACKING_VIDEO_REPORT.md and
docs/TRACKED_VIDEO.md. Keep planner -> implementation -> independent review ->
repair, at most three repair cycles. Do not claim model scores are confidence
intervals or basketball accuracy. No locked-test tuning.

Before integration, inspect git status and stop only this project's old viewer
and worker gracefully. Preserve uncommitted work and take a consistent SQLite
backup using sqlite3.Connection.backup on the actual PLAYLAB_DATA Store (or data/).
Check backup integrity. Do not copy a live SQLite file blindly, reset --hard,
clean the checkout, overwrite my database, or remove a live worker lock.

Inspect then integrate safely; don't discard divergent local commits:
git status --short --branch
git fetch origin cloud/ball-rim-video-product
git show --stat b8b0059e6aa610875d67ef7169c3986b2ba8404b
git diff HEAD b8b0059e6aa610875d67ef7169c3986b2ba8404b
Choose a safe merge/cherry-pick/new local branch after preserving existing work.

Install and launch from THIS project's root using Python3.12:
uv sync --locked --extra vision
uv run --extra vision streamlit run app.py

No second terminal is needed for the default screen: Analyze starts a same-runtime
worker automatically. Restart any old pre-change worker first; do not leave it
running outdated code. Keep --extra vision on subsequent uv commands in this
environment so optional dependencies are retained.

The default screen is ONE flow: select/upload/open a video, set its start time,
Analyze, then play/download the ten-second scored ball/rim video. Advanced tools
keeps previous workspaces out of the way. No possession cutting, frame labeling,
basket setup, training or made/missed classifier is required. The first analysis
downloads a pinned6.25MB basketball/hoop model over HTTPS with SHA verification.
Default model path: artifacts/models/basketball-ball-rim.pt
Model SHA256:72c1e7c64cc6054db3dc5429385630f81c2136f5935fed546e6887bfe158d306.
Model source and licensing limitations are documented in docs/TRACKED_VIDEO.md.

Human acceptance on MY footage: choose my existing full game from local data,
start at an allowed non-test source timestamp (e.g.750s/12:30), select Analyze,
and inspect playable ball/rim boxes with percentages. Existing imported excerpts
keep their original-source offsets. Do not assume the cloud's source exists here.
Do not overwrite my saved game or attach practice labels to it. Known locked-test
intervals stay protected, including preview. If this model misses objects on my
game, retain its raw samples/report and describe that behavior honestly; don't
replace missing detections with invented confident boxes or request bulk labels
as a prerequisite to using the screen.

Actual cloud verification:249 product tests passed,1 existing synthetic test
deselected; training/evaluation/benchmark suites were outside this product scope.
Ruff and diff checks passed. Independent reviewer passed69focused checks and
inspected the actual browser result. Chromium opened a real practice source,
clicked Analyze, automatically started the worker, played/searched and downloaded
the output in46.71s CPU. MP4:H264960x540300frames29.97fps10.01s plus AAC10s.
Model emitted rim boxes297/300frames and ball boxes49/300frames. These are output
counts,NOTrecall/accuracy. Visual check confirms simultaneous readable Ball/Rim
boxes and current detector scores. Missed detections remain gaps. No made/missed,
independent game accuracy, GPU or uninterrupted cloud training claim.

Cloud-only deliverables (code fetch does NOT download them):
artifacts/product-demo/ball-rim-video-v1/basketball-tracked-10s.mp4
Video SHA256:3d5ed16c4902d2e16e6fa6064226733e1f20d4ae93d35636dc72ab86a1a3828a.
artifacts/product-demo/ball-rim-video-v1/tracking-video-demo.zip
ZIP SHA256:15805433845bcde67db240042d21135d6cc4630c6427041b1926bf254d8184b0.
The ZIP has video, screenshots, framewise samples and reports; no model weights,
human annotations, source full game or SQLite database. Artifact-index.json lists
file identities. Never restore the isolated cloud demo Store over my local Store.

Cloud demo source was a public solo-practice clip from the model author's repo,
not my original game and not an independent accuracy benchmark. My original
full game/50s demo was unavailable remotely, and YouTube remained proxy403.
No training, fine-tuning, bulk labeling or human annotation writes were performed.
No dependency declaration/lockfile changed. Phone receipt remains unverified.

All task-owned cloud viewer/worker/browser processes were stopped after checks.
Start local processes yourself. The managed worker can outlive the local viewer;
stop it gracefully when appropriate, never remove a live/unverifiable lock. Cloud
install/start instructions were saved as a draft; publication/restoration is
separate and was not tested.
```
