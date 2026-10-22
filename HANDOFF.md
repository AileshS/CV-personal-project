# Cloud continuation handoff — October 20, 2026

Continued independently from `main` at
`5c39f6c249a648d1f1480dfa5ffc88099667a368` on
`cloud/basketball-pilot-20261020`. Deadline remains October 23; free resources only.

Read [CLOUD_REPORT.md](CLOUD_REPORT.md) for measured results, paths and hashes, and
[docs/ANNOTATION.md](docs/ANNOTATION.md) / [docs/TRAINING.md](docs/TRAINING.md) for
tested workflows. The final chat report and `LOCAL_HANDOFF.md` identify the exact
commit to fetch. Preserve existing local work and leave older projects untouched.

Current code checks: **103 tests passed**, Ruff and whitespace checks passed.
Separate planning/implementation/independent review completed with one repair cycle.
Automatic analysis now ignores unreviewed/human shot anchors; assisted analysis
uses reviewed anchors. Freshness, dataset/prediction provenance, single-use evaluation
and manifest-only training/benchmark inputs are enforced. Malformed notification
configuration fails soft. Neural runtime commands must consistently retain vision.

A real CPU detector checkpoint smoke on artificial data paused/resumed optimizer
and RNG state and completed three epochs. The full CPU dataset/classifier/evaluation
pipeline and a detector benchmark script also executed on artificial data. None
establishes basketball accuracy, GPU recovery or Colab/Drive readiness.

Remote footage obtained: a separate **112.712604-second, 960×540 solo-practice clip**
from the public source credited in `artifacts/source-provenance.json`. It is training-
partition development material only. The original full-game YouTube URL remains
blocked by proxy CONNECT 403; required YouTube domain additions are saved in the
environment draft, not proven applied/published.

Prepared `artifacts/review/public-practice-v2`: **60 native frames, four boundary-
review windows, 21 proposed ball boxes on 19 frames**, all unreviewed. These windows
are not claimed possessions. No rim suggestions exist in the COCO class set. The
clip has no defender evidence and cannot replace the requested multi-possession game
batch. No real reviewed train/validation dataset exists. Do not train from or bulk
accept these suggestions before the user checks every frame/label.

Next actions:

1. Apply the saved YouTube network settings and retry the original full-game import,
   or continue on the user's already downloaded local full game after preserving
   its existing Store. Declare game splits/guarded blocks before sampling 40–60
   whole-possession review items and roughly 300 diverse frames.
2. Human-review possession boundaries, basket/team setup, all ball/rim boxes and shot
   outcomes. Correct missed objects; empty suggestions are not negative frames.
3. Export a new immutable reviewed dataset; train/select on train/validation only.
   Free GPU/Colab and actual Drive recovery still require testing.
4. Confirm one Codex phone notification receipt. The user said they would enable
   notifications but has not confirmed receipt. Optional worker ntfy URL/token are
   absent; no notification delivery is claimed. Pending requests persist in SQLite.

Cloud viewer and worker were restarted and validated, then deliberately stopped.
**No task-owned viewer, worker or training process remains running.** Live processes
must be started again in another task. The cloud dataset, native review images,
videos and model artifacts are outside Git; local code fetch does not retrieve them.

---

# Historical local -> cloud snapshot (superseded by the current status above)

Paused at the user's request on October 20, 2026. Requested delivery: October 23.
Repository destination: https://github.com/AileshS/CV-personal-project
Local checkout folder: `Basketball-Play-Lab` inside the user's CV1 workspace.

## Accepted scope

This is a NEW project; do not build on or copy ArcLab. Build editable whole
possessions (retain offensive rebounds), shot made/missed analysis and closest
visible opposing player at release. Approximate image-space distance is accepted;
court distance comes later. Manual short setup and human label review are
accepted. Fouls/turnovers remain manual for this sprint. Flag uncertainty.

Use free resources first. Do not spend money. Use a planner -> implementer ->
independent reviewer -> repair loop, bounded to three repairs per milestone.
Do not tune against locked test data. Communicate with smart brevity.

## Implemented

- Independent Python package, dependency lock, Streamlit review app and CLI.
- SQLite job queue with deduplication, explicit recovery, cancellation and saved
  human-action requests; processing runs separately from the viewer.
- Timestamp-based streaming FFmpeg access, audio-preserving selected clips,
  URL/local import and source provenance; no 600-frame analysis limit.
- Editable split/merge possessions, frame/shot/manual event labels, team/rim
  setup, whole-game splits and guarded same-game partitions.
- Separate pretrained person tracking and optional fine-tuned ball/rim adapter;
  resumable selected-possession chunks, evidence images and approximate defenders.
- Reviewed-label dataset export, leakage checks, trained logistic shot classifier
  versus geometry baseline, bounded YOLO training and checkpoint/RNG provenance.
- Frozen final-test evaluation and metrics helpers. Colab notebook included.
- Software tests, repository agent instructions and CPU CI workflow.

## Evidence so far

The supplied source was successfully queried:
https://www.youtube.com/watch?v=B-c3Ybc8HYE
Title: The Oakridge School vs St. Mark's School of Texas Mens Varsity Basketball.
Published January 30, 2026; duration approximately 7,171 seconds.

A 15-second 720p sample from original source time 20:00–20:15 was downloaded,
decoded with audio, and visually inspected. Its camera view shows players, ball,
basket and a scoreboard inset. It is local-only under `data/sources/`.

A fresh YOLO11n pretrained inference smoke on the sample found one sports-ball
candidate and 21 people (including spectators). This establishes execution only,
not basketball accuracy. A fine-tuned detector has NOT been trained. No reviewed
real-game gold dataset exists. No real-game accuracy result is available.

The final local combined suite passed **65 tests**; Ruff and Python compilation
also passed. Verify the snapshot afresh; use the sandbox/default local runtime, or on Linux
`uv sync --locked && uv run pytest -q`. An elevated Windows test run ran into
temporary-directory ownership conflicts and is not a reliable code failure signal.

## Cloud is NOT yet configured

No Codex Cloud environment was created/published and no cloud coding task was
started by this session. No phone endpoint or push receipt was verified. The user
will open a cloud session after this push.

Remote code cannot access this computer's `data/`, `.venv` or `artifacts/`.
Retrieve the video remotely or upload a portable dataset package when reviewed
labels become available. Do not claim local assets are already in the cloud.

## Next cloud work, in priority order

1. Install the locked environment, run lint/tests, inspect all current files and
   repair genuine failures. Confirm incomplete scripts are usable before relying
   on them. Do not accidentally turn synthetic tests into accuracy claims.
2. Finish README/annotation/training instructions and inspect the Colab notebook
   by executing a small smoke workflow when access permits. The existing
   checkpoint unit test uses a stub, not an actual GPU resume proof.
3. Retrieve representative real footage. Verify native ball size and rim/player
   visibility. Additional game recordings are NOT yet verified; timebox discovery.
4. Prepare model-assisted review batches for 40–60 possessions and roughly300
   diverse frames; keep all prelabels unreviewed until a human verifies them.
   Split by game first; otherwise use declared guarded same-game blocks.
5. Audit model and evaluation correctness before training: reject features from
   stale analysis, keep every crop/replay within its partition, count unknowns
   and missed events in coverage, and evaluate automatic and human-anchored
   releases separately. Current ID-based defender scoring only compares IDs
   within the same frozen tracking run; prefer human boxes/points for comparisons
   between different models/runs.
6. Finish fair detector-before/after and dataset-prediction export commands if
   those scripts are missing/incomplete. Standard pretrained YOLO has no rim
   class: evaluate ball baseline fairly and report rim localization separately.
7. Once reviewed data exists, train on training data, select on validation data,
   freeze weights/configuration and evaluate the locked test once. Stop for
   human labels/access or free-compute limits instead of inventing progress.

## Known limits to inspect

- Possession segmentation is an experimental activity/control baseline; scene
  cuts and fixed windows do not constitute reliable basketball possessions.
- Person detections can include officials/bench/spectators; jersey colors and
  proximity alone cannot guarantee the correct defender. Retain explicit unknowns.
- Automatic shot release is an estimate. Report assisted results separately.
- Frame sampling timestamps are approximate to the selected sampling interval;
  source clips use elapsed FFmpeg seeking rather than nominal source-frame FPS.
- The complete source finished downloading before the pause took effect and was
  imported locally: `data/sources/B-c3Ybc8HYE.mp4`, 1920x1080, 30fps, audio,
  7,170.62 seconds. The local workspace contains its saved game record.
  It is excluded from Git and NOT available to the cloud checkout. Full-game
  indexing and basketball accuracy remain unverified. No import worker is left running.
- Notification failures must never stop the worker. Malformed topic URLs were
  changed to fail soft and preserve the human-action record; retest this.

## Phone and return handoff contract

Ask the user to enable ChatGPT/Codex phone notifications for this cloud task and
verify one delivery. Separately configure/test the optional worker notification
channel if needed. Do not promise an untested notification path. Free Colab is
interruptible, so checkpoints are mandatory; no keep-alive bypasses.

When cloud work stops, push all source changes to this repository and give the
user a copy-paste prompt for their LOCAL session. That prompt must include:

- Exact branch and commit hash to fetch.
- Changes made, checks actually run and their results.
- Location/hash of model/dataset/result bundles; which are NOT in Git.
- Exact local install/run/import commands and any required dependency changes.
- Outstanding label batches, camera/team setup and phone steps.
- Whether cloud workers remain running, and how to stop them.
- Verified results versus unverified claims, including known failure cases.

Do not modify older projects when the user returns locally. Preserve uncommitted
local changes; fetch and inspect before applying cloud changes.
