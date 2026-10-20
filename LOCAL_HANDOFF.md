# Ready-to-copy prompt for LOCAL Codex

Copy the text below into the LOCAL session. It pins the tested implementation
commit; the remote branch also contains this documentation-only handoff commit.

```text
Continue Basketball Play Lab in my existing CV1/Basketball-Play-Lab checkout.
Deadline: October 23, 2026. Free resources only. Keep this independent of ArcLab.
Preserve all existing local source changes, untracked files, videos, weights and
SQLite data. Leave every older project untouched. Inspect git status first; do not
reset --hard, clean, blindly replace data, or discard my local commits. If dirty,
preserve work before switching; inspect differences and integrate safely.

Fetch branch cloud/basketball-pilot-20261020 from AileshS/CV-personal-project.
Exact tested implementation commit:
350107a06a03765bfadd420fa1f267e00e2715d6

Read AGENTS.md, README.md, HANDOFF.md, CLOUD_REPORT.md, docs/ANNOTATION.md and
docs/TRAINING.md. Use separate planning, implementation and independent review
agents, at most three repair cycles per milestone. Never tune on locked test data.

Safe inspection commands from this project's root:
git status --short --branch
git fetch origin cloud/basketball-pilot-20261020
git show --stat 350107a06a03765bfadd420fa1f267e00e2715d6
git diff HEAD 350107a06a03765bfadd420fa1f267e00e2715d6
After preserving existing work, integrate that exact commit in a new local branch
or safely merge/cherry-pick as appropriate. Do not overwrite my local project.

Changes: automatic analysis ignores human anchors entirely; assisted analysis uses
only reviewed anchors. Separate caches, source/setup/model/code freshness and stale
feature rejection; provenance-bearing prediction export/frozen single-use evaluation;
manifest-exact detector inputs; guarded checkpoint resume; split-safe native-frame
batches whose model suggestions stay unreviewed. Fixed invalid-media errors and
malformed notification URLs/ports. Added workflow docs and notebook safeguards.
No dependency declarations/lockfiles were changed.

Actual cloud checks: 103 tests passed; Ruff and whitespace checks passed. Viewer
HTTP health/root and AppTests passed; persistent worker completed an index job.
Real CPU detector smoke used artificial images, paused/resumed with optimizer/RNG,
completed three epochs in16.51s. Final CPU export/classifier/prediction/freeze/eval
smoke took10.55s; manufactured heldout8 eligible/0 emitted, coverage0%, accuracy
undefined. Detector comparison script also ran on two artificial images and
detected nothing. These are software checks, not basketball accuracy. No real
reviewed dataset, basketball fine-tuning, GPU/Colab/Drive recovery or final real-game
test was completed. Track IDs cannot establish cross-model defender accuracy.

Install and check using Python3.12 and uv from the project root:
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run playlab init

CPU viewer and worker in SEPARATE terminals:
uv run streamlit run app.py
uv run playlab worker

Optional neural runtime:
uv sync --locked --extra vision
Then consistently use uv run --extra vision for viewer, worker, analysis, export,
status and training; plain uv run can remove optional packages and invalidate
analysis freshness. Examples for separate terminals:
uv run --extra vision streamlit run app.py
uv run --extra vision playlab worker

The ORIGINAL full game was already downloaded LOCALLY before the pause:
data/sources/B-c3Ybc8HYE.mp4 (1920x1080,30fps,audio,approximately7170.62s).
Check it exists; preserve its saved game record. Import is idempotent:
uv run --extra vision playlab import data/sources/B-c3Ybc8HYE.mp4
uv run --extra vision playlab status
If missing, retrieve https://www.youtube.com/watch?v=B-c3Ybc8HYE remotely; do not
assume a cloud file exists locally. Cloud retrieval was blocked by proxy403;
YouTube domain additions and refreshed startup instructions were saved in the
environment draft, not proven applied/published. Local networking is separate.

Cloud artifacts are OUTSIDE GIT; fetching code does not retrieve these paths:
- artifacts/review/public-practice-v2/index.html and manifest.json:60 native frames,
  four boundary-review windows,21 proposed ball boxes on19 frames,ALL unreviewed.
- This is a DIFFERENT112.7s solo-practice clip,not my full game. COCO has no rim
  class; it cannot validate defender/full-possession behavior. Do not transfer its
  labels onto the full game or blindly merge cloud SQLite into my local Store.
- Download/extract the separate review ZIP only into a new location:
  artifacts/handoff/review-public-practice-v2.zip
  SHA256 d16eee9c4f02f23734dd91ae05d57bc435a5f9cb6b5536cec5e419011fc87d6c
- Execution evidence ZIP (reports/commands;no weights/datasets/video):
  artifacts/handoff/cloud-execution-evidence.zip
  SHA256 ecb3ff0b27b4e23371a2cd899fe60868514c5baab7db578847beef52dae18dc5
- artifacts/handoff/artifact-index.json lists archive sizes/hashes/model identities.
- Practice source cloud path:data/sources/public-basketball-video-test-5.mp4;
  SHA25633c8337dca309217b8ec3608ed5baa4a204a13e1a9c0f7edbfa8612fb14ae3e5.
  Attribution:avishah3/AI-Basketball-Shot-Detection-Tracker,
  commit4eb004437919f3667dce3d0a95575a510d763485,file video_test_5.mp4.
  Only footage was used,no source code or third-party trained weights.
- Official baseline:artifacts/models/yolo11n.pt (hash in CLOUD_REPORT.md).
- Artificial-data model:artifacts/smoke/training-20261020-01/detector-run/detector/weights/best.pt.
- Synthetic classifier:artifacts/smoke/pipeline-02/classifier-run/classifier.joblib.
  These smoke models are NOT basketball models;do not use them for real predictions.
  Model/dataset/report hashes and exact evidence paths are in CLOUD_REPORT.md.

Next:inspect original game footage;declare whole-game splits before sampling,or
explicit guarded contiguous blocks for a same-game pilot. Prepare40–60 editable
whole-possession review items and roughly300 diverse native frames;do not call
arbitrary windows possessions. Example once enough split-safe proposals exist:
uv run --extra vision playlab index GAME_ID --queue
uv run --extra vision playlab prepare-label-batch artifacts/review/game-batch-01 --frames 300 --possessions 50 --split train --split val --model-assisted
Assign splits in Game setup BEFORE sampling;replace GAME_ID with actual status ID.
Human review remains required for boundaries/offensive rebounds,replays,basket/team
colors,every visible ball/rim,shot release/outcome,and closest visible opponent.
Keep suggestions unreviewed until I check them. Empty suggestions are not negatives.
Fouls/turnovers remain manual. Export only reviewed labels to a NEW dataset version.
Continue independent coding/validation while awaiting labels/free GPU;clearly record
real blockers. Do not promise uninterrupted training or basketball accuracy.

Phone: I said I would enable Codex task notifications;receipt is still unverified.
Help me confirm one actual phone receipt. Optional worker PLAYLAB_NTFY_URL and
PLAYLAB_NTFY_TOKEN were absent remotely;do not claim delivery. Configure secure
values only on the machine running the worker,then run playlab notify-test and
ask me to confirm receipt. Service acceptance alone is not phone receipt.

Cloud viewer,worker,and training processes were deliberately stopped;NONE remain
running. Start local services yourself. Stop your worker/viewer gracefully with
Ctrl+C. Never use worker --recover until the previous worker has been proved stopped.
```
