# Quick analysis product handoff

Base: `local/basketball-pilot-20261020`, verified exact commit
`27d60e16e070e8a56291e2529d8465a8ae515f7c`.
Work branch: `cloud/quick-analysis-product`.

## Product behavior

**Quick analysis** is the default workspace. Open a video, choose a timestamp and
a short duration, analyze, then inspect bounded playback and individual attempts
with evidence. No possession cutting, frame picking, box labeling or verified
setup is required. The default duration is twenty seconds; requests are bounded
to one through sixty seconds and the source file's extent.

Inference reuses the existing automatic engine through an isolated interval
context. Requests, results, chunks and files use separate Quick analysis storage.
No fake possession is persisted, no human shot anchors are read, and no result
is copied into verified annotations or training exports. Optional basket/team
hints are request settings, not gold setup. Multiple inferred attempts remain
separate, and unsupported outcomes/opponents remain unresolved.

Queue publication is atomic. Cancellation retains completed chunks for resume.
Immutable request identity includes source/time mapping, settings and model
identity. Source/model/code changes invalidate results instead of silently
reusing stale evidence. Known locked-test intervals reject before source-content
reads; a shared source with test blocks uses metadata identity rather than
hashing held-out head/tail bytes. There is no training or evaluation action in
this workflow.

See [the product guide](docs/QUICK_ANALYSIS.md) for startup and user steps.

## Preservation

The previous cloud commit `67587f8d9dbb316a7807d5cc364fea486af95b27` is preserved
on `preserve/cloud-before-quick-analysis-67587f8` and remains in the new base's
ancestry. The initial checkout had no uncommitted source changes. Ignored data,
media, previous evidence and local environments were retained. A consistent
SQLite backup passed integrity checks at
`/workspace/cloud-preservation/playlab-before-quick-analysis.sqlite`.

The original local database and full game were not copied to cloud or modified.
Never replace the user's populated local database with the cloud Store. This
change adds record kinds to the existing Store, not a destructive data migration.

## Attached demo and actual execution

The uploaded archive contains only `clip.mp4` and `provenance.json`, not a
database, human gold, model weights or locked-test footage. Verified identities:

- Archive SHA-256:
  `0c72b32de653f9027cddeb833cb7ba028f2de8b41f99bbb04b1f934f37e0bc11`.
- Video SHA-256:
  `8a735b8f1aee02c79cadf28cad49f9d2bb9aa5b934949254fe3039a97696a2de`.
- Native demo: fifty seconds, 960×540, 30 fps, audio present; file `0–50` maps
  to original source `750–800` seconds (**12:30–13:20**), training partition.
- Official pretrained YOLO11n SHA-256:
  `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.

The final frozen-code AppTest flow opened the demo, queued the default interval
without labels, and refreshed a live worker's result. It processed **200 frames**
from file `0–20`, original source **12:30–12:50**, using pretrained COCO inference
on CPU. The run completed in about **44 seconds** including worker processing and
UI refresh. Bounded twenty-second playback retained audio. Context evidence is
stamped original source `760.00s` (12:40). No possession, shot, box, frame, event,
verified setup or ordinary analysis records were created.

**Demo outcome: unknown / review required.** No dependable shot attempts were
proposed. The current COCO model has no rim class, and no basket/team hints or
compatible custom rim model were supplied. Zero proposals do not mean zero
actual shots. This run demonstrates execution and honest abstention, not
basketball accuracy or made/missed capability on this footage.

Retained cloud-only evidence:

```text
artifacts/handoffs/quick-analysis-demo/clip.mp4
artifacts/handoffs/quick-analysis-demo/provenance.json
artifacts/product-demo/quick-analysis-demo-v2/report.json
artifacts/product-demo/quick-analysis-demo-v2/data/
```

Git does not transfer these files. The original package remains the portable
source; do not transfer this isolated demo database over the user's local Store.

## Checks and limits

- Scoped suite: **175 passed, 1 deselected**. Exact command:

  ```sh
  uv run --locked pytest -q --ignore=tests/test_training.py --ignore=tests/test_evaluation.py --ignore=tests/test_benchmark.py -k 'not missing_optional_weights_allow_honest_fallback_export_and_freeze'
  ```

- Ruff and `git diff --check` passed. No dependency declarations or lockfile changed.
- Independent final focused review: **71 passed, 1 deselected**. One repair cycle
  fixed rejection-before-source-hashing and queued default-model path identity.
  No remaining material findings were identified.
- Live Streamlit health and root requests returned the expected HTTP 200 content;
  the AppTest flow exercised actual UI state and the pretrained worker result.
- Runtime: Python 3.12, Torch 2.14.1, Ultralytics 8.4.183; CPU only, CUDA unavailable.
  The cloud optional runtime is retained at `/workspace/.venv-playlab-vision`.
- All task-owned viewer/worker processes were stopped after validation. No
  training process was started and no money was spent. Actual browser pointer
  interaction was not exercised; Quick analysis uses ordinary Streamlit widgets.

No training, real held-out evaluation or accuracy measurement was performed.
The training/evaluation modules above and the named synthetic freeze regression
were deliberately outside the final suite. Fine-tuning and bulk labeling remain
deferred. A compatible rim detector or optional basket hint is still needed for
rim-based proposals; missing evidence must not be converted into invented results.
Phone receipt remains unverified; no notification delivery is claimed.

## Return to local

Use the exact tested commit from the final chat handoff. Stop old processes,
preserve local files and uncommitted work, take a consistent SQLite backup, and
integrate on the local pilot branch without resetting the database or source.
Retain `--extra vision` for installation, viewer and worker. Open **Quick
analysis**, choose a short interval from the existing full game (for example
source time 750 seconds), analyze and inspect. No additional bulk annotation is
required to start. Keep detailed review available for later corrections.
