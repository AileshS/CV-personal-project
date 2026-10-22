# Cloud continuation results — October 20, 2026

Base: `main`, `5c39f6c249a648d1f1480dfa5ffc88099667a368`.
Work branch: `cloud/basketball-pilot-20261020`. Deadline: October 23, 2026.
No ArcLab code/data/model assumptions were used. No paid resources were provisioned.

## Changes and checks

- Separate planner, implementation agents and independent reviewer; one repair cycle.
- Baseline reproduced: 64 passed / one invalid-video exception failure; Ruff passed.
- Final suite: **103 passed in 8.58 seconds**; Ruff and `git diff --check` passed.
- Fixed system-ffprobe invalid-media errors and malformed notification URL/port
  crashes; failure handling preserves requests and continues worker processing.
- Explicit independent automatic versus reviewed-release-assisted inference, separate
  caches, source/setup/model/code freshness checks and stale feature rejection.
- Prediction JSON carries dataset/split/mode and model provenance. Locked evaluation
  checks the frozen prediction/config/model artifacts before its single-use report.
- Reviewed manifest image lists drive detector training and benchmarking. Unlisted
  images/labels are rejected. Resume checks checkpoint ownership, hashes, runtime,
  epochs, image size and RNG sidecars.
- New deterministic, partition-safe native-frame review batches and HTML index;
  all model proposals remain unreviewed and human Store records are unchanged.
- Notebook cells compiled; actual archive extraction tested for correct hash,
  wrong hash and traversal rejection. Notebook GPU/Drive cells were not executed.
- Final-code viewer restart: internal health `200 ok`, root `200`; AppTest suite
  passed. Separate persistent worker completed an index job. Both were then stopped.

CPU runtime: Python 3.12.14, uv 0.12.23. Isolated optional runtime:
Torch 2.14.1, Ultralytics 8.4.183; CUDA availability **false**.

## Measured software workflows

| Workflow | Executed result | Evidence outside Git |
| --- | --- | --- |
| Real detector recovery on artificial shapes | 4 train / 2 validation images, 96 px, CPU; paused after epoch 1, resumed at epoch 2 with 255 optimizer-state entries and RNG restored; completed epochs 1–3 in 16.51 s | `artifacts/smoke/training-20261020-01/smoke-report.json` |
| CPU export/classifier/prediction/freeze/evaluation | Exported 24 manufactured shot labels; fit on 8 train / 8 validation; geometry selected; automatic synthetic heldout: 8 eligible, 0 emitted, coverage 0%, accuracy undefined; wrong dataset/split and repeat evaluation rejected; 10.55 s | `artifacts/smoke/pipeline-02/smoke-report.json` and `commands.json` |
| Actual detector comparison script | Same two artificial validation images for official COCO baseline and artificial-data model; neither detected objects at the configured threshold (2 false negatives per class; precision undefined). COCO rim unsupported | `artifacts/smoke/detector-benchmark-01/report.json` |

These are execution/recovery checks, **not basketball accuracy measurements**.
Synthetic labels exist only in isolated smoke Stores/datasets. Do not use their
models or metrics as trained basketball results. No real final test has been opened.

Final pipeline source hash:
`f261fd0a3e9c27cbec86a307756ea9432501b06252139558d0cf8842aba6b76e`.
Smoke provenance records the precommit Git revision and exact source hashes; it is
historical execution evidence. New evaluation freezes must use the final committed
revision, and changing code/revision invalidates old freezes.

## Remote footage and human-review batch

The requested full-game URL `https://www.youtube.com/watch?v=B-c3Ybc8HYE` failed
with cloud proxy CONNECT **403 Forbidden**. Domain additions saved in the draft:
`youtube.com`, `www.youtube.com`, `youtubei.googleapis.com`, `*.googlevideo.com`.
Saving did not apply/publish runtime policy. No access bypass was attempted.

An alternate public **solo-practice** source was obtained through GitHub:
`avishah3/AI-Basketball-Shot-Detection-Tracker`, commit
`4eb004437919f3667dce3d0a95575a510d763485`, file `video_test_5.mp4`.
Only that footage was used; none of its implementation or third-party weights were
used. Source metadata/provenance remains in `artifacts/source-provenance.json`.

- Local cloud path: `data/sources/public-basketball-video-test-5.mp4`.
- Duration 112.712604 s, 960×540, 29.97 fps, audio present; native frame inspected.
- Source SHA-256:
  `33c8337dca309217b8ec3608ed5baa4a204a13e1a9c0f7edbfa8612fb14ae3e5`.
- Assigned **train** before sampling; no real validation/test data fabricated.
- Final batch: `artifacts/review/public-practice-v2/index.html` and `manifest.json`.
- **60 native frames, four boundary-review windows, 21 proposed ball boxes on 19
  frames**; every frame/item/suggestion has `reviewed: false`. No human label added.
- Proposed box widths 21.36–38.48 px; these are unreviewed detector outputs, not
  verified ball-size or detector-quality measurements. COCO has no rim class.
- Batch hash:
  `77e69f9833b313c1e018f03a61f6d2a1b09c15e9f70ce8a0cf93ae640b511d2b`.

This short clip does not supply 40–60 full possessions, broad game diversity or
defender evidence. The larger game batch remains blocked by source access and human
review. Temporal samples do not establish diversity or accuracy. The older v1 batch
is superseded by v2, which includes full baseline model/version provenance.

## Artifact identities

| Artifact | SHA-256 |
| --- | --- |
| Official baseline `artifacts/models/yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Synthetic-only detector `artifacts/smoke/training-20261020-01/detector-run/detector/weights/best.pt` | `b01b6a63c4beaaf4955cce9e2826ceddf779f47625a1938c138f88b401b08303` |
| Final pipeline synthetic classifier `artifacts/smoke/pipeline-02/classifier-run/classifier.joblib` | `a6090d5ee320da9b795658474f3baa2cdafca4b0a1e27425f35ce36ced9351bb` |
| Final synthetic pipeline dataset manifest identity | `a165dee36571078eacd175ada857d9efa3b485925fd065b1847c7dd19df18b2a` |

All media, review frames, labels, datasets, weights and generated reports remain
outside Git. Portable review/evidence archives and their file hashes are listed in
`artifacts/handoff/artifact-index.json`. A source-code fetch does not restore them.

## Remaining actions and process state

- Apply the saved network settings and retry full-game retrieval; or work locally
  from the user's existing original full game without overwriting its Store.
- User must check whole-possession boundaries, game splits/replays, basket/team
  setup, every ball/rim frame and made/missed/unknown shot label. No reviewed real
  training/validation dataset or fine-tuned basketball model exists yet.
- GPU availability, Colab/Drive checkpoint recovery and real-game accuracy remain
  unverified. Defender IDs compare only within the identical frozen tracking run;
  cross-model accuracy needs human boxes/points and a dedicated metric.
- User is enabling Codex phone notifications; receipt is not yet confirmed.
  `PLAYLAB_NTFY_URL` / `PLAYLAB_NTFY_TOKEN` are absent; no worker notification delivery
  is claimed. Human-action requests persist in `data/playlab.sqlite`.
- Saved environment startup instructions were refreshed; draft changes require
  user review/save/publication. Fresh-task restoration was not tested.
- **No task-owned viewer, worker or training process remains running.** Services
  were stopped after validation; a future task must start its own processes.
