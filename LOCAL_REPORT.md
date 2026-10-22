# Local continuation and next cloud task — October 20, 2026

This is the latest handoff. Read it before the older `HANDOFF.md` and
`CLOUD_REPORT.md`. Deadline: October 23, 2026; free resources only.

## Integrated and preserved

- Continued in the existing independent Basketball Play Lab checkout. Older
  projects were not changed.
- Created `local/basketball-pilot-20261020` and fast-forwarded to exactly tested
  cloud commit `350107a06a03765bfadd420fa1f267e00e2715d6`. Local `main` was not moved.
  The later remote documentation commit was not implicitly integrated.
- Preserved the existing SQLite database, source videos, model files and generated
  artifacts. A consistent SQLite backup was taken before integration.
- The original full game decodes locally: 1920×1080, 30 fps, approximately
  7,170.62 seconds. Source: https://www.youtube.com/watch?v=B-c3Ybc8HYE .
- Cloud practice footage/archives were not transferred. That separate source
  cannot stand in for this game or its player/defender evidence.

## Real-footage result and partitions

The full-game indexing job completed. Its six coarse proposals are **not six
verified possessions**: one spans nearly the entire game, and most others fall
after play. The baseline is unsuitable for unattended possession splitting on
this footage without further development and human boundary review.

Before review-frame sampling, declared the following continuous source blocks
(elapsed time, not game clock):

| Partition | Declared interval | Usable interval after guards |
| --- | --- | --- |
| Training | 12:00–48:00 | 12:10–47:50 |
| Validation | 57:00–71:00 | 57:10–70:50 |
| Locked final test | 73:00–94:00 | 73:10–93:50 |

This is a **same-game pilot**, not generalization across games. No final-test
labels, training examples or accuracy report have been prepared/opened. The
initial whole-game indexing scan included the test timeline before this handoff;
it was not a locked accuracy evaluation. Do not infer unobserved final-test
performance. Preserve the declared split policy during further development.

## Saved review package (outside Git)

`artifacts/review/original-game-v1/` contains **300 native 1920×1080 PNGs** and
300 suggested-box thumbnail previews across **50 thirty-second review windows**:
25 training / 25 validation windows, with 150 frames per partition. These remain
unreviewed sampling aids, not confirmed possessions. All 600 listed image hashes
and the canonical manifest hash were verified. No human shot, box, frame, event
or team/basket setup labels have been saved.

The pretrained COCO model proposed 61 ball boxes on 53 frames and no rim boxes;
these counts are not detection accuracy. All proposals remain unreviewed. Empty
outputs are not verified negatives. Native frames still need a human diversity
and label-quality check.

- Batch hash: `50b7579f30938973c05b4ed62657a0d8e15a7bf25c723d709edb130654ba0c6c`.
- Full archive: `artifacts/review/original-game-v1.zip`, 421,234,962 bytes;
  SHA-256 `2fd2f41c60bce8748da736cbbafe8c72627bcf1d8cefc1a440f6c67703d8efee`.
- Preview archive: `artifacts/review/original-game-v1-preview.zip`, 7,363,066 bytes;
  SHA-256 `8e943b95f5824eb3e89de2bf7f4f4c229a6acf3a26563b3727caf74946bc26bf`.
  Native PNGs are intentionally absent from this smaller archive. It is only for
  UI/context review; its manifest references missing native files. Never train
  from thumbnails or describe it as a complete dataset.
- Machine-readable checks: `artifacts/review/original-game-v1.integrity.json`.

Neither archive contains the video or SQLite database. Attach an archive to the
cloud task explicitly; a Git fetch alone provides only the code and this report.

## Local repair and checks

Independent review found that unavailable optional pretrained weights were listed
as inference artifacts. Diagnostic fallback could run, but evaluation freezing
then tried to hash a nonexistent file. The repair records unavailable weights
separately, preserves cache invalidation when they become available, and leaves
explicit missing custom weights as an error. Analysis dependency version is 6.

- Final local suite: **104 tests passed**; Ruff and whitespace checks passed.
- Focused analysis/benchmark checks: 23 passed. Independent reviewer checked the
  repair; no further issue was identified in that change.
- Neural runtime is installed locally, CPU only. Keep `--extra vision` on `uv`
  commands that should retain it, including viewer/worker commands.
- No real basketball model has been fine-tuned, selected or accuracy-tested.
  Artificial fixtures and software tests are not basketball accuracy evidence.

The local viewer and worker were started and the viewer health endpoint returned
`ok`. Both were stopped after the indexing job finished; no unfinished queue jobs
remain. Recovery was run once after stopping the old worker to clear its stale
lock. A second consistent SQLite backup preserves the final local state. No
task-owned viewer, worker or training process is left running for this handoff.

## Next cloud milestone

1. Fetch `local/basketball-pilot-20261020`; work from its pushed head after
   preserving any current cloud work. Read `AGENTS.md`, this report,
   `docs/ANNOTATION.md`, `docs/TRAINING.md` and `CLOUD_REPORT.md`.
2. Use planner → implementation → independent review → repair, at most three
   repairs per milestone. Improve the review workflow before attempting training.
3. Add a review-batch workspace to the viewer. Load the versioned batch manifest,
   move between review windows/frames, show original timestamps and unreviewed
   suggestions, and prefill frame selection without marking anything reviewed.
   Help the user correct whole-possession boundaries and partition membership
   before linking labels. Never turn a sampling window into a verified possession
   automatically. Retain offensive rebounds and uncertain cases explicitly.
4. Make review work survive restart; preserve human corrections separately from
   suggestions and use existing revision checks. Handle missing video/images with
   a clear action request. Avoid queueing a nearly full-game clip when the user
   wants a short review window. Validate manifest paths and references before
   reading files. Do not overwrite local data or bulk accept model proposals.
5. Test the workflow with temporary fixtures. If the real review archive is
   attached, inspect training/validation samples only. It has no source video;
   local clips cannot be assumed available. Do not recreate or guess gold labels.
6. Do not launch training or locked evaluation: there are no reviewed real labels
   yet. Leave detection/classifier experiments prepared for the next handoff.
7. Finish with commits, checks, remaining limitations and a copy/paste prompt for
   the local session. The local prompt must preserve the existing database,
   import only the tested code, start the viewer/worker, and guide human review
   before exporting a new immutable reviewed dataset.

## Computer-off work and phone alerts

The user will open the cloud session. This report does not start one. A local
viewer or worker cannot keep running with the computer off. Cloud code access
does not transfer local SQLite, videos, weights or review archives; attach the
review archive explicitly if the next task needs those images.

The user says Codex/ChatGPT phone notifications are enabled, but receipt is
**unverified**. Optional `PLAYLAB_NTFY_URL` / `PLAYLAB_NTFY_TOKEN` are not configured
locally; the notification test reported no delivery. Never claim a phone ping was
received. Configure secrets only in the actual remote runner, outside Git. Record
concrete human actions with `playlab.attention.request_attention`; notify only
when action is needed, work completes or a meaningful failure occurs. Do not
create a local scheduler as a substitute for cloud execution.
