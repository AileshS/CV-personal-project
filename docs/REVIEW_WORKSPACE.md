# Review the original-game batch

This workflow checks suggestions and saves human corrections. It does not train
models or open locked test results. Sampling windows are navigation aids, not
verified whole possessions. Keep offensive rebounds in the same offensive
possession, and leave uncertain frames or boundaries unreviewed.

## Preserve the local workspace first

Use the existing Basketball Play Lab checkout. Stop its viewer and worker before
integrating code. Preserve uncommitted source changes, the existing `data/`, and
`artifacts/`; do not reset the checkout, replace the database, reimport the game,
or copy a cloud database over it. Make a consistent SQLite backup with Python's
`sqlite3.Connection.backup()` and verify `PRAGMA integrity_check` on the backup.
Preserve the videos, native review images, weights and artifacts separately.

Fetch the tested cloud branch and verify the exact commit supplied in the cloud
handoff. Integrate with a fast-forward on the existing local pilot branch when
possible. If local commits diverge or changes overlap, inspect and preserve them
before merging; do not force a reset or a push.

## Start locally

The local handoff says the optional vision runtime is installed. Retain that
runtime consistently, even though browsing and manually correcting this batch
does not require loading a model:

```sh
uv sync --locked --extra vision
uv run --extra vision streamlit run labeling_app.py
```

In a second terminal from the same checkout:

```sh
uv run --extra vision playlab worker
```

Both processes must use the same existing `PLAYLAB_DATA` directory (default:
`data/`). A worker is needed only for queued operations; frame review does not
require reindexing the full game or running inference. Do not start a duplicate
worker. Use recovery only after confirming the previous worker has stopped.

## Load and review

Open **Review batches** and load the existing local manifest:

```text
artifacts/review/original-game-v1/manifest.json
```

The full local batch contains native PNG frames and thumbnails. The small preview
archive contains thumbnails only; it is for browsing context and suggestions.
It cannot substitute for native images, the original video or the local SQLite
database. No game or gold label is imported from a manifest. The viewer validates
batch identity, paths, references and selected image hashes before using them.

Navigate review windows and frames using the controls. All times are elapsed
seconds in the original source, not the scoreboard clock. Suggestions remain
unreviewed even after navigation, saving notes, or opening frame labeling. An
empty suggestion list is not a reviewed negative frame. Current selection and
review notes are saved in SQLite so review can resume after restarting.

With the matching original video available, queue a short review-context clip.
The default interval is the selected sampling window; adjust the context if the
offensive possession begins earlier or ends later. Context is limited to 120
seconds and must stay inside the same guarded training/validation partition.
The worker checks the source again before creating the clip. Only that bounded
output is played in the viewer; the coarse parent proposal is not used as a
clipping interval. Refresh after the worker completes the job.

For each frame you intend to label:

1. Inspect the original video around the sampled time. Determine the complete
   offensive possession, including rebounds and uncertain transitions; do not
   simply accept the thirty-second sampling window.
2. Enter the corrected possession boundaries or explicitly link a reviewed
   possession. Confirm its training/validation membership against the existing
   source blocks and ten-second guards. Do not change the declared source split
   policy to force a sample to fit. Save uncertainty and rebound notes explicitly.
3. Confirm boundaries only when you have checked them. A corrected possession is
   separate from the original broad proposal. Existing suggestions and human
   labels are retained. If another session changed a linked record, reload and
   reconfirm rather than overwriting it.
   Saving notes also preserves draft bounds and target selection without
   confirming them. To revise an already-linked possession's bounds, use **Edit
   linked possession boundaries**, save in **Possessions**, then return to the
   same batch frame, reload and explicitly reconfirm its current target.
4. Open the selected timestamp in **Frame labels**. Inspect the full frame,
   correct/add every visible ball and rim, and save boxes individually. There is
   no bulk acceptance of model suggestions. Save a complete-frame review only
   after all boxes are checked, or explicitly mark a checked negative frame.
5. Leave ambiguous/incomplete work unreviewed. Use **Possessions** for separately
   reviewed shot outcomes and manual foul/turnover labels, and **Game setup** for
   checked basket/team settings. The frame batch does not establish those labels.

If the original game, native image or matching source record is unavailable,
follow the displayed action request and restore the existing local assets. Do not
create a substitute game or use thumbnail coordinates as verified labels.
Reviewing thumbnails alone does not establish full-possession boundaries.

Only after human review, export a **new** immutable reviewed dataset version;
never edit a previous export in place. This milestone does not run export,
training, locked evaluation, or any accuracy measurement.

## Notifications and process limits

Codex/ChatGPT phone notifications may be enabled, but receipt remains unverified.
The optional worker notification channel is separate. Saved human-action requests
can be read in **Jobs and export** even without delivery. No successful phone
receipt is implied by an enabled setting or an accepted service request.

A local viewer or worker stops when its computer is off. A cloud code fetch does
not transfer local videos, SQLite records, native frames or model files.
