# Cloud review-workspace handoff

Base: `local/basketball-pilot-20261020`, exact commit
`17a78843e6295bcc52b9b2f05e34d34d62dc882a`.
Work branch: `cloud/review-batch-workspace-20261021`.

## Preservation and data scope

The initial cloud checkout had no uncommitted source changes. Its original commit
`5c39f6c249a648d1f1480dfa5ffc88099667a368` is preserved on the local branch
`preserve/cloud-before-review-workspace-20261021` and is also an ancestor of this
milestone. Ignored runtime files were retained. The cloud `data/` was archived and
a consistent SQLite backup passed its integrity check under
`/workspace/cloud-preservation/`. This cloud database was empty; it is not a copy
of the user's populated local database.

The attached preview archive matched SHA-256
`8e943b95f5824eb3e89de2bf7f4f4c229a6acf3a26563b3727caf74946bc26bf`.
Its manifest and 300 training/validation thumbnails were verified and extracted
under `artifacts/review/original-game-v1/`, outside Git. Batch identity:
`50b7579f30938973c05b4ed62657a0d8e15a7bf25c723d709edb130654ba0c6c`.
The native PNGs, original game and populated SQLite database remain local.
No thumbnail was used for training or as a verified label. No real human label
was invented, no locked test frames/results were opened, no training or locked
evaluation was run, and no money was spent.

## Review workflow

The viewer has a **Review batches** workspace for validated versioned manifests,
window/frame navigation, original elapsed timestamps, unreviewed suggestions,
durable review progress and human corrections. Preview-only packages can be
browsed without importing a game. Missing original assets create a deduplicated
human-action request without sending a notification.

A sampling window can contain multiple possessions. Each selected frame links
explicitly to a separately reviewed manual possession; the broad original
proposal and batch suggestions remain unchanged. Training/validation source
membership and guards are checked before labeling. Unresolved boundary work
remains a draft. Revision checks prevent stale sessions from overwriting human
corrections, including concurrent frame-box edits.

With matching native frames and source video, a confirmed frame opens the
existing **Frame labels** editor at its exact sampled timestamp. Suggestions are
reference material, never bulk-accepted gold. Users must correct every visible
ball/rim and explicitly review the complete frame or a true negative. No dataset
export is triggered by loading, navigation or linking.

Boundary review can queue short original-source context clips. Default bounds
come from the selected sampling window; explicitly adjusted context is capped at
120 seconds within its guarded training/validation partition. The worker checks
source identity and current policy again. Playback never opens the full original
game or uses the nearly full-game proposal as the clip interval.

See [the local review guide](docs/REVIEW_WORKSPACE.md) for safe integration and
the user's review actions. The final chat handoff supplies the exact pushed
commit to verify and integrate. Do not replace local data with this cloud Store.

## Validation and remaining limits

The planner, implementation agents and independent reviewer used temporary
fixtures. Software checks establish workflow behavior, not basketball accuracy.
The scoped baseline before implementation passed 78 checks. Training and locked
evaluation test modules were deliberately excluded from this milestone's checks.

Final scoped suite: **129 passed, 1 deselected**. Ruff and `git diff --check`
passed. Exact suite command:

```sh
uv run --locked pytest -q --ignore=tests/test_training.py --ignore=tests/test_evaluation.py --ignore=tests/test_benchmark.py -k 'not missing_optional_weights_allow_honest_fallback_export_and_freeze'
```

The named analysis freeze regression and the three ignored modules were not run;
this is not a claim that the complete training/evaluation suite passed. No new
dependencies or lockfile changes were needed.

Independent review found and verified repairs for stale confirmation atomicity,
per-frame draft persistence, required native-image dimensions, explicit boundary
reconfirmation and clip queue publication races. One repair cycle was used,
within the three-cycle limit. No material finding remained at review completion.

The actual attached preview was opened through Streamlit AppTest in a temporary
Store: 300 frames across 50 windows loaded, navigation resumed after restart,
and no game, possession, frame, box or shot labels were created.

A separate temporary three-second synthetic video exercised a live Streamlit
server (health and root responses), a live worker consuming a bounded clip job,
native-frame batch loading, explicit manual boundary confirmation and exact
timestamp handoff. The original coarse proposal remained unreviewed, and no
frame/box label was created by navigation or linking. All task-owned viewer and
worker processes were stopped afterward; their worker locks were removed. No
training process was started.

The human still needs to inspect the original game locally, correct complete
possession boundaries and partition membership, check basket/team setup and
verify frame/shot/event labels. Thumbnails cannot establish those facts. No
reviewed real dataset or accuracy result is claimed. Phone notification settings
are reported enabled by the user, but receipt remains unverified; optional worker
notification variables were absent in this cloud process.
