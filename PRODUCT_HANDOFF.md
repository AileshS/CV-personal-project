# Next milestone: a usable clip-analysis product

The user has confirmed that drawing boxes is quicker, but possession cutting and
frame selection still take too much time. Their latest priority is to ship a
usable product sooner and fine-tune later. Pause the large manual labeling batch.

## Current state

Continue the independent Basketball Play Lab project. Read `AGENTS.md`,
`docs/FAST_REVIEW.md`, `docs/PUBLIC_DATA.md` and the earlier review reports.
The local pilot branch includes the integrated cloud review workspace, live box
previews, two-click drafts, and Accept/Reject/Correct actions for existing model
suggestions. Latest feature commit before this handoff: `b6e2fe2`.
38 focused checks passed; independent review verified two repairs. Browser
interaction itself remains unverified because local Browser Use is disabled.

The accidental 19:40–20:10 manual possession and its review link were removed
locally after a consistent SQLite backup and an undo archive. Local source video,
native frames, weights and remaining annotations are preserved outside Git.
Never replace the local database with cloud data.

Public detector data is reproducible via `scripts/prepare_public_detector.py`:
296 train / 124 validation images, 311 training ball boxes and 33 training rim
boxes. Validation contains no rim labels. No model was trained; this package does
not establish rim accuracy or create new automatic suggestions. It is separate
from user-verified labels and the local locked test. Fine-tuning is deferred.

## Build this first

1. Add a prominent **Quick analysis** workflow: select a short source interval,
   analyze it, view the result and evidence. Do not require manual possession
   confirmation, frame selection or box labeling before analysis.
2. Let the user start from a timestamp with a short default duration and adjust
   the clip if useful. Keep preview/inference bounded and resumable. Never use
   the nearly full-game coarse proposal as the default analysis interval.
3. Sampling windows and analysis clips are not verified whole possessions.
   Keep them separate from human gold and training exports. Preserve original
   elapsed timestamps and any source offset when a demo clip is imported.
4. Wire the best existing inference into this workflow. Show made/missed only
   when supported by evidence; otherwise show unknown/review required with a
   concrete reason. Display release evidence and the closest visible opponent
   only when shooter/team/release evidence permits it. Keep image distance
   approximate. Multiple attempts in a clip must remain separate results.
5. If needed, allow a short optional basket/team setup, with clear unresolved
   status when absent. Do not turn that setup into a long annotation prerequisite.
   Keep the detailed review workspace available for later corrections.
6. Use existing pretrained inference first. Do not train, run locked evaluation,
   spend money, or reopen the large data-collection sprint in this milestone.
   If model availability blocks useful results, document the exact missing model
   and prepare a separate bounded GPU experiment rather than inventing predictions.
7. Test the complete user flow and failure cases. Use the planner → implementer →
   independent reviewer → repair loop, at most three repairs per milestone.
   Software tests and demo outputs are not an accuracy estimate.

## Real demonstration clip

An optional local upload package, `artifacts/handoffs/quick-analysis-demo.zip`,
contains one 50-second training-partition clip from original source time
12:30–13:20, plus provenance. It contains no SQLite database, human gold labels,
weights, or test footage. A Git fetch does not transfer this package; the user
must attach it explicitly if cloud access to the clip is needed.

## Return handoff

Commit/push the tested cloud work and provide an exact local continuation prompt.
That prompt must preserve existing local files/database, back up SQLite, verify
the exact cloud commit, retain `--extra vision`, and start the viewer/worker.
The desired human acceptance check is **open a short clip → analyze → inspect
results**, with no further bulk annotation required. Report limitations honestly.
