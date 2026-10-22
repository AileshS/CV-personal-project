# Quick analysis

Quick analysis is the whole site (`streamlit run app.py`). The ten-second ball/rim
video flow in [TRACKED_VIDEO.md](TRACKED_VIDEO.md) and the labeling tools are in
`labeling_app.py`. With `artifacts/models/ball_rim.pt` present, Quick analysis uses
that ball/rim model; the COCO limitations below apply only without it.

Start with a short video interval, run inference, and inspect each proposed shot
and its evidence. Possession cutting, frame selection, box labeling and dataset
partitioning are not prerequisites. Analysis intervals are not verified whole
possessions and are never exported as human training labels.

## Start the product

From the existing project checkout, retain the optional pretrained runtime:

```sh
uv sync --locked --extra vision
uv run --extra vision streamlit run app.py
```

In another terminal, using the same existing data directory:

```sh
uv run --extra vision playlab worker
```

Both processes use `data/` unless `PLAYLAB_DATA` is set. Do not start a duplicate
worker or delete an active worker's lock. After an interrupted process, confirm
the old worker has stopped before using recovery. Local processes stop when the
computer is off.

## Open → analyze → inspect

1. Open **Quick analysis**, the default workspace. Select an imported video or
   open a local video directly. Existing source files and database records are
   retained; opening a clip does not create possessions or annotations.
2. Choose a start timestamp and duration. The default is twenty seconds, capped
   at sixty seconds and shortened at the end of the file. These are inference
   intervals, not claims about complete possessions.
3. Select **Analyze interval**. The worker prepares bounded playback and runs
   automatic inference. Refresh the status to see progress. Failed or cancelled
   jobs can resume from completed inference chunks; changes to inputs require a
   new interval request.
4. Inspect the bounded video, individual attempts and release evidence. Result
   times refer to the original source. Multiple attempts remain separate; there
   is no single made/missed verdict for an entire clip. An empty attempt list
   means insufficient proposals, not proof that no shots happened.

Quick analysis never reads reviewed shot release anchors. Optional basket/team
hints can help existing inference but are separate from verified setup and gold
labels. Leave hints unset to run immediately. A fixed basket hint may become
invalid after a camera pan or cut; uncertain evidence must remain unresolved.

## Original source times and the demo

For an ordinary full video, the original source offset is zero. For an excerpt,
provide its elapsed offset or its matching provenance JSON when opening it.
The mapping is:

```text
original source seconds = file seconds + source offset seconds
```

The attached demonstration package contains `clip.mp4` and `provenance.json`.
Its file interval `0–50` maps to original source `750–800` seconds, or
**12:30–13:20**. It is unreviewed training-partition footage, not a labeled test.
When available under `artifacts/handoffs/quick-analysis-demo/`, choose **Open
demonstration clip**. Otherwise use **Open a local video**, give the extracted
video path and matching provenance path, and leave the offset blank so the
provenance supplies it. A Git fetch alone does not transfer this package.

Provenance imports check the video hash and time mapping. Conflicting metadata
for an existing source is rejected rather than silently changing its identity,
partition or existing labels. Playback uses file time; displayed results and
evidence preserve original elapsed time, not the scoreboard clock.

## What a result means

### Inspect frames and detector boxes

Under an analysis interval, enable **Show frame inspector**. In **Any frame**,
use the frame slider or Previous/Next frame, then select **Detect this frame**.
This runs a fresh diagnostic on that frame only, including for older results;
it does not save verified labels. Boxes and detector scores appear when the
model reports candidates. No detection is shown explicitly. Scores are not
confidence intervals or calibrated probabilities. The default COCO model has
no rim class, so it cannot provide predicted rim boxes or rim scores.

New analyses retain raw **Saved model samples**, approximately ten per second,
even when shot results are unknown. These are distinct from native frame steps.
Saved samples are hidden when their result is stale. Original source timestamps
and native image dimensions are displayed. Frame stepping is approximate for
variable-frame-rate footage. Optional basket hints are visibly separate from
model predictions and have no detector confidence.

If a job stays queued, check the worker terminal and select **Refresh analysis
status**. The viewer reports a confirmed stopped worker. Recovery refuses to
remove a live or unverifiable worker's lock. Start the worker with the same
data directory; use `worker --recover` only after confirming the old worker
has stopped.

The existing COCO pretrained model detects people and balls, **not rims**.
Without dependable basket evidence it cannot establish a rim crossing, and may
produce no supported shot proposals. The product must show **unknown / review
required** with that reason. Missing balls, ambiguous tracks, occlusion and camera
changes can also leave an attempt unresolved. Diagnostic color candidates are
not trained basketball detections and cannot justify made/missed claims.

Any made/missed prediction remains unverified evidence to inspect. Closest visible
opponent is approximate image proximity and is withheld when shooter, team or
release evidence does not permit it. No software test or demonstration output is
a game-accuracy measurement.

Fine-tuning, bulk labeling, locked evaluation and paid compute are deferred.
Detailed review remains available for later corrections; it is not a prerequisite
for using this product. Phone notification receipt remains unverified.
