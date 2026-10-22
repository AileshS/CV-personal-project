# Quick Analysis frame evidence and CPU speed report

Code commit: `692d3547eb672a50a8f92acf302e71a8a9d726cd`

The main Quick Analysis screen now retains and displays every bounding box
returned by every model inference call. The frame browser includes the initial
scan, dense shot-window checks, and player checks around a proposed release.
Each entry identifies its timestamp, sequence, inference stage, and full-frame
or rim-crop scope. Repeated timestamps remain separate. Predictions remain
unreviewed and are never copied to human labels.

CPU inference now limits PyTorch to four compute threads and one inter-op thread.
The detector input remains 960 pixels with the same score thresholds. A 640-pixel
probe was rejected because it changed detections.

## Measured runtime

The comparison used the same cloud machine, source interval, public two-class
checkpoint, and 60-frame initial scan:

- Source: `data/sources/public-basketball-video-test-5.mp4`, file seconds 0–12.
- Weights: `artifacts/models/basketball-ball-rim.pt`.
- Before: 112.17 seconds in `perform_analysis_clip`.
- After final code: 19.28 seconds in `perform_analysis_clip` and 20.78 seconds
  for the complete benchmark process.
- Measured end-to-end analysis speedup: 5.82×.
- The final run evaluated 60 frames and found no supported shot candidate, so
  it did not exercise dense shot-window inference.

The implementer's paired evidence comparison found identical raw scan outputs
before and after thread control: 60 evaluated frames and 20 boxes. This is
runtime and software evidence, not an estimate of basketball accuracy.

## Checks actually performed

- Focused product suite: 92 passed.
- Independent final review: 65 focused tests passed; Ruff and diff checks passed.
- Full repository suite: 289 passed and one Streamlit AppTest timed out at its
  15-second harness limit under concurrent load. The exact failed test passed
  alone in 2.90 seconds.
- Real Chromium acceptance opened the final real result, moved from inference
  call 1 to call 6, displayed a real ball box and detector-score table, and
  showed the measured runtime and score caveat.
- Independent review approved after two repair cycles. The repairs retained
  dense/release boxes and eliminated an unreported chunk-overlap inference call.

## Limits

The practice interval is not an independent accuracy benchmark. Detector scores
are raw model scores, not calibrated probabilities or confidence intervals.
The cloud machine does not have the user's original full game, so performance
and detections on that game remain unverified. Dense shot-window runtime remains
unmeasured on real candidate footage. Local videos, weights, SQLite data, and
generated result records remain outside Git.
