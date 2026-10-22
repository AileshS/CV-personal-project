# Ready-to-copy prompt for local Codex

Continue Basketball Play Lab from remote branch
`cloud/ball-rim-video-product` at exact commit
`692d3547eb672a50a8f92acf302e71a8a9d726cd`.

Preserve all existing local work. Leave older projects untouched. Do not hard
reset, delete, overwrite, or replace the populated local SQLite database, local
videos, weights, labels, artifacts, or ignored files. Before integration, record
`git status --short --branch`, stop only this project's old viewer/worker after
verifying their command lines, and create a consistent SQLite backup outside the
checkout if `data/playlab.sqlite` exists. Verify the backup with SQLite
`PRAGMA integrity_check`.

Fetch and verify the exact remote commit:

```sh
git fetch origin cloud/ball-rim-video-product
git rev-parse origin/cloud/ball-rim-video-product
git show --stat --oneline 692d3547eb672a50a8f92acf302e71a8a9d726cd
```

Integrate it without discarding local commits or changes. If the local branch is
clean and directly behind, fast-forward it. Otherwise create a safety branch and
merge or cherry-pick with the local work preserved. Do not use `git reset --hard`.

Install and launch from the repository root with Python 3.12:

```sh
export UV_CACHE_DIR="$PWD/.uv-cache"
uv sync --locked --extra vision
uv run --locked --extra vision streamlit run app.py
```

The main page is Quick Analysis. Open the local game, select a short interval,
and choose Analyze. Under **Frame-by-frame model boxes**, use Saved model samples
and Previous/Next or the slider. Confirm scan, dense-shot-check, and
release-player stages appear when the selected clip produces those calls, and
that every row displays all returned ball, rim, and player boxes. Any-frame
detection is a separate one-frame diagnostic.

Expected local inputs are outside Git:

- Original full game: `data/sources/B-c3Ybc8HYE.mp4` if already present.
- Default custom model: `artifacts/models/ball_rim.pt`, or the path configured by
  `PLAYLAB_BALL_RIM_WEIGHTS`.
- Existing records and labels: `data/playlab.sqlite`.
- Cloud-only practice source used for timing:
  `data/sources/public-basketball-video-test-5.mp4` (not transferred by Git).

Run these checks after integration:

```sh
uv run --locked --extra vision pytest -q
uv run --locked --extra vision ruff check .
```

If the full suite has only `tests/test_app.py::test_unconfirmed_shot_is_not_saved`
timing out at the Streamlit 15-second harness limit, rerun that exact test alone
and report both results. Do not treat synthetic tests, pretrained smoke checks,
or detector coverage as basketball accuracy. Keep all proposed labels unreviewed
until the user verifies them. Fouls and turnovers remain manual.

Measure one 12-second interval on the user's actual video and report total time,
inference calls by stage, and returned-box counts. Do not claim the cloud's 5.82×
speedup will transfer unchanged to the local machine. The cloud acceptance used
a no-shot practice interval, so dense candidate-window speed and original-game
accuracy are still open checks.

No cloud training or analysis process remains running after this handoff. No
training was started. The saved cloud environment draft still requires user
review and publication if it has not already been published.
