# Basketball Play Lab

This is a NEW project. Do not copy code, data, results or assumptions from ArcLab.

## Goal and scope

Deliver a measured pilot by October 23, 2026. Whole-possession proposals remain
editable and experimental. Automate shot made/missed and approximate closest
visible opponent at release; foul and turnover labels are manual only.

## Work loop

Use planner -> implementation agent -> independent reviewer -> repair. Limit
repairs to three cycles per milestone. Select changes using development and
validation evidence. Never tune against locked test results.

## Commands

- Install: `uv sync --locked` (CPU viewer/tests). Optional model runtime:
  `uv sync --locked --extra vision`.
- Check: `uv run pytest -q` and `uv run ruff check .`.
- Website: `uv run --extra vision playlab web` (http://localhost:8600). Old Streamlit page: `streamlit run app.py`. Labeling/review: `uv run --extra vision streamlit run labeling_app.py`.
- Worker: `uv run playlab worker` (separate process).
- Inspect work needing a person: `uv run playlab attention`.

## Evidence and data boundaries

Never claim synthetic fixtures, smoke tests or detector coverage establish game
accuracy. Human-review training labels. Assign splits by game before sampling.
Same-game fallback requires declared contiguous blocks and ten-second guards.
Keep predictions separate from human corrections and retain original source
timestamps. Do not put videos, weights, secrets or generated datasets in Git.

## Cloud and human handoffs

Code and CPU checks can run in Codex Cloud once an environment is published.
Training uses a separate GPU notebook; free GPU availability is not guaranteed.
Do not spend money, create paid resources, or bypass notebook runtime limits.
Do not describe a local worker as running after the user's computer is off.

When blocked on labels/access, record a concrete action via
`playlab.attention.request_attention`. Optional phone delivery is configured by
PLAYLAB_NTFY_URL on the remote runner; absent configuration is not delivery.
Keep notifications free of footage, player names and secrets. Never print tokens.

## Communication

Use simple, scannable explanations: main result first, details only as needed.
