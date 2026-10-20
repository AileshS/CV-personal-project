#!/usr/bin/env sh
set -eu
python3 -m pip install uv
uv sync --locked
uv run pytest -q
uv run playlab init
