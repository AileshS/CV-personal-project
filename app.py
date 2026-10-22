"""Basketball play lab: open a video, analyze a short clip, read each shot attempt.

Labeling and review tools live in ``labeling_app.py`` (``streamlit run labeling_app.py``).
"""
from __future__ import annotations

import os

import streamlit as st

from playlab.quick_analysis_ui import render_quick_analysis
from playlab.store import Store


def clock(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, float(seconds)), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}:{minutes:02d}:{remainder:05.2f}"


st.set_page_config(page_title="Basketball play lab", page_icon=":material/sports_basketball:",
                   layout="centered", initial_sidebar_state="collapsed")
st.title("Basketball play lab")
store = Store(os.environ.get("PLAYLAB_DATA", "data"))
if "quick_imported_game" in st.session_state:
    st.session_state["game_choice"] = st.session_state.pop("quick_imported_game")

games = {g["id"]: g for g in store.list("game")}
selected = st.selectbox("Video", list(games), format_func=lambda key: games[key]["name"],
                        key="game_choice") if games else None
render_quick_analysis(store, games.get(selected), clock)
