"""Short-interval product workspace; all decoding and inference runs in the worker."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from playlab.detection_inspector_ui import render_detection_inspector
from playlab.quick_analysis import get_analysis_clip, queue_analysis_clip
from playlab.quick_import import import_quick_source
from playlab.worker import worker_status


def _import(store, source, **metadata):
    try:
        game = import_quick_source(store, source, **metadata)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not open source: {exc}")
        return
    st.session_state["quick_imported_game"] = game["id"]
    st.rerun()


def render_quick_analysis(store, game, clock):
    st.subheader("Quick analysis")
    st.write("Choose a short interval, analyze it, then inspect each attempt and its evidence.")
    st.caption("No possession, frame selection or box labels needed. Predictions never become human training labels.")
    with st.expander("Open a local video", expanded=game is None):
        with st.form("quick_import"):
            source = st.text_input("Local video path", key="quick_source")
            name = st.text_input("Video name (optional)", key="quick_name")
            offset = st.text_input("Original source offset in seconds (optional)", key="quick_offset",
                                   help="Leave blank for zero, or let provenance supply the offset.")
            provenance = st.text_input("Provenance JSON path (optional)", key="quick_provenance")
            submitted = st.form_submit_button("Open video", key="quick_open")
        if submitted:
            _import(store, source, name=name.strip() or None, offset=offset.strip() or None,
                    provenance_path=provenance.strip() or None)
        demo = Path(__file__).resolve().parents[1] / "artifacts/handoffs/quick-analysis-demo"
        if (demo / "clip.mp4").is_file() and (demo / "provenance.json").is_file():
            if st.button("Open demonstration clip", key="quick_open_demo"):
                _import(store, demo / "clip.mp4", name="Training demonstration · 12:30–13:20",
                        provenance_path=demo / "provenance.json")
    if game is None:
        st.info("Open a local video to choose a short interval. Detailed review remains available in Workspace.")
        return
    duration = float(game["duration"])
    offset = float(game.get("source_offset_seconds", 0))
    st.write(f"**{game['name']}** · original source {clock(offset)}–{clock(offset+duration)}")
    st.caption(f"File seconds 0–{duration:g} map to original source seconds {offset:g}–{offset+duration:g}.")
    if duration < 1:
        st.warning("Choose a video at least one second long.")
        return
    start = st.number_input("Start timestamp (original source seconds)", min_value=offset,
                            max_value=offset+duration-1, value=offset, step=1.,
                            key=f"quick_start_{game['id']}")
    length = st.number_input("Duration (seconds)", min_value=1., max_value=min(60., duration),
                             value=min(20., duration), step=1., key=f"quick_duration_{game['id']}")
    end = min(start + length, offset + duration)
    st.caption(f"Selected original interval {clock(start)}–{clock(end)} ({end-start:g}s). "
               f"Worker reads file seconds {start-offset:g}–{end-offset:g}.")
    if end < start+length:
        st.info("Duration is shortened at the end of this source.")
    setup = {}
    with st.expander("Optional basket and team hints"):
        st.caption("Leave blank to analyze now. Missing basket/team evidence can leave results unresolved.")
        use_basket = st.checkbox("Use a basket hint", key=f"quick_use_basket_{game['id']}")
        if use_basket:
            st.caption("Basket bounds as fractions of image width/height (0–1). Hint applies only to this camera view.")
            columns = st.columns(4)
            setup["rim_box"] = [col.number_input(label, min_value=0., max_value=1., value=value,
                                                 step=.01, key=f"quick_rim_{game['id']}_{i}")
                                for i, (col, label, value) in enumerate(zip(columns,
                                    ["Basket left", "Basket top", "Basket right", "Basket bottom"],
                                    [.7, .2, .8, .3]))]
        use_teams = st.checkbox("Use uniform color hints", key=f"quick_use_teams_{game['id']}")
        if use_teams:
            left, right = st.columns(2)
            setup["team_colors"] = [left.color_picker("Team A uniform", value="#ffffff", key="quick_team_a"),
                                    right.color_picker("Team B uniform", value="#163b78", key="quick_team_b")]
    with st.expander("Model diagnostics"):
        st.caption("Default COCO weights detect balls and people, but have no rim class. "
                   "Without dependable rim evidence, made/missed remains unknown. No training runs here.")
        weights = st.text_input("Detector weights path (optional)", key="quick_weights")
        bundle = st.text_input("Shot classifier bundle (optional)", key="quick_bundle")
    if st.button("Analyze interval", key="quick_analyze", type="primary"):
        try:
            clip = queue_analysis_clip(store, game["id"], start-offset, end-offset,
                                       weights=weights.strip() or None, model_bundle=bundle.strip() or None,
                                       setup=setup)
            st.session_state[f"quick_selected_{game['id']}"] = clip["id"]
            st.success("Analysis queued. Keep the worker running and refresh progress below.")
        except (ValueError, TypeError) as exc:
            st.error(f"Could not queue analysis: {exc}")
    st.caption("Run the worker in another terminal: uv run --extra vision playlab worker")
    st.button("Refresh analysis status", key="quick_refresh")
    clips = store.list("analysis_clip", game_id=game["id"])
    if not clips:
        st.info("No interval analyses yet. Choose a timestamp and select Analyze interval.")
        return
    lookup = {c["id"]: c for c in reversed(clips)}
    selected_key = f"quick_selected_{game['id']}"
    selected = st.selectbox("Analysis interval", list(lookup), key=selected_key,
                            format_func=lambda key: f"{clock(lookup[key]['original_start'])}–"
                            f"{clock(lookup[key]['original_end'])} · {key[:8]}")
    clip = get_analysis_clip(store, selected)
    job = clip.get("job") or {}
    st.write(f"**Status: {job.get('status', clip.get('status', 'unknown'))}**")
    st.progress(float(job.get("progress", 0)))
    if job.get("error"):
        st.error(job["error"])
    if job.get("status") in {"queued", "running"}:
        status = worker_status(store)
        if status["state"] == "stopped":
            st.warning("The analysis worker is stopped. This request stays queued until the worker is restarted in its separate terminal.")
        if st.button("Cancel analysis", key="quick_cancel"):
            store.cancel_job(job["id"])
            st.rerun()
    elif job.get("status") in {"failed", "cancelled"}:
        if st.button("Resume analysis", key="quick_resume"):
            try:
                store.resume_job(job["id"])
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    result = clip.get("result")
    render_detection_inspector(store, game, clip, result, fresh=bool(clip.get("fresh")))
    if not result:
        st.info("Results appear here when the worker finishes. No result is available yet.")
        return
    if not clip.get("fresh"):
        st.warning("This result is stale because its source, settings or model changed. Analyze the interval again.")
        return
    playback = clip.get("playback_path") or result.get("playback_path")
    if playback and Path(playback).is_file():
        st.video(str(playback))
        st.caption("Bounded interval preview; playback starts at zero. Result times use the original source timeline.")
    st.caption("Unverified predictions. Software checks and this demonstration do not measure game accuracy.")
    for warning in result.get("warnings", []):
        st.warning(warning)
    events = result.get("events", result.get("shots", []))
    if not events:
        st.info("Unknown / review required: no dependable shot attempts were found in this interval.")
        context_image = result.get("context_evidence_path")
        if context_image and Path(context_image).is_file():
            context_time = result.get("context_time")
            caption = "Interval context · no confirmed release"
            if context_time is not None:
                caption += f" · original source {clock(context_time)}"
            st.image(str(context_image), caption=caption, width="stretch")
    for index, event in enumerate(events, 1):
        with st.container(border=True):
            outcome = event.get("result", "unknown")
            release = event.get("release", event.get("time"))
            st.write(f"**Attempt {index} · {outcome}**")
            if release is not None:
                st.write(f"Release · original source {clock(release)}")
            reason = event.get("reason") or event.get("review_reason")
            if reason:
                st.write(reason)
            if outcome == "unknown":
                st.info("Review required: available evidence does not support a made/missed decision.")
            image = event.get("evidence_path")
            if image and Path(image).is_file():
                st.image(str(image), caption="Release evidence · inspect the highlighted players", width="stretch")
            defender = event.get("defender") or {}
            if defender.get("status") == "approximate":
                st.write(f"Closest visible opponent (approximate): track {defender.get('track_id', '?')}")
                if defender.get("reason"):
                    st.caption(defender["reason"])
            else:
                st.caption(f"Closest opponent unresolved: {defender.get('reason', 'shooter, team or release evidence missing')}")
