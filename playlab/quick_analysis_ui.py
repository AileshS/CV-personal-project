"""Single-page Quick analysis: pick a clip, analyze it here, read the results.

Short clips run in this viewer process. A separate worker is optional; if one
claims the job first, this page shows its progress instead.
"""
from __future__ import annotations

import math
from pathlib import Path

import streamlit as st

from playlab.analysis import default_ball_rim_weights
from playlab.box_drawer import draw_frame
from playlab.detection_inspector_ui import render_detection_inspector
from playlab.media import frame_at
from playlab.quick_analysis import get_analysis_clip, queue_analysis_clip
from playlab.quick_import import import_quick_source
from playlab.worker import run_job_now, worker_status


@st.cache_data(max_entries=8, show_spinner=False)
def _still(path, seconds, modification_time):
    return frame_at(path, seconds, max_width=1280)


def _valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in box)
            and box[0] < box[2] and box[1] < box[3])


def _import(store, source, **metadata):
    try:
        game = import_quick_source(store, source, **metadata)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not open source: {exc}")
        return
    st.session_state["quick_imported_game"] = game["id"]
    st.rerun()


def _open_video(store, game, advanced):
    with st.expander("Open a video", expanded=game is None):
        with st.form("quick_import"):
            source = st.text_input("Local video path", key="quick_source")
            name = st.text_input("Video name (optional)", key="quick_name")
            offset = provenance = ""
            if advanced:
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


def _hints(game, start, offset, default_weights):
    setup = {}
    rim_key, drawer_key = f"quick_rim_box_{game['id']}", f"quick_rim_draw_{game['id']}"
    with st.expander("Basket and team hints (optional)", expanded=default_weights is None):
        source = Path(game["source_path"])

        def take_rim():
            state = st.session_state[drawer_key]
            drawn = state.get("box") if isinstance(state, dict) else getattr(state, "box", None)
            if _valid_box(drawn):
                st.session_state[rim_key] = [float(v) for v in drawn]

        if default_weights is None:
            st.caption("No ball/rim model is installed, so mark the basket: click two opposite corners "
                       "around the rim. It is followed through pans and cuts.")
        else:
            st.caption("The ball/rim model finds the basket. Mark it here only if results miss it.")
        if source.is_file():
            rim = st.session_state.get(rim_key)
            try:
                still = _still(str(source), start - offset, source.stat().st_mtime)
            except (OSError, ValueError, RuntimeError) as exc:
                st.warning(f"Could not read the starting frame: {exc}")
            else:
                draw_frame(still, [{"label": "Basket", "box": rim}] if rim else [],
                           identity=f"{game['id']}:{start:.3f}", key=drawer_key, on_box_change=take_rim)
        if rim := st.session_state.get(rim_key):
            setup["rim_box"] = rim
            if st.button("Clear basket", key=f"quick_rim_clear_{game['id']}"):
                st.session_state.pop(rim_key, None)
                st.rerun()
        if st.checkbox("Set uniform colors (otherwise detected automatically)",
                       key=f"quick_use_teams_{game['id']}"):
            left, right = st.columns(2)
            setup["team_colors"] = [left.color_picker("Team A uniform", value="#ffffff", key="quick_team_a"),
                                    right.color_picker("Team B uniform", value="#163b78", key="quick_team_b")]
    return setup


def _stage_text(checkpoint):
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    stage = checkpoint.get("stage")
    if stage == "preparing_playback":
        return "Preparing the playback clip"
    if stage == "detecting" and {"seconds_done", "seconds_total"} <= set(checkpoint):
        return (f"Finding players, ball and rim · {checkpoint['seconds_done']:.0f}s "
                f"of {checkpoint['seconds_total']:.0f}s")
    if stage == "checking_shots" and {"shot", "shots"} <= set(checkpoint):
        return f"Checking possible shot {checkpoint['shot']} of {checkpoint['shots']} frame by frame"
    return "Analyzing"


def _run_here(store, job_id):
    bar = st.progress(0., text="Starting… keep this page open.")

    def show(fraction, checkpoint=None):
        fraction = min(1., float(fraction))
        bar.progress(fraction, text=f"{_stage_text(checkpoint)} · {fraction:.0%}")

    ran = run_job_now(store, job_id, on_progress=show)
    bar.empty()
    return ran


def _job_controls(store, job):
    status = job.get("status")
    if job.get("error"):
        st.error(job["error"])
    if status in {"queued", "running"}:
        st.progress(float(job.get("progress", 0)),
                    text=f"{status.capitalize()} in the background worker · {_stage_text(job.get('checkpoint'))}")
        if status == "queued" and worker_status(store)["state"] == "stopped":
            if st.button("Run it here", key="quick_run_here", type="primary"):
                _run_here(store, job["id"])
                st.rerun()
        left, right = st.columns(2)
        left.button("Refresh", key="quick_refresh")
        if right.button("Cancel analysis", key="quick_cancel"):
            store.cancel_job(job["id"])
            st.rerun()
    elif status in {"failed", "cancelled"}:
        left, right = st.columns(2)
        if left.button("Resume analysis", key="quick_resume"):
            try:
                store.resume_job(job["id"])
            except ValueError as exc:
                st.error(str(exc))
            else:
                _run_here(store, job["id"])
                st.rerun()
        right.button("Refresh", key="quick_refresh")


def _results(store, game, clip, clock, advanced):
    job = clip.get("job") or {}
    _job_controls(store, job)
    result = clip.get("result")
    render_detection_inspector(store, game, clip, result, fresh=bool(clip.get("fresh")))
    if not result:
        st.info("No result is available yet.")
        return
    if not clip.get("fresh"):
        st.warning("This result is stale because its source, settings or model changed. Analyze the interval again.")
        return
    events = result.get("events", result.get("shots", []))
    columns = st.columns(3)
    columns[0].metric("Attempts", len(events))
    columns[1].metric("Made", sum(e.get("result") == "made" for e in events))
    columns[2].metric("Missed", sum(e.get("result") == "missed" for e in events))
    performance = result.get("performance", {})
    if performance.get("elapsed_seconds") is not None:
        elapsed = float(performance["elapsed_seconds"])
        rate = performance.get("evaluated_frames_per_second")
        inferred = performance.get("inferred_frames", result.get("frames_processed", 0))
        cached = performance.get("cached_scan_frames", 0)
        detail = f"{inferred} model calls this run"
        if cached:
            detail += f" · {cached} scan frames reused"
        if rate is not None:
            detail += f" · {float(rate):.1f} frames/s"
        runtime = (f"{performance['cpu_threads']} CPU threads" if performance.get("cpu_threads")
                   else str(performance.get("device", "model runtime")))
        st.caption(f"Model analysis: {elapsed:.1f}s · {detail} · {runtime}")
    playback = clip.get("playback_path") or result.get("playback_path")
    if playback and Path(playback).is_file():
        st.video(str(playback))
    if not events:
        st.info("Unknown / review required: no dependable shot attempts were found in this interval.")
        context_image = result.get("context_evidence_path")
        if context_image and Path(context_image).is_file():
            context_time = result.get("context_time")
            caption = "Interval context · no confirmed release"
            if context_time is not None:
                caption += f" · {clock(context_time)}"
            st.image(str(context_image), caption=caption, width="stretch")
    for index, event in enumerate(events, 1):
        with st.container(border=True):
            outcome = event.get("result", "unknown")
            release = event.get("release", event.get("time"))
            when = f" · {clock(release)}" if release is not None else ""
            st.write(f"**Attempt {index} · {outcome}**{when}")
            reason = event.get("reason") or event.get("review_reason")
            if reason:
                st.write(reason)
            image = event.get("evidence_path")
            if image and Path(image).is_file():
                st.image(str(image), caption="Release frame: shooter in green, closest opponent in orange",
                         width="stretch")
            defender = event.get("defender") or {}
            if defender.get("status") == "approximate":
                st.write(f"Closest visible opponent: track {defender.get('track_id', '?')} (approximate)")
            else:
                st.caption(f"Closest opponent unresolved: {defender.get('reason', 'shooter or team unclear')}")
    warnings = result.get("warnings", [])
    if warnings:
        with st.expander(f"Analysis notes ({len(warnings)})"):
            for warning in warnings:
                st.write(f"- {warning}")
            st.caption("Unverified predictions; inspect the evidence before relying on them.")


def render_quick_analysis(store, game, clock, advanced=False):
    st.write("Open a video, pick up to 60 seconds, analyze, and read each shot attempt.")
    _open_video(store, game, advanced)
    if game is None:
        st.info("Open a local video to choose a short interval.")
        return
    duration = float(game["duration"])
    offset = float(game.get("source_offset_seconds", 0))
    if duration < 1:
        st.warning("Choose a video at least one second long.")
        return
    st.write(f"**{game['name']}** · {clock(offset)}–{clock(offset+duration)}")
    left, right = st.columns(2)
    start = left.number_input("Start (seconds)", min_value=offset, max_value=offset+duration-1, value=offset,
                              step=1., key=f"quick_start_{game['id']}")
    length = right.number_input("Length (seconds)", min_value=1., max_value=min(60., duration),
                                value=min(20., duration), step=1., key=f"quick_duration_{game['id']}")
    end = min(start + length, offset + duration)
    st.caption(f"Analyzes {clock(start)}–{clock(end)} ({end-start:g}s).")
    st.caption(f"Fast CPU profile evaluates about {math.ceil((end-start) * 5)} frames in the initial scan, "
               "then checks only possible shot windows more closely.")
    default_weights = default_ball_rim_weights()
    setup = _hints(game, start, offset, default_weights)
    weights = bundle = ""
    if advanced:
        with st.expander("Model files"):
            st.caption(f"Ball/rim detector: {default_weights}" if default_weights else
                       "No ball/rim weights found. Place them at artifacts/models/ball_rim.pt "
                       "or set PLAYLAB_BALL_RIM_WEIGHTS.")
            weights = st.text_input("Detector weights path (optional)", key="quick_weights",
                                    placeholder=str(default_weights or ""))
            bundle = st.text_input("Shot classifier bundle (optional)", key="quick_bundle")
    selected_key = f"quick_selected_{game['id']}"
    if st.button("Analyze", key="quick_analyze", type="primary", width="stretch"):
        try:
            clip = queue_analysis_clip(store, game["id"], start-offset, end-offset,
                                       weights=weights.strip() or None, model_bundle=bundle.strip() or None,
                                       setup=setup)
        except (ValueError, TypeError) as exc:
            st.error(f"Could not start analysis: {exc}")
        else:
            st.session_state[selected_key] = clip["id"]
            if clip.get("status") == "queued":
                _run_here(store, clip["job_id"])
    clips = store.list("analysis_clip", game_id=game["id"])
    if not clips:
        return
    st.divider()
    lookup = {c["id"]: c for c in reversed(clips)}
    if st.session_state.get(selected_key) not in lookup:
        st.session_state.pop(selected_key, None)
    selected = st.selectbox("Results for", list(lookup), key=selected_key,
                            format_func=lambda key: f"{clock(lookup[key]['original_start'])}–"
                            f"{clock(lookup[key]['original_end'])}")
    _results(store, game, get_analysis_clip(store, selected), clock, advanced)
