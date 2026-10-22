"""Frame controls for raw detector evidence, separate from outcome claims."""
from __future__ import annotations

import math
import threading

import streamlit as st

from playlab import analysis
from playlab.detection_inspector import inspect_frame, overlay, sampled_detections, validate_frame
from playlab.detectors import BasketballDetector
from playlab.media import frame_at


@st.cache_data(max_entries=32, show_spinner=False)
def _native_frame(path, at, version):
    return frame_at(path, at)


@st.cache_resource(max_entries=2)
def _diagnostic_detector(weights, device, model_version, default_version):
    if not analysis._default_model_path().is_file():
        raise ValueError("Restore the existing local detector weights before inspection.")
    return BasketballDetector(weights, device), threading.RLock()


@st.cache_data(max_entries=1200, show_spinner=False)
def _predict_frame(game, clip, at, source_version, model_version, default_version):
    detector, lock = _diagnostic_detector(clip.get("weights"), clip.get("device", "cpu"),
                                         model_version, default_version)
    with lock:
        return inspect_frame(game, clip, at, detector)


def render_detection_inspector(store, game, clip, result, *, fresh):
    st.subheader("Frame-by-frame model boxes")
    st.caption("See every raw box returned by every model inference call in the interval analysis. "
               "Yellow = ball; cyan = rim; orange = player; "
               "magenta = optional basket hint. Detector scores are not confidence intervals or calibrated probabilities.")
    if "start" not in clip or "end" not in clip:
        return
    key = f"inspect_{clip['id']}"
    if not st.checkbox("Show frame-by-frame detections", value=True, key=key+"_show"):
        return
    rows = sampled_detections(store, clip, result) if fresh else []
    mode = st.segmented_control("Inspection source", ["Any frame", "Saved model samples"],
                                default="Saved model samples" if rows else "Any frame", key=key+"_mode")
    row = None
    if mode == "Saved model samples":
        if not rows:
            st.info("No raw model samples are saved for this result. Use Any frame for an automatic diagnostic. "
                    "New interval analyses save raw samples even when the shot result is unknown.")
            return
        sample_key = key+"_sample"
        st.session_state.setdefault(sample_key, 0)
        left, right = st.columns(2)
        if left.button("Previous evaluated frame", key=key+"_sample_prev"):
            st.session_state[sample_key] = max(0, st.session_state[sample_key]-1)
        if right.button("Next evaluated frame", key=key+"_sample_next"):
            st.session_state[sample_key] = min(len(rows)-1, st.session_state[sample_key]+1)
        index = st.slider("Evaluated frame", 0, max(1, len(rows)-1), key=sample_key)
        row = rows[min(index, len(rows)-1)]
        at = float(row["time"])
        stage = row.get("stage", "scan").replace("_", " ")
        scope = row.get("inference_scope", "full_frame").replace("_", " ")
        st.caption(f"Inference call {min(index, len(rows)-1)+1} of {len(rows)} · {stage} · {scope}. "
                   "The initial scan evaluates about 5 frames per second; dense shot checks and release-player calls "
                   "are included separately, so the same timestamp can appear more than once.")
    else:
        st.caption("Each selected frame automatically shows the current model's best guess, even at very low confidence. "
                   "This is a new one-frame diagnostic, not a replay of the original analysis. "
                   "It does not establish a shot outcome or save training labels.")
        # Frame stepping is nominal for variable-frame-rate sources; timestamp seeking is authoritative.
        fps = float(game.get("fps") or 30)
        count = max(1, math.ceil((clip["end"]-clip["start"])*fps))
        frame_key = key+"_frame"
        st.session_state.setdefault(frame_key, 0)
        left, right = st.columns(2)
        if left.button("Previous frame", key=key+"_prev"):
            st.session_state[frame_key] = max(0, st.session_state[frame_key]-1)
        if right.button("Next frame", key=key+"_next"):
            st.session_state[frame_key] = min(count-1, st.session_state[frame_key]+1)
        index = st.slider("Frame position", 0, max(1, count-1), key=frame_key)
        at = min(float(clip["end"])-1e-6, float(clip["start"])+min(index, count-1)/fps)
        st.caption(f"Step size 1/{fps:g} second. Variable-frame-rate footage uses approximate frame steps. "
                   "Guesses are cached as you browse; the full game is not processed.")
        row = next((r for r in rows if abs(r["time"]-at) < 1e-6), None)
    try:
        validate_frame(game, clip, at)
        native = _native_frame(game["source_path"], at, analysis._file_version(game["source_path"]))
        st.caption(f"File timestamp {at:.3f}s · original source {at+float(game.get('source_offset_seconds', 0)):.3f}s "
                   f"· native {native.shape[1]} × {native.shape[0]}")
        if mode != "Saved model samples":
            weights = clip.get("weights")
            identity = (at, analysis._file_version(game["source_path"]),
                        analysis._file_version(weights) if weights else None,
                        analysis._file_version(analysis._default_model_path()))
            if st.button("Prepare every frame (up to 10 seconds)", key=key+"_prepare"):
                seconds = min(10., clip["end"]-clip["start"])
                total = min(1200, math.ceil(seconds*fps))
                progress = st.progress(0., text="Preparing weak model guesses for frame scrubbing…")
                for frame_index in range(total):
                    frame_at_time = clip["start"] + frame_index/fps
                    _predict_frame(game, clip, frame_at_time, *identity[1:])
                    progress.progress((frame_index+1)/total,
                                      text=f"Prepared {frame_index+1}/{total} frames")
                st.success(f"Prepared the first {seconds:g} seconds for this session. Scrub below without rerunning those guesses.")
            st.button("Detect this frame", key=key+"_detect", help="Predictions now run automatically as you scrub.")
            cached = st.session_state.get(key+"_prediction")
            if not cached or cached[0] != identity:
                with st.spinner("Running detector on this frame only…"):
                    predicted = _predict_frame(game, clip, at, *identity[1:])
                    st.session_state[key+"_prediction"] = (identity, predicted)
            cached = st.session_state.get(key+"_prediction")
            if cached and cached[0] == identity:
                row = cached[1]
        if row is None:
            st.image(native, alt="Native source frame without model predictions", width="stretch")
            st.info("This frame has no saved predictions. Select Detect this frame to inspect the model here.")
            return
        image, table = overlay(native, row, clip.get("setup", {}).get("rim_box"))
        st.image(image, alt="Native frame with raw ball, rim and player detector candidates", width="stretch")
        detector_mode = row.get("detector_mode", "unknown")
        st.caption(f"Detector: {detector_mode}. Raw candidates include false positives and candidates rejected by shot logic. "
                   f"Stage: {row.get('stage', 'unknown').replace('_', ' ')}.")
        raw = row.get("raw_detections", {})
        if not raw.get("balls"):
            st.info("Ball: no candidate returned, even at the diagnostic floor of 0.0001%. No location is invented."
                    if row.get("diagnostic") else
                    "Ball: no candidate returned at this analysis call's reporting threshold. No location is invented.")
        if not raw.get("rims"):
            if detector_mode == "pretrained_coco_ball":
                st.info("Rim: unsupported by the current COCO model (no rim class). This is not a low-confidence rim prediction.")
            else:
                st.info("Rim: no detection reported in this frame.")
        if detector_mode.startswith("color_"):
            st.warning("Color candidates are heuristic diagnostics. Their scores are not learned detector confidence.")
        if table:
            st.dataframe(table, hide_index=True, column_config={
                "Detector score": st.column_config.NumberColumn(format="%.8f")})
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        st.error(f"Could not inspect this frame: {exc}")
