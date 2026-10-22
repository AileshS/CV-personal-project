"""Frame controls for raw detector evidence, separate from outcome claims."""
from __future__ import annotations

import math

import streamlit as st

from playlab import analysis
from playlab.detection_inspector import inspect_frame, overlay, sampled_detections, validate_frame
from playlab.media import frame_at


@st.cache_data(max_entries=32, show_spinner=False)
def _native_frame(path, at, version):
    return frame_at(path, at)


@st.cache_data(max_entries=128, show_spinner=False)
def _predict_frame(game, clip, at, source_version, model_version, default_version):
    return inspect_frame(game, clip, at)


def render_detection_inspector(store, game, clip, result, *, fresh):
    st.subheader("Frame inspector")
    st.caption("Inspect any frame without labeling or rerunning the interval. Yellow = ball; cyan = rim; "
               "magenta = optional basket hint. Detector scores are not confidence intervals or calibrated probabilities.")
    if "start" not in clip or "end" not in clip:
        return
    key = f"inspect_{clip['id']}"
    if not st.checkbox("Show frame inspector", key=key+"_show"):
        return
    st.caption("Detect this frame runs the current local detector as a new diagnostic, not a replay of the original analysis. "
               "It does not establish a shot outcome or save training labels.")
    rows = sampled_detections(store, clip, result) if fresh else []
    mode = st.segmented_control("Inspection source", ["Any frame", "Saved model samples"],
                                default="Any frame", key=key+"_mode")
    row = None
    if mode == "Saved model samples":
        if not rows:
            st.info("No raw model samples are saved for this result. Use Any frame and Detect this frame. "
                    "New interval analyses save raw samples even when the shot result is unknown.")
            return
        index = st.slider("Model sample", 0, max(1, len(rows)-1), 0, key=key+"_sample")
        row = rows[min(index, len(rows)-1)]
        at = float(row["time"])
        st.caption("Saved analysis samples are approximately 10 per second, not predictions for every native frame.")
    else:
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
                   "Only a selected frame is decoded; the full game is not processed.")
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
            if st.button("Detect this frame", key=key+"_detect", type="primary"):
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
        st.image(image, alt="Native frame with raw ball and rim detector candidates", width="stretch")
        detector_mode = row.get("detector_mode", "unknown")
        st.caption(f"Detector: {detector_mode}. Raw candidates include false positives and candidates rejected by shot logic.")
        raw = row.get("raw_detections", {})
        if not raw.get("balls"):
            st.info("Ball: no detection above the model's reporting threshold in this frame.")
        if not raw.get("rims"):
            if detector_mode == "pretrained_coco_ball":
                st.info("Rim: unsupported by the current COCO model (no rim class). This is not a low-confidence rim prediction.")
            else:
                st.info("Rim: no detection reported in this frame.")
        if detector_mode.startswith("color_"):
            st.warning("Color candidates are heuristic diagnostics. Their scores are not learned detector confidence.")
        if table:
            st.dataframe(table, hide_index=True, column_config={
                "Detector score": st.column_config.NumberColumn(format="percent")})
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        st.error(f"Could not inspect this frame: {exc}")
