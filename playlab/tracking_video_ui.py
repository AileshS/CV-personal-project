"""One short clip in, playable ball/basket tracking video out."""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path

import streamlit as st

from playlab.clip_tracking import ensure_tracking_worker, get_tracking_clip, queue_tracking_clip
from playlab.media import frame_at, source_fingerprint
from playlab.quick_analysis import _allow_interval
from playlab.quick_import import import_quick_source
from playlab.worker import worker_status


def save_video_upload(store, name, content):
    """Save a content-addressed upload in ignored local data, never over human media."""
    extension = Path(name).suffix.lower()
    if extension not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise ValueError("Choose an MP4, MOV, AVI, MKV, WebM or M4V video.")
    if not content:
        raise ValueError("The uploaded video is empty.")
    digest = hashlib.sha256(content).hexdigest()
    directory = store.root / "uploads"
    if directory.is_symlink() or not directory.resolve().is_relative_to(store.root.resolve()):
        raise ValueError("The upload directory is unavailable. Choose a local video instead.")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}{extension}"
    if destination.is_symlink():
        raise ValueError("The saved upload is unavailable. Choose a local video instead.")
    if destination.exists():
        with destination.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("The saved upload changed. Open the original video again.")
        return destination
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".upload", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def open_tracking_source(store, source, *, name=None):
    """Reuse existing source mappings instead of overwriting excerpt provenance."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise ValueError("That video file is unavailable on this machine.")
    games = store.list("game")
    for game in games:
        if Path(game["source_path"]).resolve() == source:
            return game
    fingerprint = source_fingerprint(source)
    for game in games:
        if game.get("source_fingerprint") == fingerprint:
            if not Path(game["source_path"]).is_file():
                raise ValueError("This video already exists, but its original file is unavailable. Restore that file first.")
            return game
    return import_quick_source(store, source, name=name)


def _select_source(store):
    games = store.list("game")
    lookup = {game["id"]: game for game in games}
    imported = st.session_state.pop("tracking_imported_video", None)
    if imported in lookup:
        st.session_state["tracking_video_choice"] = imported
    selected = st.selectbox("Video", list(lookup), key="tracking_video_choice",
                            format_func=lambda key: lookup[key].get("name", "Video")) if lookup else None
    with st.expander("Open a video" if not games else "Open another video", expanded=not games):
        upload = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
                                  key="tracking_upload")
        if upload is not None:
            identity = getattr(upload, "file_id", None) or (upload.name, upload.size)
            if st.session_state.get("tracking_loaded_upload") != identity:
                try:
                    source = save_video_upload(store, upload.name, upload.getbuffer())
                    game = open_tracking_source(store, source, name=Path(upload.name).stem)
                    st.session_state["tracking_loaded_upload"] = identity
                    st.session_state["tracking_imported_video"] = game["id"]
                    st.rerun()
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    st.error(f"Could not open video: {exc}")
        with st.form("tracking_local_video"):
            source = st.text_input("Or open a local video path", key="tracking_local_path")
            submitted = st.form_submit_button("Open video", key="tracking_open_local")
        if submitted:
            try:
                game = open_tracking_source(store, source)
                st.session_state["tracking_imported_video"] = game["id"]
                st.rerun()
            except (OSError, ValueError, TypeError, KeyError) as exc:
                st.error(f"Could not open video: {exc}")
    return lookup.get(selected)


def _start_worker(store):
    try:
        status = ensure_tracking_worker(store)
        st.session_state.pop("tracking_worker_error", None)
        return status
    except (OSError, ValueError, RuntimeError) as exc:
        st.session_state["tracking_worker_error"] = f"Could not start tracking: {exc}"
        return False


def _retry_stopped_worker(store, clip_id):
    """Restart one proven-dead worker and requeue only its selected interrupted job."""
    request = store.get("tracking_clip", clip_id)
    before = next((job for job in store.jobs() if job["id"] == request["job_id"]), None) if request else None
    started = _start_worker(store)
    if not started or not started.get("recovered") or not before or before["status"] != "running":
        return
    after = next((job for job in store.jobs() if job["id"] == before["id"]), None)
    if (after and after["status"] == "failed"
            and after.get("error") == "Worker interrupted; resume this job"):
        store.resume_job(after["id"])


def _result_panel(store, clip_id, clock, *, polling=False):
    clip = get_tracking_clip(store, clip_id) or {}
    job = clip.get("job") or {}
    status = job.get("status", clip.get("status", "queued"))
    if polling and status not in {"queued", "running"}:
        st.rerun()  # Replace the polling fragment once the job is terminal.
    if status in {"queued", "running"}:
        st.progress(float(job.get("progress", 0)), text="Preparing your tracked clip…")
        worker = worker_status(store)
        failure = st.session_state.get("tracking_worker_error")
        if not failure and worker["state"] == "stopped":
            failure = "The tracking worker stopped. Select Retry to continue."
        elif not failure and worker["state"] == "unknown":
            failure = "The tracking worker's status cannot be verified. Select Retry to check again."
        if failure:
            st.error(failure)
            if st.button("Retry", key=f"tracking_start_retry_{clip_id}"):
                try:
                    _retry_stopped_worker(store, clip_id)
                except (OSError, ValueError, KeyError) as exc:
                    st.session_state["tracking_worker_error"] = f"Could not retry tracking: {exc}"
                st.rerun()
        st.button("Refresh", key=f"tracking_refresh_{clip_id}")
        return
    if status in {"failed", "cancelled"}:
        error = job.get("error") or "The clip could not be completed."
        st.error(f"Tracking did not finish. {error}")
        if st.button("Retry", key=f"tracking_retry_{clip_id}"):
            try:
                store.resume_job(job["id"])
                _start_worker(store)
                st.rerun()
            except (OSError, ValueError, KeyError) as exc:
                st.error(f"Could not retry tracking: {exc}")
        return
    if not clip.get("fresh"):
        st.info("This clip needs a new analysis. Select Analyze to refresh it.")
        return
    path = clip.get("video_path")
    if not path or not Path(path).is_file():
        st.error("The tracked video is unavailable. Select Analyze to recreate it.")
        return
    st.success("Your tracked clip is ready.")
    st.video(str(path))
    start, end = clip.get("original_start"), clip.get("original_end")
    if start is not None and end is not None:
        st.caption(f"Tracked clip · {clock(start)}–{clock(end)}")
    st.caption("Box numbers show detector scores.")
    with Path(path).open("rb") as stream:
        st.download_button("Download tracked video", data=stream, file_name="basketball-tracked.mp4",
                            mime="video/mp4", key=f"tracking_download_{clip_id}")


@st.cache_data(max_entries=12, show_spinner=False)
def _starting_frame(path, start, modification_time):
    """Decode one requested frame, never load a full game into Streamlit playback."""
    return frame_at(path, start, max_width=1280)


def render_tracking_video(store, clock):
    st.write("Choose a video and a start time, then get a 10-second video with ball and basket boxes.")
    game = _select_source(store)
    if game is None:
        st.info("Upload or open a video to begin.")
        return
    duration = float(game["duration"])
    offset = float(game.get("source_offset_seconds", 0))
    if not all(math.isfinite(value) for value in (duration, offset)) or duration < .25:
        st.error("Choose a video at least a quarter of a second long.")
        return
    start = st.number_input("Start time (seconds)", min_value=offset, max_value=offset + duration - .25,
                            value=offset, step=1., format="%.2f", key=f"tracking_start_{game['id']}")
    file_start = start - offset
    end = min(file_start + 10, duration)
    st.caption(f"{clock(start)}–{clock(offset + end)} · {end - file_start:g} seconds")
    if end - file_start < 10:
        st.caption("The clip is shortened at the end of this video.")
    with st.expander("Preview selected clip"):
        if st.checkbox("Show starting frame", key=f"tracking_show_preview_{game['id']}"):
            path = Path(game["source_path"])
            try:
                _allow_interval(game, file_start, end)
                pixels = _starting_frame(str(path), file_start, path.stat().st_mtime_ns)
                st.image(pixels, caption=f"Clip starts at {clock(start)}", width="stretch")
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Could not preview this video: {exc}")
    selected_key = f"tracking_selected_{game['id']}"
    if st.button("Analyze", key="tracking_analyze", type="primary"):
        try:
            clip = queue_tracking_clip(store, game["id"], file_start, end)
            st.session_state[selected_key] = clip["id"]
            _start_worker(store)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            st.error(f"Could not analyze this clip: {exc}")
    clip_id = st.session_state.get(selected_key)
    if not clip_id:
        return
    request = store.get("tracking_clip", clip_id)
    if (not request or abs(float(request["start"]) - file_start) > .001
            or abs(float(request["end"]) - end) > .001):
        return
    clip = get_tracking_clip(store, clip_id) or {}
    job = clip.get("job") or {}
    pending = job.get("status", clip.get("status")) in {"queued", "running"}
    panel = st.fragment(run_every=2.0 if pending else None)(_result_panel)
    panel(store, clip_id, clock, polling=pending)
