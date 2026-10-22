"""Short annotated tracking clips, separate from analysis and human labels."""
from __future__ import annotations

import copy
import json
import math
import shutil
import uuid
from pathlib import Path

from playlab.analysis import _file_version
from playlab.dataset import canonical_hash, file_hash
from playlab.quick_analysis import _allow_interval, _game, _source_identity
from playlab.store import utcnow
from playlab.worker import ensure_tracking_worker

__all__ = ["queue_tracking_clip", "get_tracking_clip", "perform_tracking_clip", "ensure_tracking_worker"]

DEFAULT_DURATION = MAX_DURATION = 10.0
MIN_DURATION = .25


def _code_identity():
    directory = Path(__file__).parent
    return {name: file_hash(directory / name) for name in (
        "clip_tracking.py", "tracked_video.py", "model_assets.py", "quick_analysis.py",
        "media.py", "worker.py", "detectors.py")}


def _settings():
    from playlab.tracked_video import TRACKING_SETTINGS
    return copy.deepcopy(TRACKING_SETTINGS)


def _model(weights):
    from playlab.model_assets import DEFAULT_MODEL_SHA256, default_ball_rim_path
    path = Path(weights).resolve() if weights else default_ball_rim_path().resolve()
    if not path.is_file() and weights:
        raise ValueError("Selected ball/basket model is missing. Restore that file or choose the default model.")
    digest = file_hash(path) if path.is_file() else DEFAULT_MODEL_SHA256
    if not weights and digest != DEFAULT_MODEL_SHA256:
        raise ValueError("Default ball/basket model checksum does not match. Restore the pinned model.")
    return {"path": str(path), "sha256": digest, "default": not bool(weights)}


def _digest(clip):
    return canonical_hash({key: clip[key] for key in (
        "game_id", "start", "end", "device", "model", "settings", "code_identity",
        "source_snapshot", "source_content_fingerprint", "source_offset_seconds",
        "source_provenance", "source_duration")})


def _request_current(store, clip, *, missing_default_ok=False):
    """Check test policy before reading source bytes or returning retained output."""
    game = _game(store, clip["game_id"], start=clip["start"], end=clip["end"])
    if (clip["request_fingerprint"] != _digest(clip)
            or clip["code_identity"] != _code_identity() or clip["settings"] != _settings()
            or clip["source_snapshot"] != _file_version(game["source_path"])
            or clip["source_content_fingerprint"] != game["quick_source_fingerprint"]
            or clip["source_offset_seconds"] != game["source_offset_seconds"]
            or clip["source_provenance"] != game.get("source_provenance", game.get("provenance"))
            or clip["source_duration"] != float(game["duration"])):
        raise ValueError("Tracking request, source or processing code changed. Analyze a new interval.")
    model = clip["model"]
    path = Path(model["path"])
    if model["default"]:
        from playlab.model_assets import DEFAULT_MODEL_SHA256, default_ball_rim_path
        if path != default_ball_rim_path().resolve() or model["sha256"] != DEFAULT_MODEL_SHA256:
            raise ValueError("Default tracking model changed. Analyze a new interval.")
    if not path.is_file():
        if not (missing_default_ok and model["default"]):
            raise ValueError("Tracking model is missing. Restore its original file or analyze a new interval.")
    elif file_hash(path) != model["sha256"]:
        raise ValueError("Tracking model changed. Restore its original file or analyze a new interval.")
    return game


def _root(store, clip_id):
    if len(clip_id) != 32 or any(letter not in "0123456789abcdef" for letter in clip_id):
        raise ValueError("Invalid tracking clip identity.")
    base = store.root / "tracking_video"
    path = base / clip_id
    if (base.is_symlink() or path.is_symlink()
            or not path.resolve().is_relative_to(store.root.resolve())):
        raise ValueError("Unsafe tracking output directory.")
    return path


def _output_hashes(store, clip_id, result):
    root = _root(store, clip_id)
    hashes = {}
    for key in ("video_path", "samples_path", "report_path"):
        path = Path(result[key])
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("Tracking output is missing or outside its clip directory.")
        hashes[key] = file_hash(path)
    return hashes


def queue_tracking_clip(store, game_id, start, end, *, device="cpu", weights=None):
    """Publish one immutable request and its job atomically; no model loads here."""
    game = _game(store, game_id)
    if any(isinstance(value, bool) for value in (start, end)):
        raise ValueError("Choose a finite tracking interval inside the video.")
    start, end = float(start), float(end)
    if (not all(math.isfinite(value) for value in (start, end))
            or not 0 <= start < end <= float(game["duration"])
            or not MIN_DURATION <= end - start <= MAX_DURATION):
        raise ValueError("Choose 0.25–10 seconds within the source video.")
    _allow_interval(game, start, end)
    source_identity = _source_identity(game)
    now = utcnow()
    clip = {"id": uuid.uuid4().hex, "job_id": uuid.uuid4().hex, "game_id": game_id,
            "start": start, "end": end, "device": str(device), "model": _model(weights),
            "settings": _settings(), "code_identity": _code_identity(),
            "source_snapshot": _file_version(game["source_path"]),
            "source_content_fingerprint": source_identity,
            "source_offset_seconds": game["source_offset_seconds"],
            "source_provenance": game.get("source_provenance", game.get("provenance")),
            "source_duration": float(game["duration"]),
            "original_start": start + game["source_offset_seconds"],
            "original_end": end + game["source_offset_seconds"], "created_at": now,
            "status": "queued", "record_type": "unverified_tracking_interval"}
    clip["request_fingerprint"] = _digest(clip)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        for row in db.execute("SELECT payload FROM records WHERE kind='tracking_clip'").fetchall():
            old = json.loads(row["payload"])
            if old.get("request_fingerprint") == clip["request_fingerprint"]:
                job = db.execute("SELECT status FROM jobs WHERE id=?", (old["job_id"],)).fetchone()
                if job and job["status"] in {"queued", "running"}:
                    return {**old, "status": job["status"]}
        db.execute("INSERT INTO records VALUES ('tracking_clip',?,?,1)",
                   (clip["id"], json.dumps(clip, allow_nan=False)))
        db.execute("INSERT INTO jobs(id,task,payload,status,created_at,updated_at) "
                   "VALUES (?,'track_clip',?,'queued',?,?)",
                   (clip["job_id"], json.dumps({"clip_id": clip["id"]}), now, now))
    return {**clip, "revision": 1}


def get_tracking_clip(store, clip_id):
    clip = store.get("tracking_clip", clip_id)
    if not clip:
        return None
    job = next((item for item in store.jobs() if item["id"] == clip["job_id"]), None)
    result = store.get("tracking_result", clip_id)
    fresh = False
    if result and result.get("status") == "complete":
        try:
            _request_current(store, clip)
            fresh = (result["request_fingerprint"] == clip["request_fingerprint"]
                     and result["model_sha256"] == clip["model"]["sha256"]
                     and result["output_hashes"] == _output_hashes(store, clip_id, result))
        except (OSError, ValueError, KeyError, TypeError):
            fresh = False
    status = job["status"] if job else "missing_job"
    if status == "done" and not fresh:
        status = "stale"
    return {**clip, "job": job, "status": status, "result": result, "fresh": fresh,
            "video_path": result["video_path"] if fresh else None}


def perform_tracking_clip(store, payload, *, cancelled=None, progress=None):
    """Render a fresh attempt; incomplete attempts never become playable results."""
    clip = store.get("tracking_clip", payload["clip_id"])
    if not clip:
        raise ValueError("Tracking interval no longer exists.")

    def check_cancelled():
        if cancelled and cancelled():
            raise InterruptedError("Tracking cancelled; retry starts this short clip cleanly.")

    check_cancelled()
    game = _request_current(store, clip, missing_default_ok=True)
    if clip["model"]["default"]:
        from playlab.model_assets import ensure_ball_rim_model
        ensure_ball_rim_model()
    _request_current(store, clip)
    existing = get_tracking_clip(store, clip["id"])
    if existing["fresh"]:
        return existing["result"]
    root = _root(store, clip["id"])
    root.mkdir(parents=True, exist_ok=True)
    destination = root / uuid.uuid4().hex
    from playlab.tracked_video import render_tracked_video
    try:
        rendered = render_tracked_video(game["source_path"], destination,
            start=clip["start"], end=clip["end"], weights=clip["model"]["path"],
            device=clip["device"], source_offset_seconds=clip["source_offset_seconds"],
            cancelled=cancelled, progress=progress)
        check_cancelled()
        _request_current(store, clip)
        if rendered["model_sha256"] != clip["model"]["sha256"]:
            raise ValueError("Renderer model identity changed. Analyze a new interval.")
        result = {**rendered, "id": clip["id"], "clip_id": clip["id"], "status": "complete",
                  "request_fingerprint": clip["request_fingerprint"], "completed_at": utcnow(),
                  "reviewed": False, "record_type": "unverified_tracking_output"}
        result["output_hashes"] = _output_hashes(store, clip["id"], result)
        return store.save("tracking_result", result)
    except BaseException:
        # Only this freshly chosen attempt is removed; older output is retained.
        if destination.exists() and not destination.is_symlink():
            shutil.rmtree(destination)
        raise
