"""Bounded inference requests, isolated from possessions and human gold.

Decode times are local to the imported file. Public event times retain original
elapsed source time; both are explicitly available on every returned attempt.
"""
from __future__ import annotations

import copy
import json
import math
import uuid
from pathlib import Path

from playlab import analysis
from playlab.dataset import canonical_hash, file_hash
from playlab.media import clip_video, source_fingerprint
from playlab.store import utcnow

DEFAULT_DURATION = 20.0
MAX_DURATION = 60.0


def _setup(value):
    value = value or {}
    if not isinstance(value, dict) or set(value) - {"rim_box", "rim", "team_colors"}:
        raise ValueError("Quick setup accepts only an optional rim box and team colors.")
    value = copy.deepcopy(value)
    box = value.get("rim_box") or value.get("rim")
    if box is not None:
        if len(box) != 4:
            raise ValueError("Rim box needs four normalized coordinates.")
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Rim box must be normalized [left, top, right, bottom].")
        value.pop("rim", None)
        value["rim_box"] = [x1, y1, x2, y2]
    colors = value.get("team_colors")
    if colors:
        from playlab.detectors import hex_rgb
        if not isinstance(colors, (dict, list)) or len(colors) != 2:
            raise ValueError("Provide both team colors or leave them unresolved.")
        for color in colors.values() if isinstance(colors, dict) else colors:
            hex_rgb(color)
    return value


def _game(store, game_id, *, start=None, end=None):
    game = store.get("game", game_id)
    if not game:
        raise ValueError("Source game not found.")
    if not Path(game["source_path"]).is_file():
        raise ValueError("Source video is missing. Restore its local file before analysis.")
    offset = float(game.get("source_offset_seconds", 0))
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("Source offset must be finite and nonnegative.")
    game = {**game, "source_offset_seconds": offset}
    if start is not None and end is not None:
        _allow_interval(game, start, end)
        game["quick_source_fingerprint"] = _source_identity(game)
    return game


def _allow_interval(game, start, end):
    if game.get("split") == "test" or any(
            block.get("split") == "test" and start < float(block["end"])
            and end > float(block["start"]) for block in game.get("split_blocks", [])):
        raise ValueError("This interval overlaps the declared locked test partition. Choose another interval.")


def _source_identity(game):
    # Head/tail hashing could read held-out bytes even for a train interval in
    # a shared file. Such sources use metadata only; inference reads the chosen
    # allowed interval. Stat identity detects ordinary edits, not adversarial ones.
    shared_test = any(block.get("split") == "test" for block in game.get("split_blocks", []))
    return canonical_hash({"declared_fingerprint": game.get("source_fingerprint"),
        "identity": analysis._file_version(game["source_path"]) if shared_test
                    else source_fingerprint(game["source_path"]),
        "method": "stat+declared_fingerprint" if shared_test else "size+head/tail+declared_fingerprint"})


def _request_matches(game, clip):
    default = Path(clip["default_model_path"])
    default_ok = (clip["default_model_snapshot"] == "missing" or
                  (default.is_file() and file_hash(default) == clip["default_model_snapshot"]))
    return (clip["request_fingerprint"] == _request_digest(clip) and default_ok
            and str(analysis._default_model_path()) == clip["default_model_path"]
            and analysis._file_version(game["source_path"]) == clip["source_snapshot"]
            and game["quick_source_fingerprint"] == clip["source_content_fingerprint"]
            and game["source_offset_seconds"] == clip["source_offset_seconds"]
            and game.get("source_provenance", game.get("provenance")) == clip["source_provenance"]
            and float(game["duration"]) == clip["source_duration"])


def _request_digest(clip):
    return canonical_hash({key: clip[key] for key in (
        "game_id", "start", "end", "setup", "device", "weights", "model_bundle",
        "source_snapshot", "source_content_fingerprint", "source_offset_seconds",
        "source_provenance", "source_duration", "requested_models",
        "default_model_path", "default_model_snapshot")})


class _IntervalStore:
    """Narrow adapter: synthetic interval exists only in memory, never in Store."""

    def __init__(self, store, clip):
        self.store, self.clip = store, clip
        self.root = store.root / "quick_analysis" / clip["id"]

    def get(self, kind, identifier):
        if kind == "possession":
            return {"id": self.clip["id"], "game_id": self.clip["game_id"],
                    "start": self.clip["start"], "end": self.clip["end"], "revision": 1}
        if kind == "game":
            game = _game(self.store, self.clip["game_id"], start=self.clip["start"], end=self.clip["end"])
            return {**game,
                    "quick_analysis_context": {"clip_id": self.clip["id"],
                        "source_fingerprint": game["quick_source_fingerprint"],
                        "code_hash": file_hash(__file__)}}
        if kind == "setup":
            current = self.store.get("analysis_clip", self.clip["id"]) or self.clip
            return copy.deepcopy(current.get("setup", {}))
        if kind in {"analysis", "analysis_chunk"}:
            return self.store.get("quick_" + kind, self._id(identifier))
        raise ValueError(f"Quick analysis cannot read {kind} records.")

    def list(self, kind, **filters):
        raise ValueError(f"Quick analysis cannot list {kind} records.")

    def _id(self, identifier):
        return f"{self.clip['id']}:{identifier}"

    def save(self, kind, record):
        if kind not in {"analysis", "analysis_chunk"}:
            raise ValueError(f"Quick analysis cannot write {kind} records.")
        record = copy.deepcopy(record)
        record["id"] = self._id(record["id"])
        record["clip_id"] = self.clip["id"]
        record.pop("possession_id", None)
        if kind == "analysis":
            record["local_start"], record["local_end"] = self.clip["start"], self.clip["end"]
            offset = record["dependencies"]["source_offset_seconds"]
            record["source_offset_seconds"] = offset
            for event in record["events"]:
                event.pop("possession_id", None)
                event["local_time"], event["local_release"] = event["time"], event["release"]
                event["original_time"] = event["time"] + offset
                event["original_release"] = event["release"] + offset
                if event["result"] == "unknown":
                    event["reason"] = ("Learned detector unavailable; color candidates cannot establish an outcome."
                        if event.get("method") == "color_diagnostic_only" else
                        "Insufficient continuous ball/rim trajectory or model confidence to establish made/missed.")
                else:
                    event["reason"] = "Provisional visual evidence; inspect the estimated release and trajectory."
                event["review_reason"] = event["reason"]
            record["shots"] = record["events"]
            record["warnings"] = [w for w in record["warnings"] if "Add a shot release anchor" not in w]
            if not record["events"]:
                record["warnings"].append("No dependable attempts found; outcome unknown. Inspect this clip or try another short interval.")
                game = self.get("game", self.clip["game_id"])
                at = (self.clip["start"] + self.clip["end"]) / 2
                record["context_evidence_path"] = analysis._evidence(
                    game["source_path"], at, [], None, {"status": "unknown"},
                    self.root / "context.png", source_offset_seconds=offset)
                record["context_time"] = at + offset
                record["local_context_time"] = at
            if not self.clip.get("setup", {}).get("rim_box") and not record["dependencies"].get("weights"):
                record["warnings"].append("Basket evidence unresolved: the pretrained COCO detector has no rim class. Optional basket setup or compatible ball/rim weights are needed for rim-based attempts.")
            record["outcome_status"] = "review_required"
        return self.store.save("quick_" + kind, record)


def queue_analysis_clip(store, game_id, start, end, *, device="cpu", weights=None,
                        model_bundle=None, setup=None):
    """Publish an immutable interval request and its queue job in one transaction."""
    game = _game(store, game_id)
    start, end = float(start), float(end)
    if not all(math.isfinite(t) for t in (start, end)) or not (
            0 <= start < end <= float(game["duration"]) and 1 <= end - start <= MAX_DURATION):
        raise ValueError("Choose 1–60 seconds within the imported source video.")
    _allow_interval(game, start, end)
    game["quick_source_fingerprint"] = _source_identity(game)
    setup = _setup(setup)
    weights = weights or analysis.default_ball_rim_weights()
    request = {"game_id": game_id, "start": start, "end": end, "setup": setup,
               "device": str(device), "weights": str(Path(weights).resolve()) if weights else None,
               "model_bundle": str(Path(model_bundle).resolve()) if model_bundle else None}
    clip_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex
    offset = game["source_offset_seconds"]
    now = utcnow()
    default_model = analysis._default_model_path()
    clip = {"id": clip_id, **request, "job_id": job_id, "created_at": now,
            "source_offset_seconds": offset, "original_start": start + offset,
            "source_duration": float(game["duration"]),
            "source_content_fingerprint": game["quick_source_fingerprint"],
            "original_end": end + offset, "source_snapshot": analysis._file_version(game["source_path"]),
            "source_provenance": game.get("source_provenance", game.get("provenance")),
            "requested_models": {path: file_hash(path) if Path(path).is_file() else "missing"
                                 for path in (request["weights"], request["model_bundle"]) if path},
            "default_model_path": str(default_model),
            "default_model_snapshot": file_hash(default_model) if default_model.is_file() else "missing",
            "status": "queued", "record_type": "unverified_analysis_interval"}
    clip["request_fingerprint"] = _request_digest(clip)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        # Deduplicate unfinished identical clicks, including across viewer sessions.
        rows = db.execute("SELECT payload FROM records WHERE kind='analysis_clip'").fetchall()
        for row in rows:
            old = json.loads(row["payload"])
            if old.get("request_fingerprint") == clip["request_fingerprint"]:
                job = db.execute("SELECT status FROM jobs WHERE id=?", (old["job_id"],)).fetchone()
                if job and job["status"] in {"queued", "running"}:
                    return {**old, "status": job["status"]}
        db.execute("INSERT INTO records(kind,id,payload,revision) VALUES ('analysis_clip',?,?,1)",
                   (clip_id, json.dumps(clip, allow_nan=False)))
        db.execute("INSERT INTO jobs(id,task,payload,status,created_at,updated_at) VALUES (?,?,?,'queued',?,?)",
                   (job_id, "analyze_clip", json.dumps({"clip_id": clip_id}, sort_keys=True), now, now))
    return {**clip, "revision": 1}


def get_analysis_result(store, clip_id):
    result = store.get("quick_analysis", f"{clip_id}:{clip_id}:automatic")
    if not result:
        return None
    result = copy.deepcopy(result)
    for event in result["events"]:
        event["time"], event["release"] = event["original_time"], event["original_release"]
    result["shots"] = result["events"]
    result["playback_path"] = str(store.root / "quick_analysis" / clip_id / "playback.mp4")
    return result


def analysis_clip_is_fresh(store, clip_or_result):
    if not clip_or_result:
        return False
    clip_id = clip_or_result.get("clip_id", clip_or_result.get("id"))
    clip = store.get("analysis_clip", clip_id)
    result = get_analysis_result(store, clip_id)
    if not clip or not result or result.get("stale") or result.get("status") != "complete":
        return False
    try:
        game = _game(store, clip["game_id"], start=clip["start"], end=clip["end"])
        _allow_interval(game, clip["start"], clip["end"])
        if not _request_matches(game, clip):
            return False
        saved = result["dependencies"]
        current = analysis.analysis_dependencies(_IntervalStore(store, clip), clip_id, mode="automatic",
            weights=clip.get("weights"), model_bundle=clip.get("model_bundle"), device=clip["device"])
        return saved == current and canonical_hash(saved)[:24] == result["fingerprint"]
    except (KeyError, ValueError, OSError):
        return False


def get_analysis_clip(store, clip_id):
    clip = store.get("analysis_clip", clip_id)
    if not clip:
        return None
    job = next((job for job in store.jobs() if job["id"] == clip["job_id"]), None)
    path = store.root / "quick_analysis" / clip_id / "playback.mp4"
    return {**clip, "job": job, "status": job["status"] if job else "missing_job",
            "result": get_analysis_result(store, clip_id), "fresh": analysis_clip_is_fresh(store, clip),
            "playback_path": str(path) if path.is_file() else None}


def perform_analysis_clip(store, payload, *, cancelled=None, progress=None):
    clip = store.get("analysis_clip", payload["clip_id"])
    if not clip:
        raise ValueError("Analysis interval no longer exists.")
    if cancelled and cancelled():
        raise InterruptedError("Analysis cancelled; completed stages retained.")
    adapter = _IntervalStore(store, clip)
    game = _game(store, clip["game_id"], start=clip["start"], end=clip["end"])
    _allow_interval(game, clip["start"], clip["end"])
    if not _request_matches(game, clip):
        raise ValueError("Analysis request, source, model or original timestamp provenance changed. Queue a new interval request.")
    # Validate explicit model requests even if no attempts are discovered.
    for label, path in (("Ball/rim weights", clip.get("weights")), ("Outcome classifier", clip.get("model_bundle"))):
        if path and not Path(path).is_file():
            raise ValueError(f"{label} file is missing: {path}")
        if path and file_hash(path) != clip["requested_models"][path]:
            raise ValueError(f"{label} changed after this request. Queue a new interval request.")
    path = adapter.root / "playback.mp4"
    preview_state = adapter.root / "playback-source.json"
    source_state = {"source": analysis._file_version(game["source_path"]), "start": clip["start"], "end": clip["end"]}
    if not path.is_file() or not preview_state.is_file() or json.loads(preview_state.read_text()) != source_state:
        if progress:
            progress(0., {"stage": "preparing_playback", "clip_id": clip["id"]})
        clip_video(game["source_path"], path, clip["start"], clip["end"], cancelled=cancelled)
        preview_state.write_text(json.dumps(source_state))
    if progress:
        progress(.1, {"stage": "playback_ready", "clip_id": clip["id"]})

    def report(fraction, checkpoint=None):
        if progress:
            progress(.1 + .9 * fraction, {"clip_id": clip["id"], **(checkpoint or {})})

    analysis.analyze_possession(adapter, clip["id"], device=clip["device"], weights=clip.get("weights"),
        model_bundle=clip.get("model_bundle"), mode="automatic", progress=report, cancelled=cancelled)
    if not _request_matches(_game(store, clip["game_id"], start=clip["start"], end=clip["end"]), clip):
        raise ValueError("Source or original timestamp provenance changed during inference. Queue a new interval request.")
    return get_analysis_result(store, clip["id"])
