"""Short, source-verified review context, independent of coarse possessions."""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

from playlab.dataset import canonical_hash
from playlab.labeling import _safe_spans
from playlab.media import clip_video, source_fingerprint
from playlab.review import match_game
from playlab.store import utcnow

MAX_REVIEW_SECONDS = 120


def _bounds(game, start, end, split):
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (start, end)):
        raise ValueError("Review clip bounds must be finite numbers")
    if (split not in {"train", "val"} or not 0 <= start < end <= game["duration"]
            or end - start > MAX_REVIEW_SECONDS):
        raise ValueError("Review clip must be a train/val interval of at most 120 seconds")
    if not any(low <= start < end <= high and partition == split
               for low, high, partition in _safe_spans(game)):
        raise ValueError("Review clip violates current split guards")


def _policy(game):
    return canonical_hash(_safe_spans(game))


def _identity(record):
    return canonical_hash([record[k] for k in (
        "batch_hash", "item_id", "game_id", "start", "end", "split",
        "expected_source_fingerprint", "expected_source_size", "split_policy")])


def _validate_source(game, record):
    _bounds(game, record["start"], record["end"], record["split"])
    if (game.get("split_guard_seconds", 10) != 10 or _policy(game) != record["split_policy"]):
        raise ValueError("Review clip source partition policy changed")
    source = Path(game.get("source_path", ""))
    if (not source.is_file() or source.stat().st_size != record["expected_source_size"]
            or game.get("source_fingerprint") != record["expected_source_fingerprint"]
            or source_fingerprint(source) != record["expected_source_fingerprint"]):
        raise ValueError("Matching original video required; review clip source changed or missing")
    return source


def _request(store, batch, frame_id, start, end):
    if frame_id not in batch.frames:
        raise ValueError("Frame not in review batch")
    frame = batch.frames[frame_id]
    item = batch.items[frame["review_item_id"]]
    game = match_game(store, batch, item["game_id"])
    if (start is None) != (end is None):
        raise ValueError("Supply both review clip bounds")
    start, end = (item["start"], item["end"]) if start is None else (start, end)
    _bounds(game, start, end, item["split"])
    source = batch.manifest["sources"][item["game_id"]]
    record = {"batch_hash": batch.id, "item_id": item["id"], "game_id": game["id"],
              "start": float(start), "end": float(end), "split": item["split"],
              "expected_source_fingerprint": source["source_fingerprint"],
              "expected_source_size": source["source_size_bytes"], "split_policy": _policy(game)}
    _validate_source(game, record)
    record["id"] = record["clip_id"] = _identity(record)
    return record


def _destination(store, clip_id):
    path = store.root / "clips" / f"review-{clip_id}.mp4"
    partial = path.with_name(path.stem + ".partial.mp4")
    if (not path.resolve().is_relative_to(store.root.resolve()) or path.parent.is_symlink()
            or path.is_symlink() or partial.is_symlink()):
        raise ValueError("Unsafe review clip destination")
    return path


def _view(store, record):
    result = {k: v for k, v in record.items() if k != "path"}
    path = _destination(store, record["clip_id"])
    if record.get("status") == "ready" and record.get("path") == str(path) and path.is_file():
        result["path"] = str(path)
        return result
    job = next((j for j in store.jobs() if j["id"] == record.get("job_id")), None)
    result["status"] = job["status"] if job else "missing"
    if result["status"] == "done":
        result["status"] = "missing"
    return result


def get_review_clip(store, batch, frame_id, start=None, end=None):
    """Return only a validated bounded clip path; never expose the source video."""
    request = _request(store, batch, frame_id, start, end)
    record = store.get("review_clip", request["id"])
    return _view(store, record) if record else None


def queue_review_clip(store, batch, frame_id, start=None, end=None):
    request = _request(store, batch, frame_id, start, end)
    # The worker must never claim a request before its companion record exists,
    # and a fast completion must never be overwritten by a queued-state write.
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT payload,revision FROM records WHERE kind='review_clip' AND id=?",
                         (request["id"],)).fetchone()
        if row:
            current = _view(store, {**json.loads(row["payload"]), "revision": row["revision"]})
            if current["status"] in {"ready", "queued", "running"}:
                return current
        encoded = json.dumps(request, sort_keys=True, allow_nan=False)
        job = db.execute("SELECT id FROM jobs WHERE task='review_clip' AND payload=? "
                         "AND status IN ('queued','running')", (encoded,)).fetchone()
        job_id = job["id"] if job else uuid.uuid4().hex
        if not job:
            now = utcnow()
            db.execute("INSERT INTO jobs(id,task,payload,status,created_at,updated_at) "
                       "VALUES (?,'review_clip',?,'queued',?,?)", (job_id, encoded, now, now))
        record = {**request, "job_id": job_id, "status": "queued"}
        revision = row["revision"] + 1 if row else 1
        db.execute("INSERT INTO records VALUES ('review_clip',?,?,?) ON CONFLICT(kind,id) "
                   "DO UPDATE SET payload=excluded.payload,revision=excluded.revision",
                   (request["id"], json.dumps(record, allow_nan=False), revision))
        record["revision"] = revision
    return _view(store, record)


def perform_review_clip(store, payload, *, cancelled, progress):
    """Recheck queued bounds and source at execution; derive all paths locally."""
    record = store.get("review_clip", payload["clip_id"])
    if (not record or _identity(payload) != payload["clip_id"]
            or any(record.get(k) != value for k, value in payload.items())):
        raise ValueError("Review clip request changed")
    game = store.get("game", payload["game_id"])
    if not game:
        raise ValueError("Review clip game no longer exists")
    source = _validate_source(game, payload)
    destination = _destination(store, payload["clip_id"])
    progress(0.05)
    clip_video(source, destination, payload["start"], payload["end"], cancelled=cancelled)
    # A changed source/policy must not make the generated file available for playback.
    _validate_source(store.get("game", payload["game_id"]) or {}, payload)
    progress(0.95)
    return store.save("review_clip", {**record, "path": str(destination), "status": "ready"})
