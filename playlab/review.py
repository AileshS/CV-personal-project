"""Untrusted review packages and durable human boundary decisions.

Suggestions remain in the package; none of these operations saves gold labels.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path, PurePosixPath

from PIL import Image

from playlab.attention import request_attention
from playlab.dataset import canonical_hash, file_hash
from playlab.labeling import _safe_spans
from playlab.media import source_fingerprint


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid SHA-256 hash")


def _path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute() or ":" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("Unsafe batch path")
    result = root / relative
    if not result.resolve().is_relative_to(root):
        raise ValueError("Batch path escapes root (possibly via symlink)")
    return result


def _unique(rows):
    result = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Invalid record ID")
        if identifier in result:
            raise ValueError("Duplicate record ID")
        result[identifier] = row
    return result


class ReviewBatch:
    def __init__(self, root, manifest):
        self.root, self.manifest = root, manifest
        self.items = _unique(manifest["review_items"])
        self.frames = _unique(manifest["frames"])

    @property
    def id(self):
        return self.manifest["batch_hash"]

    def media(self, frame_id):
        """Check selected file hashes before exposing paths to a media reader."""
        frame = self.frames[frame_id]
        result = {}
        for key, relative in (("native_path", frame["image"]),
                              ("preview_path", f"images/{frame_id}-preview.jpg")):
            path = _path(self.root, relative)
            expected = self.manifest["files"].get(relative)
            if not path.is_file():
                result[key] = None
                continue
            if expected is None or file_hash(path) != expected:
                raise ValueError("Review image hash mismatch")
            if key == "native_path":
                with Image.open(path) as image:
                    if image.format != "PNG" or image.size != (frame["width"], frame["height"]):
                        raise ValueError("Native frame dimensions or format differ from manifest")
            result[key] = path
        return result


def load_batch(path):
    path = Path(path).expanduser()
    path = path / "manifest.json" if path.is_dir() else path
    root = path.parent.resolve()
    if path.is_symlink():
        raise ValueError("Manifest must not be a symlink")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"),
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        if (manifest.get("schema") != 1 or manifest.get("kind") != "unreviewed_label_batch"
                or manifest.get("reviewed") is not False):
            raise ValueError("Expected schema 1 unreviewed label batch")
        _digest(manifest.get("batch_hash"))
        if canonical_hash({k: v for k, v in manifest.items() if k != "batch_hash"}) != manifest["batch_hash"]:
            raise ValueError("Canonical batch hash mismatch")
        batch = ReviewBatch(root, manifest)
        if (manifest["frame_count"] != len(batch.frames)
                or manifest["review_item_count"] != len(batch.items) or not batch.frames or not batch.items):
            raise ValueError("Batch counts differ")
        for relative, digest in manifest["files"].items():
            _path(root, relative)
            _digest(digest)
        for source in manifest["sources"].values():
            _digest(source["source_fingerprint"])
            if any(type(source[key]) is not int or source[key] <= 0 for key in ("width", "height")):
                raise ValueError("Source requires positive native width and height")
            _safe_spans(source)  # Includes policy metadata for locked test, without opening it.
        for item in batch.items.values():
            if item.get("reviewed") is not False or item["split"] not in {"train", "val"}:
                raise ValueError("Locked test/reviewed items cannot be opened")
            source = manifest["sources"][item["game_id"]]
            start, end = _number(item["start"]), _number(item["end"])
            if not any(low <= start < end <= high and split == item["split"]
                       for low, high, split in _safe_spans(source)):
                raise ValueError("Review item violates source split guards")
            if not isinstance(item["possession_id"], str) or not item["possession_id"]:
                raise ValueError("Missing parent reference")
        seen = set()
        for frame in batch.frames.values():
            item = batch.items[frame["review_item_id"]]
            if (frame.get("reviewed") is not False or frame.get("negative") is not False
                    or any(frame[k] != item[k] for k in ("game_id", "possession_id", "split"))
                    or not item["start"] <= _number(frame["time"]) <= item["end"]):
                raise ValueError("Frame reference/time/split mismatch")
            key = (frame["game_id"], frame["time"])
            if key in seen:
                raise ValueError("Duplicate source frame")
            seen.add(key)
            if frame["image"] != f"images/{frame['id']}.png" or frame["image"] not in manifest["files"]:
                raise ValueError("Native frame reference missing or invalid")
            for dimension in ("width", "height"):
                if type(frame[dimension]) is not int or frame[dimension] <= 0:
                    raise ValueError("Invalid native dimensions")
            for box in frame["proposed_boxes"]:
                coords = [_number(value) for value in box["box"]]
                if (len(coords) != 4 or not 0 <= coords[0] < coords[2] <= 1
                        or not 0 <= coords[1] < coords[3] <= 1 or box["label"] not in {"ball", "rim"}
                        or box.get("reviewed") is not False or not 0 <= _number(box["confidence"]) <= 1):
                    raise ValueError("Invalid unreviewed box proposal")
        if any(not any(f["review_item_id"] == item_id for f in batch.frames.values())
               for item_id in batch.items):
            raise ValueError("Review item has no frames")
        for frame in batch.frames.values():
            source = manifest["sources"][frame["game_id"]]
            if any(source[key] != frame[key] for key in ("width", "height")):
                raise ValueError("Frame dimensions differ from native source")
        return batch
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Malformed review manifest") from exc


def register_batch(store, path):
    batch = load_batch(path)
    old = store.get("review_batch", batch.id)
    if old and old["path"] == str(batch.root / "manifest.json"):
        return old
    return store.save("review_batch", {**(old or {}), "id": batch.id,
                      "path": str(batch.root / "manifest.json"),
                      "current_frame_id": (old or {}).get("current_frame_id", next(iter(batch.frames)))},
                      expected_revision=old["revision"] if old else 0)


def save_navigation(store, batch, frame_id, *, expected_revision):
    if frame_id not in batch.frames:
        raise ValueError("Frame not in batch")
    old = store.get("review_batch", batch.id)
    if not old:
        raise ValueError("Register batch first")
    return store.save("review_batch", {**old, "current_frame_id": frame_id}, expected_revision=expected_revision)


def item_review(store, batch, item_id):
    if item_id not in batch.items:
        raise ValueError("Item not in batch")
    identifier = canonical_hash([batch.id, item_id])
    return store.get("review_item", identifier) or {"id": identifier, "revision": 0,
        "batch_hash": batch.id, "item_id": item_id, "notes": "", "uncertain": False,
        "offensive_rebound": False}


def frame_review(store, batch, frame_id):
    if frame_id not in batch.frames:
        raise ValueError("Frame not in batch")
    identifier = canonical_hash([batch.id, "frame", frame_id])
    return store.get("review_frame", identifier) or {"id": identifier, "revision": 0,
        "batch_hash": batch.id, "frame_id": frame_id, "notes": "", "uncertain": False,
        "offensive_rebound": False}


def save_frame_review(store, batch, frame_id, *, notes, uncertain, offensive_rebound,
                      expected_revision, start=None, end=None, split=None,
                      target_id=None, target_revision=None):
    """Save a human draft without creating or accepting annotations.

    An uncertain draft retains its historical target for context, but cannot be
    used for labeling until the person explicitly confirms its boundaries again.
    """
    old = frame_review(store, batch, frame_id)
    draft = {}
    if start is not None or end is not None:
        start, end = _number(start), _number(end)
        source = batch.manifest["sources"][batch.frames[frame_id]["game_id"]]
        if not 0 <= start < end <= source["duration"]:
            raise ValueError("Draft boundaries must fit within the source game")
        draft.update(draft_start=start, draft_end=end)
    if split is not None:
        if split not in {"train", "val"}:
            raise ValueError("Draft split must be train or val")
        draft["draft_split"] = split
    if start is not None or target_id is not None or target_revision is not None:
        if target_id is not None and not isinstance(target_id, str):
            raise ValueError("Draft target ID must be text")
        if target_revision is not None and (type(target_revision) is not int or target_revision < 1):
            raise ValueError("Draft target revision must be a positive integer")
        draft.update(draft_target_id=target_id, draft_target_revision=target_revision)
    changed_boundaries = False
    if old.get("possession_id") and draft:
        target = store.get("possession", old["possession_id"])
        snapshot = {"draft_start": target.get("start") if target else None,
                    "draft_end": target.get("end") if target else None,
                    "draft_split": target.get("split") if target else None,
                    "draft_target_id": old["possession_id"],
                    "draft_target_revision": old.get("possession_revision")}
        changed_boundaries = any(value != snapshot[key] for key, value in draft.items())
    return store.save("review_frame", {**old, "notes": str(notes), "uncertain": bool(uncertain),
                      "requires_boundary_confirmation": bool(uncertain) or changed_boundaries or old.get(
                          "requires_boundary_confirmation", False),
                      "offensive_rebound": bool(offensive_rebound), **draft},
                      expected_revision=expected_revision)


def save_review(store, batch, item_id, *, notes, uncertain, offensive_rebound, expected_revision):
    old = item_review(store, batch, item_id)
    return store.save("review_item", {**old, "notes": str(notes), "uncertain": bool(uncertain),
                      "offensive_rebound": bool(offensive_rebound)}, expected_revision=expected_revision)


def match_game(store, batch, game_id):
    source = batch.manifest["sources"][game_id]
    game = store.get("game", game_id)
    if not game or game.get("source_fingerprint") != source["source_fingerprint"]:
        raise ValueError("Import the matching original game locally; batch does not import games")
    if any(game.get(key) is not None and game[key] != source[key] for key in ("width", "height")):
        raise ValueError("Batch and local game native dimensions differ")
    for key in ("duration", "split_mode", "split", "split_blocks"):
        if game.get(key, [] if key == "split_blocks" else None) != source.get(key, [] if key == "split_blocks" else None):
            raise ValueError("Batch and local source split policy differ")
    if game.get("split_guard_seconds", 10) != 10:
        raise ValueError("Source guard policy differs")
    return game


def original_available(store, batch, game_id):
    game = match_game(store, batch, game_id)
    path = Path(game.get("source_path", ""))
    if not path.is_file():
        return False
    source = batch.manifest["sources"][game_id]
    return (path.stat().st_size == source["source_size_bytes"]
            and source_fingerprint(path) == source["source_fingerprint"])


def confirm_possession(store, batch, item_id, *, start, end, expected_revision,
                       target_id=None, target_revision=None, frame_id=None, split=None, notes="",
                       uncertain=False, offensive_rebound=False):
    item = batch.items[item_id]
    if frame_id is None:
        candidates = [f["id"] for f in batch.frames.values() if f["review_item_id"] == item_id]
        if len(candidates) != 1:
            raise ValueError("Select an explicit frame for this possession link")
        frame_id = candidates[0]
    if frame_id not in batch.frames or batch.frames[frame_id]["review_item_id"] != item_id:
        raise ValueError("Selected frame does not belong to this review item")
    if uncertain:
        raise ValueError("Resolve boundary uncertainty before confirming a reviewed possession")
    game = match_game(store, batch, item["game_id"])
    if not original_available(store, batch, item["game_id"]):
        raise ValueError("Matching original video required for boundary review")
    start, end = _number(start), _number(end)
    split = item["split"] if split is None else split
    if split != item["split"] or not any(low <= start < end <= high and s == split
                                         for low, high, s in _safe_spans(game)):
        raise ValueError("Corrected possession violates declared split or guards")
    selected = [batch.frames[frame_id]]
    if any(not start <= f["time"] <= end for f in selected):
        raise ValueError("Corrected possession must contain the selected frame")
    if any(batch.media(f["id"])["native_path"] is None for f in selected):
        raise ValueError("Native frames required before boundary confirmation")
    old = frame_review(store, batch, frame_id)
    checks = [("game", game["id"], game["revision"])]
    if old.get("possession_id"):
        linked = store.get("possession", old["possession_id"])
        if (not linked or linked["revision"] != old["possession_revision"]) and not (
                target_id == old["possession_id"] and linked and target_revision == linked["revision"]):
            raise ValueError("Linked possession changed; reload and explicitly relink it")
        if target_id is None:
            raise ValueError("Already linked; choose that existing possession to reconfirm")
        checks.append(("possession", linked["id"], linked["revision"]))
    changes = []
    collections = []
    if target_id:
        target = store.get("possession", target_id)
        if (not target or target_revision is None or target["revision"] != target_revision
                or target.get("reviewed") is not True or target.get("source") != "manual"
                or target["game_id"] != item["game_id"] or target.get("split") != split
                or target["start"] != start or target["end"] != end):
            raise ValueError("Choose a current reviewed manual possession with these bounds and split")
        checks.append(("possession", target_id, target_revision))
        revision = target_revision
    else:
        manual = store.list("possession", game_id=game["id"], source="manual")
        collections.append(("possession", {"game_id": game["id"], "source": "manual"},
                            {p["id"]: p["revision"] for p in manual}))
        if any(p["start"] < end and start < p["end"] for p in manual):
            raise ValueError("An overlapping manual possession exists; review and link it explicitly")
        target_id, revision = uuid.uuid4().hex, 1
        target = {"id": target_id, "game_id": game["id"], "start": start, "end": end,
                  "split": split, "reviewed": True, "source": "manual",
                  "review_batch_hash": batch.id, "review_item_id": item_id,
                  "proposal_parent_id": item["possession_id"]}
        changes.append(("possession", target, 0))
    record = {**old, "possession_id": target_id, "possession_revision": revision,
              "requires_boundary_confirmation": False,
              "draft_start": start, "draft_end": end, "draft_split": split,
              "draft_target_id": target_id, "draft_target_revision": revision,
              "notes": str(notes), "uncertain": bool(uncertain), "offensive_rebound": bool(offensive_rebound)}
    changes.append(("review_frame", record, expected_revision))
    return store.save_many(changes, checks=checks, collections=collections)[-1]


def frame_selection(store, batch, frame_id):
    frame = batch.frames[frame_id]
    game = match_game(store, batch, frame["game_id"])
    review = frame_review(store, batch, frame_id)
    target = store.get("possession", review.get("possession_id", ""))
    if (review.get("uncertain") or review.get("requires_boundary_confirmation")
            or not target or target["revision"] != review.get("possession_revision")
            or not target.get("reviewed") or target.get("source") != "manual"
            or target["game_id"] != game["id"] or target["split"] != frame["split"]
            or not target["start"] <= frame["time"] <= target["end"]
            or not any(low <= target["start"] < target["end"] <= high and split == frame["split"]
                       for low, high, split in _safe_spans(game))):
        raise ValueError("Confirm a current whole-possession link before labeling")
    if not original_available(store, batch, game["id"]) or batch.media(frame_id)["native_path"] is None:
        raise ValueError("Matching original video and native frame required for labeling")
    return {"game_id": game["id"], "possession_id": target["id"], "time": frame["time"],
            "possession_revision": target["revision"], "review_item_id": review["id"],
            "review_revision": review["revision"], "review_kind": "review_frame",
            "game_revision": game["revision"]}


def request_missing_media(store, batch, frame_id):
    if frame_id not in batch.frames:
        raise ValueError("Frame not in batch")
    return request_attention(store, f"review-media:{batch.id}:{batch.frames[frame_id]['game_id']}",
        "Restore the matching original video, local game record and full native review frames "
        "on this machine, then reopen the batch. Preview thumbnails are context only.", deliver=False)
