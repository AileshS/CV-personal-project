"""Validated possession edits and annotations, separate from model predictions."""
from __future__ import annotations

import math

SPLITS = {"unassigned", "train", "val", "test"}


def _finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Times and coordinates must be finite")
    return result


def _invalidate(store, possession_id):
    for kind in ("analysis", "clip"):
        record = store.get(kind, possession_id)
        if record:
            store.save(kind, {**record, "stale": True})


def save_possession(store, game_id, start, end, record_id=None, reviewed=False,
                    split="unassigned", **extra):
    game = store.get("game", game_id)
    if not game:
        raise ValueError("Game not found")
    start, end = _finite(start), _finite(end)
    if not 0 <= start < end <= game["duration"] + 0.001:
        raise ValueError("Possession must fit within the game and have positive duration")
    if split not in SPLITS:
        raise ValueError("Choose unassigned, train, val or test")
    if split == "unassigned":
        split = inherited_split(game, start, end)
    old = store.get("possession", record_id) if record_id else None
    if old and old["game_id"] != game_id:
        raise ValueError("A possession cannot move between games")
    # A label cannot silently become out of range after a boundary change.
    if old:
        children = [r for kind in ("shot", "box", "frame", "event")
                    for r in store.list(kind, possession_id=record_id)]
        if any(not start <= float(r.get("release", r.get("time", start))) <= end for r in children):
            raise ValueError("This edit would exclude saved annotations. Move/remove those labels first.")
        if old.get("split", "unassigned") != split and children:
            raise ValueError("Assign the split before labeling. Existing labels must be moved explicitly.")
    record = {**(old or {}), **extra, "game_id": game_id, "start": start, "end": end,
              "reviewed": bool(reviewed), "split": split, "source": "manual"}
    if record_id:
        record["id"] = record_id
    saved = store.save("possession", record)
    if old:
        _invalidate(store, record_id)
    return saved


def split_possession(store, possession_id, at):
    old = store.get("possession", possession_id)
    if old is None:
        raise ValueError("Possession not found")
    at = _finite(at)
    if not old["start"] < at < old["end"]:
        raise ValueError("Split point must be inside the possession")
    # Preserve the original ID for left; migrate existing annotations to the right.
    right = save_possession(store, old["game_id"], at, old["end"], reviewed=False,
                            split=old.get("split", "unassigned"))
    left = store.save("possession", {**old, "end": at, "reviewed": False, "source": "manual"})
    for kind in ("shot", "box", "frame", "event"):
        for record in store.list(kind, possession_id=possession_id):
            if float(record.get("release", record.get("time", 0))) >= at:
                store.save(kind, {**record, "possession_id": right["id"]})
    _invalidate(store, possession_id)
    return left, right


def merge_possessions(store, possession_ids):
    ids = list(dict.fromkeys(possession_ids))
    if len(ids) < 2:
        raise ValueError("Select at least two possessions")
    records = [store.get("possession", x) for x in ids]
    if any(r is None for r in records):
        raise ValueError("Possession not found")
    if len({r["game_id"] for r in records}) != 1 or len({r.get("split") for r in records}) != 1:
        raise ValueError("Merge requires possessions from the same game and dataset split")
    records.sort(key=lambda r: r["start"])
    others = store.list("possession", game_id=records[0]["game_id"])
    if any(r["id"] not in ids and r["start"] < records[-1]["end"] and r["end"] > records[0]["start"]
           for r in others):
        raise ValueError("Select all possessions between the merge endpoints")
    saved = store.save("possession", {**records[0], "end": records[-1]["end"], "reviewed": False,
                                      "source": "manual"})
    for old in records[1:]:
        for kind in ("shot", "box", "frame", "event"):
            for record in store.list(kind, possession_id=old["id"]):
                store.save(kind, {**record, "possession_id": saved["id"]})
        _invalidate(store, old["id"])
        store.delete("possession", old["id"])
    _invalidate(store, saved["id"])
    return saved


def validate_annotation(store, kind: str, record: dict) -> dict:
    if kind not in {"shot", "box", "frame", "event"}:
        raise ValueError("Unsupported annotation type")
    record = dict(record)
    parent = store.get("possession", record.get("possession_id", ""))
    if not parent or parent["game_id"] != record.get("game_id"):
        raise ValueError("Annotation must belong to a valid possession and game")
    time_key = "release" if kind == "shot" else "time"
    time = _finite(record.get(time_key, -1))
    if not parent["start"] <= time <= parent["end"]:
        raise ValueError("Annotation time must be inside its possession")
    if record.get("split", parent.get("split", "unassigned")) != parent.get("split", "unassigned"):
        raise ValueError("Annotation and possession splits differ")
    record["split"] = parent.get("split", "unassigned")
    record[time_key] = time
    if kind == "shot" and record.get("result") not in {"made", "missed", "unknown"}:
        raise ValueError("Shot result must be made, missed or unknown")
    if kind == "box":
        if record.get("label") not in {"ball", "rim"}:
            raise ValueError("Box label must be ball or rim")
        box = [_finite(x) for x in record.get("box", [])]
        if len(box) != 4 or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
            raise ValueError("Boxes need normalized left/top/right/bottom values in 0–1")
        record["box"] = box
        if any(f.get("negative") for f in store.list("frame", possession_id=parent["id"])
               if abs(f["time"]-time) < 0.001):
            raise ValueError("Remove the negative-frame label before adding a box")
    if kind == "frame" and record.get("negative"):
        if any(abs(b["time"]-time) < 0.001 for b in store.list("box", possession_id=parent["id"])):
            raise ValueError("A negative frame cannot contain labeled boxes")
    if kind == "event" and record.get("type") not in {"foul", "turnover"}:
        raise ValueError("Manual event must be foul or turnover")
    record["source"] = "human"
    return record


def assign_game_split(store, game_id: str, split: str):
    if split not in {"train", "val", "test"}:
        raise ValueError("Expected train, val or test")
    game = store.get("game", game_id)
    if not game:
        raise ValueError("Game not found")
    store.save("game", {**game, "split": split, "split_mode": "game", "split_blocks": []})
    for possession in store.list("possession", game_id=game_id):
        store.save("possession", {**possession, "split": split})
    for kind in ("shot", "box", "frame", "event"):
        for annotation in store.list(kind, game_id=game_id):
            store.save(kind, {**annotation, "split": split})


def inherited_split(game: dict, start: float, end: float) -> str:
    if game.get("split_mode") == "game" and game.get("split") in {"train", "val", "test"}:
        return game["split"]
    if game.get("split_mode") == "contiguous":
        for block in game.get("split_blocks", []):
            low = float(block["start"]) + (10 if block["start"] > 0 else 0)
            high = float(block["end"]) - (10 if block["end"] < game["duration"] else 0)
            if low <= start < end <= high:
                return block["split"]
    return "unassigned"


def assign_split_blocks(store, game_id: str, blocks: list[dict]):
    game = store.get("game", game_id)
    if not game:
        raise ValueError("Game not found")
    if any(store.list(kind, game_id=game_id) for kind in ("shot", "box", "frame", "event")):
        raise ValueError("Declare split blocks before annotating this game")
    blocks = sorted(blocks, key=lambda b: b["start"])
    if not blocks:
        raise ValueError("Supply at least one split block")
    for index, block in enumerate(blocks):
        if (block.get("split") not in {"train", "val", "test"}
                or not 0 <= _finite(block["start"]) < _finite(block["end"]) <= game["duration"]):
            raise ValueError("Split blocks need valid times and train/val/test labels")
        if index and blocks[index-1]["end"] > block["start"]:
            raise ValueError("Split blocks overlap")
    game = store.save("game", {**game, "split": "unassigned", "split_mode": "contiguous",
                               "split_blocks": blocks, "split_guard_seconds": 10})
    for record in store.list("possession", game_id=game_id):
        store.save("possession", {**record, "split": inherited_split(game, record["start"], record["end"])})
    return game
