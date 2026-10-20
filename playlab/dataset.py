"""Export human-reviewed labels with explicit split and leakage checks."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

SPLITS = {"train", "val", "test"}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                          .encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _interval(record):
    if "release" in record:
        release = float(record["release"])
        return float(record.get("context_start", max(0, release - 1))), float(record.get("context_end", release + 5))
    t = float(record["time"])
    return t, t


def validate_splits(records: list[dict], games: list[dict], guard_seconds: float = 10) -> None:
    """Games stay in one split unless explicitly declared contiguous split blocks.

    Each block has {start,end,split}; label contexts stay 10 s away from internal
    boundaries. replay_group/replay_of and possession identity also stay grouped.
    """
    game_map = {str(g["id"]): g for g in games}
    groups = {}
    grouped_games = {}
    for r in records:
        if r.get("reviewed") is not True:
            raise ValueError("Every exported annotation must be human reviewed.")
        split = r.get("split")
        if split not in SPLITS:
            raise ValueError("Assign every reviewed label a train, val or test split.")
        game_id = str(r.get("game_id", ""))
        if game_id not in game_map:
            raise ValueError(f"Unknown source game: {game_id}")
        if not r.get("possession_id"):
            raise ValueError("Each label needs a possession_id for leakage protection.")
        group_keys = [("possession", game_id, str(r["possession_id"])), ("replay", str(r["id"]))]
        for key in ("replay_group", "replay_of"):
            if r.get(key):
                group_keys.append(("replay", str(r[key])))
        for key in group_keys:
            if key in groups and groups[key] != split:
                raise ValueError("Possession or replay labels cross dataset splits.")
            groups[key] = split
        start, end = _interval(r)
        duration = float(game_map[game_id]["duration"])
        if not all(math.isfinite(v) for v in (start, end, duration)) or not 0 <= start <= end <= duration:
            raise ValueError("Annotation time/context is outside the source game.")
        grouped_games.setdefault(game_id, []).append((r, start, end))
    for game_id, rows in grouped_games.items():
        game = game_map[game_id]
        if game.get("split_mode") == "game" and game.get("split") in SPLITS:
            if any(r["split"] != game["split"] for r, _, _ in rows):
                raise ValueError("Annotations disagree with the source game's declared split.")
        blocked = game.get("split_mode") in {"contiguous", "blocks"}
        if len({r["split"] for r, _, _ in rows}) <= 1 and not blocked:
            continue
        if not blocked:
            raise ValueError("A game crosses splits. Declare contiguous blocks or split by game.")
        blocks = sorted(game.get("split_blocks", []), key=lambda b: b["start"])
        if not blocks:
            raise ValueError("Contiguous mode requires explicit split_blocks.")
        for i, block in enumerate(blocks):
            if block.get("split") not in SPLITS or not 0 <= block["start"] < block["end"] <= game["duration"]:
                raise ValueError("Invalid split block.")
            if i and blocks[i - 1]["end"] > block["start"]:
                raise ValueError("Split blocks overlap.")
        for r, start, end in rows:
            fits = False
            for block in blocks:
                low = block["start"] + (guard_seconds if block["start"] > 0 else 0)
                high = block["end"] - (guard_seconds if block["end"] < game["duration"] else 0)
                if r["split"] == block["split"] and low <= start <= end <= high:
                    fits = True
                    break
            if not fits:
                raise ValueError("Label context touches a split boundary or its 10-second guard.")


def export_dataset(store, destination: str | Path) -> Path:
    """Export a portable, immutable YOLO/shot bundle without the source videos.

    Sources remain local. Images are the labeled frames at native resolution.
    'box', 'frame' and 'shot' record kinds form the public annotation interface.
    Empty YOLO labels are created only for explicitly reviewed negative frames.
    """
    from PIL import Image

    from playlab.media import frame_at

    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Dataset destination exists. Export a new version instead of overwriting.")
    games = store.list("game")
    records = [dict(r, kind=kind) for kind in ("box", "frame", "shot", "event")
               for r in store.list(kind) if r.get("reviewed") is True]
    if not records:
        raise ValueError("No human-reviewed labels are available.")
    validate_splits(records, games)
    # Visual features come only from saved predictions matched by time; never
    # derive them from the human outcome label or turn the prediction into gold.
    for record in records:
        if record["kind"] == "shot" and not record.get("features"):
            analysis = store.get("analysis", record["possession_id"])
            if analysis and analysis.get("stale"):
                continue
            candidates = [event for event in (analysis or {}).get("events", [])
                          if event.get("type") == "shot" and event.get("features")
                          and abs(float(event.get("release", -100)) - float(record["release"])) <= 0.3]
            if candidates:
                nearest = min(candidates, key=lambda e: abs(float(e["release"]) - float(record["release"])))
                record["features"] = nearest["features"]
                record["features_source"] = {"analysis_id": record["possession_id"], "event_id": nearest["id"],
                                             "analysis_fingerprint": analysis.get("fingerprint")}
    game_map = {str(g["id"]): g for g in games}
    frames = {}
    for r in records:
        if r["kind"] == "event":
            if r.get("type") not in {"foul", "turnover"}:
                raise ValueError("Manual events must be reviewed foul or turnover annotations.")
            continue
        if r["kind"] == "shot":
            if r.get("result") not in {"made", "missed", "unknown"}:
                raise ValueError("Shot result must be made, missed or unknown.")
            continue
        key = (str(r["game_id"]), round(float(r["time"]), 6))
        entry = frames.setdefault(key, {"split": r["split"], "boxes": [], "negative": False,
                                        "frame_reviewed": False, "records": []})
        if entry["split"] != r["split"]:
            raise ValueError("The same frame occurs in multiple splits.")
        entry["records"].append(r)
        if r["kind"] == "box":
            box = r.get("box", [])
            if (r.get("label") not in {"ball", "rim"} or len(box) != 4
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
                raise ValueError("Boxes need ball/rim labels and normalized, nonempty xyxy coordinates.")
            entry["boxes"].append(r)
        else:
            entry["frame_reviewed"] = True
            if r.get("negative") is True:
                entry["negative"] = True
    for entry in frames.values():
        if entry["boxes"] and not entry["frame_reviewed"]:
            raise ValueError("Mark the complete frame reviewed after labeling every visible ball and rim.")
    # Do not silently turn an unreviewed object in an exported frame into a negative.
    for r in store.list("box"):
        key = (str(r.get("game_id")), round(float(r.get("time", -1)), 6))
        if key in frames and r.get("reviewed") is not True:
            raise ValueError("An exported frame still has unreviewed boxes. Finish its review first.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dataset-", dir=destination.parent) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        files = {}
        seen_images = {}
        for (game_id, time), entry in frames.items():
            if entry["negative"] and entry["boxes"]:
                raise ValueError("A negative frame cannot contain positive box annotations.")
            if not entry["boxes"] and not entry["negative"]:
                continue
            token = canonical_hash([game_id, time])[:24]
            image_rel = Path("images") / entry["split"] / f"{token}.png"
            label_rel = Path("labels") / entry["split"] / f"{token}.txt"
            for rel in (image_rel, label_rel):
                (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            pixels = frame_at(game_map[game_id]["source_path"], time, max_width=None)
            Image.fromarray(pixels).save(staging / image_rel)
            digest = file_hash(staging / image_rel)
            if digest in seen_images and seen_images[digest] != entry["split"]:
                raise ValueError("Identical decoded frame images cross splits (possible replay/duplicate).")
            seen_images[digest] = entry["split"]
            lines = []
            for r in entry["boxes"]:
                x1, y1, x2, y2 = r["box"]
                lines.append(f"{0 if r['label'] == 'ball' else 1} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} "
                             f"{x2-x1:.8f} {y2-y1:.8f}")
                r["image"] = image_rel.as_posix()
            (staging / label_rel).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            files[image_rel.as_posix()] = digest
            files[label_rel.as_posix()] = file_hash(staging / label_rel)
        (staging / "records.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
        files["records.json"] = file_hash(staging / "records.json")
        # Relative paths remain portable after uploading or unpacking in Colab.
        (staging / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\n"
                                            "names:\n  0: ball\n  1: rim\n", encoding="utf-8")
        files["data.yaml"] = file_hash(staging / "data.yaml")
        metadata = [{k: v for k, v in g.items() if k != "source_path"} for g in games
                    if str(g["id"]) in {str(r["game_id"]) for r in records}]
        manifest = {"schema": 1, "kind": "human_reviewed_pilot", "games": metadata,
                    "record_count": len(records), "files": files,
                    "split_policy": "game_or_declared_contiguous_blocks_10s_guard",
                    "replay_policy": "replay groups checked when labeled; manually identify unmarked replays"}
        manifest["dataset_hash"] = canonical_hash(manifest)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(destination)
    return destination


def verify_dataset(dataset: str | Path) -> dict:
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    payload = {k: v for k, v in manifest.items() if k != "dataset_hash"}
    if canonical_hash(payload) != manifest.get("dataset_hash"):
        raise ValueError("Dataset manifest changed; export a new version.")
    for relative, expected in manifest["files"].items():
        path = (dataset / relative).resolve()
        if not path.is_relative_to(dataset) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Dataset file changed or missing: {relative}")
    return manifest
