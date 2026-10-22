"""Portable review batches; machine suggestions never become human labels."""
from __future__ import annotations

import html
import json
import math
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PIL import Image, ImageDraw

from playlab.dataset import canonical_hash, file_hash
from playlab.media import frame_at, source_fingerprint

SPLITS = {"train", "val", "test"}


def _safe_spans(game):
    """Only use explicitly assigned source partitions, before any sampling."""
    duration = float(game["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Source duration must be finite and positive.")
    if game.get("split_mode") == "game" and game.get("split") in SPLITS:
        return [(0.0, duration, game["split"])]
    if game.get("split_mode") not in {"contiguous", "blocks"}:
        raise ValueError("Assign source games or declared split blocks before preparing a review batch.")
    blocks = sorted(game.get("split_blocks", []), key=lambda row: float(row["start"]))
    if not blocks:
        raise ValueError("Declared split blocks are missing.")
    spans = []
    previous_end = 0.0
    for block in blocks:
        low, high = float(block["start"]), float(block["end"])
        if (block.get("split") not in SPLITS or not all(math.isfinite(v) for v in (low, high))
                or not 0 <= low < high <= duration or low < previous_end):
            raise ValueError("Split blocks must be finite, nonoverlapping source intervals.")
        previous_end = high
        start = low + (10 if low > 0 else 0)
        end = high - (10 if high < duration else 0)
        if start < end:
            spans.append((start, end, block["split"]))
    return spans


def _evenly_spaced(rows, count):
    if len(rows) <= count:
        return list(rows)
    return [rows[min(len(rows) - 1, int((index + .5) * len(rows) / count))] for index in range(count)]


def _balanced_select(groups, count):
    """Round-robin partitions/games, with temporal stratification inside each."""
    ordered = {key: list(rows) for key, rows in sorted(groups.items()) if rows}
    allocated = dict.fromkeys(ordered, 0)
    for _ in range(count):
        available = [key for key, rows in ordered.items() if allocated[key] < len(rows)]
        if not available:
            break
        key = min(available, key=lambda item: (allocated[item], item))
        allocated[key] += 1
    return [row for key, rows in ordered.items() for row in _evenly_spaced(rows, allocated[key])]


def _review_items(store, games, splits, allow_review_windows, window_seconds):
    groups = {}
    for game in sorted(games, key=lambda item: item["id"]):
        spans = _safe_spans(game)
        possessions = sorted(store.list("possession", game_id=game["id"]), key=lambda row: (row["start"], row["id"]))
        if not possessions:
            raise ValueError("Run possession indexing or add editable proposals before preparing a review batch.")
        for parent in possessions:
            start, end = float(parent["start"]), float(parent["end"])
            if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= game["duration"]:
                raise ValueError("Possession proposal is outside its source game.")
            for low, high, split in spans:
                if split not in splits:
                    continue
                if allow_review_windows:
                    left, right = max(start, low), min(end, high)
                    windows = []
                    while right - left >= .5:
                        stop = min(right, left + window_seconds)
                        windows.append((left, stop))
                        left = stop
                elif (parent.get("split") == split and low <= start < end <= high):
                    windows = [(start, end)]
                else:
                    continue
                for left, right in windows:
                    whole = left == start and right == end
                    item = {"id": canonical_hash([game["id"], parent["id"], left, right, split])[:24],
                            "game_id": game["id"], "possession_id": parent["id"],
                            "possession_revision": parent["revision"], "start": left, "end": right,
                            "split": split, "item_kind": "possession_proposal" if whole else "review_window",
                            "reviewed": False, "source": "review_sampling",
                            "requires_boundary_review": True,
                            "requires_parent_split_edit": parent.get("split") != split,
                            "warning": "Review whole-possession boundaries; retain offensive rebounds. "
                                       "A review window is not a claimed possession."}
                    groups.setdefault((split, game["id"]), []).append(item)
    return groups


def _suggestions(detector, pixels):
    height, width = pixels.shape[:2]
    detections = detector(pixels)
    result = []
    for key, label in (("balls", "ball"), ("rims", "rim")):
        for detection in detections.get(key, []):
            raw = detection["bbox"]
            confidence = float(detection["confidence"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                continue
            box = [max(0.0, min(1.0, value / (width if index % 2 == 0 else height)))
                   for index, value in enumerate(raw)]
            if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[0] >= box[2] or box[1] >= box[3]:
                continue
            result.append({"label": label, "box": box, "confidence": confidence,
                           "reviewed": False, "source": "model_proposal"})
    return result


def prepare_label_batch(store, destination, *, frame_count=300, possession_count=50,
                        splits=("train", "val"), game_ids=None, model_assisted=False,
                        weights=None, device="cpu", allow_review_windows=False,
                        window_seconds=30, detector=None) -> Path:
    """Write a versioned review bundle without changing Store or human labels.

    Use real split-assigned possession proposals by default. Explicit review-window
    mode can divide a long unsplit proposal inside its source's declared partitions;
    these windows require human boundary/split edits before labeling in the viewer.
    The held-out test split is excluded unless requested explicitly. Uniform-time
    sampling covers the source timeline; it does not prove visual/basketball diversity.
    ``detector`` is an optional injected callable for offline testing/adapters.
    """
    if (not isinstance(frame_count, int) or not isinstance(possession_count, int)
            or possession_count < 1 or frame_count < possession_count):
        raise ValueError("Choose positive review counts with at least one frame per review item.")
    if not math.isfinite(window_seconds) or window_seconds < .5:
        raise ValueError("Review windows must be at least half a second.")
    splits = tuple(sorted(set(splits)))
    if not splits or any(split not in SPLITS for split in splits):
        raise ValueError("Choose train, val and/or explicitly requested test partitions.")
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Review destination exists; choose a new batch version.")
    available = {game["id"]: game for game in store.list("game")}
    requested = set(game_ids or available)
    if not requested or not requested <= available.keys():
        raise ValueError("Import source footage first; every requested game must exist.")
    games = [available[key] for key in sorted(requested)]
    # Exclude held-out games without opening or decoding their media.
    games = [game for game in games if any(split in splits for _, _, split in _safe_spans(game))]
    groups = _review_items(store, games, splits, allow_review_windows, window_seconds)
    count = sum(len(rows) for rows in groups.values())
    if count < possession_count:
        raise ValueError(f"Only {count} split-safe review items are available; requested {possession_count}. "
                         "Add/edit proposals, lower the count, or explicitly allow boundary-review windows.")
    items = _balanced_select(groups, possession_count)
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Duplicate review intervals detected.")
    provenance = {"mode": "none", "weights_sha256": None}
    if model_assisted:
        if weights and not Path(weights).is_file():
            raise FileNotFoundError("Requested model weights are unavailable.")
        provided_detector = detector is not None
        if detector is None:
            from playlab.detectors import BasketballDetector
            detector = BasketballDetector(weights, device)
        provenance = {"mode": getattr(detector, "mode", "provided_detector"), "device": str(device),
                      "weights_sha256": file_hash(weights) if weights else None,
                      "limitation": "Pretrained execution/prelabels do not establish basketball accuracy. "
                                    "COCO has no rim class; missing suggestions are not negative labels."}
        if not provided_detector:
            model_dir = Path(os.environ.get("PLAYLAB_MODEL_DIR", Path(__file__).parents[1] / "artifacts/models"))
            baseline = model_dir / "yolo11n.pt"
            provenance["baseline_weights_sha256"] = file_hash(baseline) if baseline.is_file() else None
            for package in ("ultralytics", "torch"):
                try:
                    provenance[f"{package}_version"] = version(package)
                except PackageNotFoundError:
                    provenance[f"{package}_version"] = "unavailable"
    elif detector is not None or weights:
        raise ValueError("Enable model_assisted explicitly when supplying a detector or weights.")
    source_provenance = {}
    for game in games:
        path = Path(game["source_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Source footage is unavailable for game {game['id']}; import it on this machine.")
        source_provenance[game["id"]] = {
            "duration": game["duration"], "width": game.get("width"), "height": game.get("height"),
            "source_fingerprint": source_fingerprint(path), "source_size_bytes": path.stat().st_size,
            "split_mode": game.get("split_mode"), "split": game.get("split"),
            "split_blocks": game.get("split_blocks", []),
            "fingerprint_definition": "size + first/last MiB, not a full video SHA-256"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".review-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "batch"
        (staging / "images").mkdir(parents=True)
        frames, files, seen = [], {}, set()
        cards = []
        for index, item in enumerate(items):
            amount = frame_count // len(items) + (index < frame_count % len(items))
            for sample in range(amount):
                at = item["start"] + (sample + .5) * (item["end"] - item["start"]) / amount
                key = (item["game_id"], round(at, 6))
                if key in seen:
                    raise ValueError("Overlapping proposals produced duplicate samples; edit boundaries before batching.")
                seen.add(key)
                game = available[item["game_id"]]
                pixels = frame_at(game["source_path"], at, max_width=None)
                identifier = canonical_hash([item["id"], at])[:24]
                relative = f"images/{identifier}.png"
                image = Image.fromarray(pixels)
                image.save(staging / relative)
                files[relative] = file_hash(staging / relative)
                proposals = []
                if model_assisted:
                    if hasattr(detector, "reset"):
                        detector.reset()  # Independent stills must never share tracking state.
                    proposals = _suggestions(detector, pixels)
                record = {"id": identifier, "game_id": item["game_id"], "possession_id": item["possession_id"],
                          "review_item_id": item["id"], "time": at, "split": item["split"], "image": relative,
                          "width": image.width, "height": image.height, "reviewed": False,
                          "negative": False, "source": "review_sampling", "proposed_boxes": proposals}
                frames.append(record)
                preview = image.copy()
                draw = ImageDraw.Draw(preview)
                for suggestion in proposals:
                    box = [value * (image.width if i % 2 == 0 else image.height)
                           for i, value in enumerate(suggestion["box"])]
                    draw.rectangle(box, outline="#ffbd45", width=3)
                    draw.text((box[0], max(0, box[1] - 12)), suggestion["label"], fill="#ffbd45")
                preview.thumbnail((480, 320))
                preview_relative = f"images/{identifier}-preview.jpg"
                preview.save(staging / preview_relative)
                files[preview_relative] = file_hash(staging / preview_relative)
                cards.append(f'<article><img src="{preview_relative}" alt="Unreviewed suggested boxes">'
                             f'<p>{html.escape(item["game_id"])} | {at:.3f}s | {item["split"]} | '
                             f'{html.escape(item["item_kind"])}<br>Frame {identifier}: '
                             f'{len(proposals)} unreviewed box suggestions</p></article>')
        manifest = {"schema": 1, "kind": "unreviewed_label_batch", "reviewed": False,
                    "selection": "deterministic temporal strata balanced by game and source partition",
                    "split_policy": "declared source groups before sampling; 10-second internal guards",
                    "requested": {"frames": frame_count, "review_items": possession_count, "splits": list(splits)},
                    "frame_count": len(frames), "review_item_count": len(items), "sources": source_provenance,
                    "model": provenance, "review_items": items, "frames": frames, "files": files,
                    "limitations": ["Not a reviewed training/evaluation dataset.",
                                    "Review every visible ball/rim, including missed detections; do not accept all suggestions.",
                                    "Review whole-possession boundaries and replays before labeling; retain offensive rebounds.",
                                    "Review windows are sampling aids, not verified whole possessions.",
                                    "No source videos are included; elapsed timestamps refer to the original imported source.",
                                    "Temporal sampling does not guarantee game/camera/shot-type diversity."]}
        manifest["batch_hash"] = canonical_hash(manifest)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        title = "Unreviewed basketball label batch"
        page = ('<!doctype html><meta charset="utf-8"><title>' + title + '</title>'
                '<style>body{font:16px sans-serif;margin:2rem}main{display:flex;flex-wrap:wrap;gap:1rem}'
                'article{width:480px;border:1px solid #ccc;padding:.5rem}img{max-width:100%}</style>'
                '<h1>' + title + '</h1><p>All suggested boxes and review items require human verification. '
                'No suggestions are training labels. Correct source possession boundaries/splits in the viewer first; '
                'then label each listed source time, review every ball/rim, and mark the complete frame reviewed '
                'only after corrections. Empty suggestions are not negative labels. Foul/turnover labels stay manual.</p>'
                '<main>' + ''.join(cards) + '</main>')
        (staging / "index.html").write_text(page, encoding="utf-8")
        staging.rename(destination)
    return destination
