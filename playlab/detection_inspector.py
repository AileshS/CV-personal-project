"""Read-only, bounded detector inspection. Predictions never become human labels."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

from playlab import analysis
from playlab.detectors import BasketballDetector
from playlab.media import frame_at
from playlab.quick_analysis import _allow_interval


def sampled_detections(store, clip, result):
    """Return every persisted inference call, including boxes rejected by shot logic."""
    if not result:
        return []
    if "diagnostic_frames" in result:
        return sorted(result["diagnostic_frames"], key=lambda row: (row["time"], row.get("sequence", 0)))
    rows = []
    for index in range(result.get("chunks", 0)):
        key = f"{clip['id']}:{result['fingerprint']}:{index}"
        chunk = store.get("quick_analysis_chunk", key) or {}
        rows.extend({"stage": "scan", "inference_scope": "full_frame", **row}
                    for row in chunk.get("frames", []) if "raw_detections" in row)
    return sorted(rows, key=lambda row: row["time"])


def validate_frame(game, clip, at):
    at = float(at)
    if not math.isfinite(at) or not float(clip["start"]) <= at < float(clip["end"]):
        raise ValueError("Choose a frame inside this interval.")
    _allow_interval(game, clip["start"], clip["end"])
    return at


def inspect_frame(game, clip, at, detector=None):
    """Run one stateless inference, independent of the shot classifier and worker."""
    at = validate_frame(game, clip, at)
    # The model input matches the 1280-wide analysis input; display uses native resolution.
    frame = frame_at(game["source_path"], at, max_width=1280)
    if detector is None:
        default = analysis._default_model_path()
        if not default.is_file():
            raise ValueError("Default detector weights are missing. Restore the existing local weights first.")
        detector = BasketballDetector(clip.get("weights"), clip.get("device", "cpu"))
    detector.reset()
    detections = detector(frame)
    height, width = frame.shape[:2]
    return {"time": at, "stage": "on_demand", "inference_scope": "full_frame",
            "raw_detections": {k: detections.get(k, []) for k in ("balls", "rims", "players")},
            "width": width, "height": height, "detector_mode": detector.mode}


def overlay(frame, row, hint=None):
    """Scale model-input boxes to the native frame; a hint has no model confidence."""
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    sx, sy = image.width / row["width"], image.height / row["height"]
    table = []
    for key, name, color in (("balls", "Ball", "#ffff00"), ("rims", "Rim", "#00ffff"),
                             ("players", "Player", "#ff8c42")):
        for item in row.get("raw_detections", {}).get(key, []):
            box = [float(v) * (sx if i % 2 == 0 else sy) for i, v in enumerate(item["bbox"])]
            score = float(item["confidence"])
            draw.rectangle(box, outline=color, width=3)
            draw.text((max(0, box[0]), max(0, box[1]-13)), f"{name} {score:.1%}", fill=color,
                      stroke_width=1, stroke_fill="black")
            table.append({"Object": name, "Detector score": score,
                          "Native box (left, top, right, bottom)": [round(v, 1) for v in box]})
    if hint:
        box = [v * (image.width if i % 2 == 0 else image.height) for i, v in enumerate(hint)]
        draw.rectangle(box, outline="#ff55ff", width=3)
        draw.text((box[0], max(0, box[1]-13)), "Basket hint (not a prediction)", fill="#ff55ff")
    return image, table
