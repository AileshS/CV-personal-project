"""Fixed, inspectable shot features; scores are not calibrated probabilities."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

FEATURE_NAMES = (
    "min_abs_x", "min_abs_y", "below_after_above", "below_x",
    "downward_fraction", "observation_count",
)


def compute_features(observations: list[dict], max_observation_gap: float = 0.35) -> dict:
    """Normalize ball positions by visible rim width, with image y increasing down.

    Observations require time,x,y,rim_x,rim_y,rim_width in consistent coordinates.
    Occluded/missing positions must be omitted, never interpolated by the caller.
    A crossing is considered only between consecutive observations <=0.35 s apart.
    These are visual heuristics, not proof that a basket went through the net.
    """
    points = []
    for row in observations:
        try:
            t, x, y, rx, ry, width = (
                float(row[k]) for k in ("time", "x", "y", "rim_x", "rim_y", "rim_width")
            )
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in (t, x, y, rx, ry, width)) and width > 0:
            points.append((t, (x - rx) / width, (y - ry) / width))
    points.sort()
    count = len(points)
    # JSON/SQLite use null for unavailable values; NaN is forbidden in records.
    result = {name: None for name in FEATURE_NAMES}
    result.update(observation_count=float(count), below_after_above=0.0, geometry_result="unknown")
    if not points:
        return result
    result["min_abs_x"] = min(abs(p[1]) for p in points)
    result["min_abs_y"] = min(abs(p[2]) for p in points)
    pairs = [(a, b) for a, b in zip(points, points[1:]) if 0 < b[0] - a[0] <= max_observation_gap]
    if pairs:
        result["downward_fraction"] = sum(b[2] > a[2] for a, b in pairs) / len(pairs)
    crossings = []
    for a, b in pairs:
        if a[2] < 0 <= b[2] and b[2] > a[2] and max(abs(a[2]), abs(b[2])) <= 2:
            alpha = -a[2] / (b[2] - a[2])
            crossings.append(abs(a[1] + alpha * (b[1] - a[1])))
    if crossings:
        result["below_after_above"] = 1.0
        result["below_x"] = min(crossings)
    if count >= 4 and crossings and min(crossings) <= 0.4:
        result["geometry_result"] = "made"
    elif count >= 4:
        # A miss requires a connected near-rim approach and subsequent lateral
        # departure. Absence of a detected crossing alone never establishes a miss.
        for a, b in pairs:
            near = abs(a[1]) <= 1.0 and abs(a[2]) <= 1.5
            departure = abs(b[1]) > 1.25 and abs(b[1]) > abs(a[1]) and abs(b[2]) <= 2
            if near and departure and not any(c <= 0.4 for c in crossings):
                result["geometry_result"] = "missed"
                break
    return result


def geometry_prediction(features: dict) -> str:
    try:
        count, distance = float(features.get("observation_count", 0)), float(features.get("min_abs_y"))
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(count) or count < 4 or not math.isfinite(distance) or distance > 2:
        return "unknown"
    value = features.get("geometry_result", "unknown")
    return value if value in {"made", "missed"} else "unknown"


def feature_vector(features: dict) -> list[float]:
    values = []
    for name in FEATURE_NAMES:
        try:
            value = float(features.get(name, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        values.append(value if math.isfinite(value) else float("nan"))
    return values


def predict_result(bundle_path: str | Path, features: dict) -> dict:
    """Load only trusted locally produced joblib bundles (joblib can execute code)."""
    import joblib

    bundle = joblib.load(bundle_path)
    if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Classifier feature version does not match this project.")
    if bundle.get("selected_method") == "geometry":
        return {"result": geometry_prediction(features), "score": None, "method": "geometry_baseline"}
    values = feature_vector(features)
    # Reject missing evidence even if imputation would produce a confident score.
    if not math.isfinite(values[5]) or values[5] < 4 or not math.isfinite(values[1]) or values[1] > 2:
        return {"result": "unknown", "score": None, "method": "trained_classifier"}
    probabilities = bundle["model"].predict_proba(np.asarray([values]))[0]
    index = int(np.argmax(probabilities))
    score = float(probabilities[index])
    result = str(bundle["model"].classes_[index]) if score >= bundle["threshold"] else "unknown"
    return {"result": result, "score": score, "method": "trained_classifier"}
