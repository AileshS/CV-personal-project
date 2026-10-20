"""Count honest coverage/errors and enforce a frozen, single-use test report."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from playlab.dataset import canonical_hash, file_hash, verify_dataset
from playlab.training import code_version


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _interval(correct, total):
    """95% Wilson interval; useful with small pilot denominators."""
    if not total:
        return None
    z, p = 1.96, correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def outcome_metrics(gold: list[dict], predictions: list[dict], tolerance: float = 0.5, *, _groups=True) -> dict:
    """Match event IDs or unique nearest release within a possession, then count.

    Unknown predictions and undetected shots remain in the coverage denominator.
    False-positive shot events are separately counted. Unknown gold is excluded
    from result correctness, but never treated as a negative made/missed example.
    """
    eligible = [g for g in gold if g.get("reviewed") is True and g.get("result") in {"made", "missed"}]

    def same_parent(g, p):
        return all(not (g.get(key) is not None and p.get(key) is not None) or g[key] == p[key]
                   for key in ("game_id", "possession_id"))

    remaining = set(range(len(predictions)))
    matched = []
    # Exact human annotation IDs are optional; automatic events use source time.
    for g in eligible:
        candidates = []
        for index in remaining:
            p = predictions[index]
            if p.get("id") and p.get("id") == g.get("id") and same_parent(g, p):
                candidates.append((0, index))
            elif (p.get("possession_id") == g.get("possession_id") and "release" in p and "release" in g
                  and (not p.get("game_id") or p.get("game_id") == g.get("game_id"))):
                error = abs(float(p["release"]) - float(g["release"]))
                if error <= tolerance:
                    candidates.append((error, index))
        index = min(candidates)[1] if candidates else None
        if index is not None:
            remaining.remove(index)
        matched.append((g, predictions[index] if index is not None else {}))
    confusion = {label: {prediction: 0 for prediction in ("made", "missed", "unknown")}
                 for label in ("made", "missed")}
    emitted, correct, baseline_emitted, baseline_correct = 0, 0, 0, 0
    defender_eligible, defender_emitted, defender_correct = 0, 0, 0
    anchored = 0
    for g, p in matched:
        result = p.get("result", "unknown")
        if result not in {"made", "missed"}:
            result = "unknown"
        confusion[g["result"]][result] += 1
        emitted += result != "unknown"
        correct += result == g["result"]
        baseline = p.get("baseline_result", "unknown")
        baseline_emitted += baseline in {"made", "missed"}
        baseline_correct += baseline == g["result"]
        anchored += p.get("origin") == "human_release_anchor"
        gold_defender = g.get("nearest_defender_id", g.get("defender_id"))
        if gold_defender is not None:
            defender_eligible += 1
            predicted = p.get("nearest_defender_id", p.get("defender_id"))
            if predicted is None and isinstance(p.get("defender"), dict):
                predicted = p["defender"].get("track_id")
            defender_emitted += predicted is not None
            defender_correct += predicted is not None and str(predicted) == str(gold_defender)
    per_class = {}
    for label in ("made", "missed"):
        count = sum(confusion[label].values())
        tp = confusion[label][label]
        predicted_count = sum(confusion[g][label] for g in ("made", "missed"))
        per_class[label] = {"gold_count": count, "correct": tp, "recall": _rate(tp, count),
                            "precision": _rate(tp, predicted_count), "supported": count >= 5}
    # Unknown gold can legitimately match a predicted shot event: do not call it
    # a false positive merely because its basket result is unassessable.
    for g in gold:
        if g.get("reviewed") is True and g.get("result") not in {"made", "missed"}:
            for index in list(remaining):
                p = predictions[index]
                exact = p.get("id") and p.get("id") == g.get("id") and same_parent(g, p)
                near = (p.get("possession_id") == g.get("possession_id") and "release" in p and "release" in g
                        and same_parent(g, p) and abs(float(p["release"]) - float(g["release"])) <= tolerance)
                if exact or near:
                    remaining.remove(index)
                    break
    for label in ("made", "missed"):
        predicted_count = sum(confusion[g][label] for g in ("made", "missed"))
        predicted_count += sum(predictions[i].get("result") == label for i in remaining)
        per_class[label]["precision"] = _rate(confusion[label][label], predicted_count)
    false_positive_outcomes = sum(predictions[i].get("result") in {"made", "missed"} for i in remaining)
    report = {"eligible": len(eligible), "matched": sum(bool(p) for _, p in matched),
            "emitted": emitted, "correct": correct, "coverage": _rate(emitted, len(eligible)),
            "accuracy_when_emitted": _rate(correct, emitted), "wilson_95": _interval(correct, emitted),
            "accuracy_including_false_positive_events": _rate(correct, emitted + false_positive_outcomes),
            "end_to_end_correct_fraction": _rate(correct, len(eligible)), "confusion": confusion,
            "per_class": per_class, "false_positive_events": len(remaining),
            "human_release_anchored_events": anchored,
            "defender": {"eligible": defender_eligible, "emitted": defender_emitted,
                         "correct": defender_correct, "coverage": _rate(defender_emitted, defender_eligible),
                         "accuracy_when_emitted": _rate(defender_correct, defender_emitted),
                         "wilson_95": _interval(defender_correct, defender_emitted),
                         "definition": "closest visible opposing track by bottom-center image distance at release"},
            "geometry_baseline": {"eligible": len(eligible), "emitted": baseline_emitted,
                                  "correct": baseline_correct, "coverage": _rate(baseline_emitted, len(eligible)),
                                  "accuracy_when_emitted": _rate(baseline_correct, baseline_emitted)}}
    if _groups:
        report["by_release_origin"] = {
            "automatic": outcome_metrics(gold, [p for p in predictions
                                                 if p.get("origin") == "automatic_rim_approach"],
                                         tolerance, _groups=False),
            "human_anchored": outcome_metrics(gold, [p for p in predictions
                                                     if p.get("origin") == "human_release_anchor"],
                                             tolerance, _groups=False),
            "unspecified": outcome_metrics(gold, [p for p in predictions if p.get("origin") not in
                                                  {"automatic_rim_approach", "human_release_anchor"}],
                                           tolerance, _groups=False),
        }
        report["origin_metrics_note"] = ("Each origin uses the full gold denominator. Automatic events from an assisted "
                                         "analysis are not an independent end-to-end automatic evaluation.")
    return report


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = max(0, a[2] - a[0]) * max(0, a[3] - a[1]), max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def detection_metrics(gold: list[dict], predictions: list[dict], iou_threshold: float = 0.5) -> dict:
    """Greedy confidence-ordered matching per frame/class; duplicates are FPs.

    Both inputs use normalized xyxy box, label and image/frame_id (or game/time).
    Callers include predictions from explicitly labeled negative frames too.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1].")

    def frame(record):
        return record.get("image", record.get("frame_id", (record.get("game_id"), record.get("time"))))

    unmatched = set(range(len(gold)))
    counts = {label: Counter() for label in {g["label"] for g in gold} | {p["label"] for p in predictions}}
    for p in sorted(predictions, key=lambda r: float(r.get("confidence", 0)), reverse=True):
        candidates = [(box_iou(p["box"], gold[i]["box"]), i) for i in unmatched
                      if p["label"] == gold[i]["label"] and frame(p) == frame(gold[i])]
        overlap, index = max(candidates) if candidates else (0, None)
        if index is not None and overlap >= iou_threshold:
            counts[p["label"]]["tp"] += 1
            unmatched.remove(index)
        else:
            counts[p["label"]]["fp"] += 1
    for index in unmatched:
        counts[gold[index]["label"]]["fn"] += 1
    result = {}
    for label, values in counts.items():
        tp, fp, fn = (values[name] for name in ("tp", "fp", "fn"))
        result[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": _rate(tp, tp + fp),
                         "recall": _rate(tp, tp + fn)}
    return {"iou_threshold": iou_threshold, "per_class": result,
            "limitation": "Fixed-threshold counts, not mAP or evidence of generalization."}


def freeze_run(dataset: str | Path, artifacts: list[str | Path], output: str | Path) -> Path:
    manifest = verify_dataset(dataset)
    if not artifacts:
        raise ValueError("Freeze trained weights and configuration before evaluating the locked test.")
    hashes = {str(Path(p).resolve()): file_hash(p) for p in artifacts}
    payload = {"dataset_hash": manifest["dataset_hash"], "artifacts": hashes, "code": code_version(),
               "policy": "single locked test; no tuning after this report"}
    payload["freeze_hash"] = canonical_hash(payload)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "freeze.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return destination


def verify_freeze(dataset: str | Path, freeze: str | Path, required_artifacts=()) -> tuple[dict, dict]:
    """Shared by detector and outcome test entry points; never opens gold labels."""
    manifest = verify_dataset(dataset)
    frozen = json.loads(Path(freeze).read_text(encoding="utf-8"))
    if canonical_hash({k: v for k, v in frozen.items() if k != "freeze_hash"}) != frozen["freeze_hash"]:
        raise ValueError("Freeze record was modified.")
    if frozen["dataset_hash"] != manifest["dataset_hash"] or frozen["code"] != code_version():
        raise ValueError("Frozen code or dataset changed. This run is not eligible for the locked test.")
    for artifact, expected in frozen["artifacts"].items():
        if file_hash(artifact) != expected:
            raise ValueError("Frozen model/configuration changed.")
    if any(str(Path(p).resolve()) not in frozen["artifacts"] for p in required_artifacts):
        raise ValueError("All evaluated weights and configuration must be included in the frozen artifacts.")
    return manifest, frozen


def evaluate_locked(dataset: str | Path, predictions, output: str | Path, freeze: str | Path | None = None) -> dict:
    """One locked-test report per output; failed attempts remain marked consumed."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    freeze_path = Path(freeze) if freeze else output / "freeze.json"
    manifest, frozen = verify_freeze(dataset, freeze_path)
    prediction_path = Path(predictions).resolve() if isinstance(predictions, (str, Path)) else None
    if prediction_path:
        if str(prediction_path) not in frozen["artifacts"]:
            raise ValueError("Freeze the prediction JSON with its effective inference configuration before evaluation.")
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    else:
        # The in-memory API must identify exactly the same frozen JSON snapshot.
        frozen_payload = False
        for artifact in frozen["artifacts"]:
            if Path(artifact).suffix.lower() == ".json":
                try:
                    frozen_payload |= json.loads(Path(artifact).read_text(encoding="utf-8")) == predictions
                except (ValueError, UnicodeError):
                    continue
        if not frozen_payload:
            raise ValueError("Freeze the prediction JSON with its effective inference configuration before evaluation.")
    if not isinstance(predictions, dict) or predictions.get("schema") != 2:
        raise ValueError("Use the provenance-bearing prediction export, not a raw prediction list.")
    if predictions.get("dataset_hash") != manifest["dataset_hash"] or predictions.get("split") != "test":
        raise ValueError("Prediction dataset hash/split must match this held-out test dataset.")
    mode = predictions.get("analysis_mode")
    if mode not in {"automatic", "assisted"}:
        raise ValueError("Predictions must identify automatic or assisted analysis mode.")
    if not isinstance(predictions.get("inference_artifacts"), dict):
        raise ValueError("Prediction inference artifact provenance is missing.")
    provenance, fingerprints = predictions.get("analysis_provenance"), predictions.get("analysis_fingerprints")
    if not isinstance(provenance, dict) or not isinstance(fingerprints, dict) or provenance.keys() != fingerprints.keys():
        raise ValueError("Prediction analysis provenance/fingerprints are missing or inconsistent.")
    if any(not isinstance(dependencies, dict) or dependencies.get("mode") != mode
           or canonical_hash(dependencies)[:24] != fingerprints[possession_id]
           for possession_id, dependencies in provenance.items()):
        raise ValueError("Prediction analysis mode/fingerprint does not match its inference configuration.")
    if any(predictions["inference_artifacts"].get(path) != digest
           for dependencies in provenance.values() for path, digest in dependencies.get("artifacts", {}).items()):
        raise ValueError("Prediction model artifacts disagree with its analysis provenance.")
    if any(frozen["artifacts"].get(path) != digest
           for path, digest in predictions["inference_artifacts"].items()):
        raise ValueError("Every inference model/bundle must match an artifact in the frozen run.")
    events = predictions.get("events")
    if not isinstance(events, list):
        raise ValueError("Predictions must contain an events list.")
    if any(p.get("possession_id") not in provenance for p in events):
        raise ValueError("Every predicted event needs its saved analysis provenance.")
    if mode == "automatic" and any(p.get("origin") != "automatic_rim_approach" for p in events):
        raise ValueError("Automatic evaluation cannot contain human release anchors or unspecified origins.")
    records = json.loads((dataset / "records.json").read_text(encoding="utf-8"))
    parents = {(r["game_id"], r["possession_id"]) for r in records if r.get("split") == "test"}
    if any((p.get("game_id"), p.get("possession_id")) not in parents for p in events):
        raise ValueError("Predictions contain a game/possession outside the test split.")
    gold = [r for r in records if r.get("kind") == "shot" and r.get("split") == "test"]
    if not gold:
        raise ValueError("No held-out test shots. Keep pilot/development results separate.")
    output.mkdir(parents=True, exist_ok=True)
    # Keep the marker next to the freeze, even if another report folder is passed.
    marker = freeze_path.parent / "locked-test-used.json"
    try:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"freeze_hash": frozen["freeze_hash"], "status": "consumed"}, stream)
    except FileExistsError as exc:
        raise ValueError("This frozen run already consumed its locked test. Do not tune/retest it.") from exc
    events = [p for p in events if p.get("type", "shot") == "shot"]
    report = outcome_metrics(gold, events)
    report.update(dataset_hash=manifest["dataset_hash"], freeze_hash=frozen["freeze_hash"],
                  prediction_hash=canonical_hash(predictions), test_used_for_selection=False, analysis_mode=mode,
                  end_to_end_automatic=mode == "automatic",
                  limitations="Pilot counts only; unsupported rare classes and abstentions remain explicit.")
    (output / "test-report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
