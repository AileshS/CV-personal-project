"""Fair before/after detector counts and prediction-only dataset exports."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PIL import Image

from playlab.dataset import file_hash, verify_dataset
from playlab.evaluation import detection_metrics, verify_freeze


def predict_dataset(store, dataset: str | Path, output: str | Path, split="test") -> Path:
    """Export saved predictions only; human outcomes never appear in this file."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    manifest = verify_dataset(dataset)
    if split not in {"train", "val", "test"}:
        raise ValueError("Expected train, val or test split.")
    records = json.loads((dataset / "records.json").read_text(encoding="utf-8"))
    parents = {(r["game_id"], r["possession_id"]) for r in records if r["split"] == split}
    events, missing, fingerprints = [], [], {}
    for game_id, possession_id in sorted(parents):
        analysis = store.get("analysis", possession_id)
        if not analysis or analysis.get("stale") or analysis.get("status") != "complete":
            missing.append(possession_id)
            continue
        fingerprints[possession_id] = analysis.get("fingerprint")
        for event in analysis.get("events", []):
            if event.get("type") != "shot" or event.get("source") != "prediction":
                continue
            exported = {key: event.get(key) for key in
                        ("id", "release", "baseline_result", "origin", "release_confirmed", "score", "method")}
            exported.update(game_id=game_id, possession_id=possession_id,
                            result=event.get("predicted_result", event.get("result", "unknown")),
                            defender=event.get("defender"), analysis_fingerprint=analysis.get("fingerprint"))
            events.append(exported)
    payload = {"dataset_hash": manifest["dataset_hash"], "split": split, "events": events,
               "missing_analysis_possessions": missing, "analysis_fingerprints": fingerprints,
               "limitations": "Missing analyses remain missed events in evaluation. Human release anchors are explicit."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    return output


def benchmark_detector(dataset: str | Path, weights: str | Path, output: str | Path, split="val",
                       device="cpu", confidence=0.15, image_size=960, iou_threshold=0.5,
                       baseline_weights=None, freeze=None, config=None) -> dict:
    """Run COCO baseline and fine-tuned model on exactly the same reviewed images.

    Default validation mode never opens test images. Test mode requires frozen
    baseline/fine-tuned weights + JSON inference settings and consumes a dedicated
    detector-test fence, allowing the same frozen run's outcome report separately.
    """
    dataset, output, weights = Path(dataset).resolve(), Path(output).resolve(), Path(weights).resolve()
    manifest = verify_dataset(dataset)
    if split not in {"val", "test"}:
        raise ValueError("Benchmark validation first; split must be val or test.")
    if output.exists():
        raise ValueError("Report exists. Choose a new validation report; do not overwrite test results.")
    model_directory = Path(os.environ.get("PLAYLAB_MODEL_DIR", Path(__file__).parents[1] / "artifacts" / "models"))
    model_directory.mkdir(parents=True, exist_ok=True)
    baseline_weights = Path(baseline_weights or model_directory / "yolo11n.pt").resolve()
    settings = {"confidence": confidence, "image_size": image_size, "iou_threshold": iou_threshold}
    if config:
        settings = json.loads(Path(config).read_text(encoding="utf-8"))
    confidence, image_size, iou_threshold = (settings[k] for k in ("confidence", "image_size", "iou_threshold"))
    if not 0 < confidence <= 1 or not 32 <= image_size <= 2048 or not 0 < iou_threshold <= 1:
        raise ValueError("Invalid confidence, image size or IoU threshold.")
    if not weights.is_file():
        raise FileNotFoundError("Fine-tuned ball/rim weights are required for before/after comparison.")
    if split == "test":
        if not freeze or not config:
            raise ValueError("Locked detector test needs --freeze and a frozen JSON --config.")
        verify_freeze(dataset, freeze, required_artifacts=[weights, baseline_weights, config])
    paths = sorted((dataset / "images" / split).glob("*.png"))
    if not paths:
        raise ValueError(f"No reviewed {split} images, including negative frames, are available.")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(model_directory.parent / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(model_directory.parent / "matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install optional vision dependencies to run detector benchmarking.") from exc
    baseline, fine_tuned = YOLO(str(baseline_weights)), YOLO(str(weights))
    if split == "test":
        try:
            marker = Path(freeze).resolve().parent / "locked-detector-test-used.json"
            with marker.open("x", encoding="utf-8") as stream:
                json.dump({"dataset_hash": manifest["dataset_hash"], "status": "consumed"}, stream)
        except FileExistsError as exc:
            raise ValueError("This frozen run already consumed its detector test. Do not tune/retest.") from exc
    records = json.loads((dataset / "records.json").read_text(encoding="utf-8"))
    gold = [r for r in records if r.get("kind") == "box" and r["split"] == split]
    predictions = {"baseline": [], "fine_tuned": []}
    elapsed = {}
    for name, model in (("baseline", baseline), ("fine_tuned", fine_tuned)):
        started = time.perf_counter()
        for path in paths:
            with Image.open(path) as image:
                width, height = image.size
            result = model.predict(str(path), conf=confidence, imgsz=image_size, device=device, verbose=False)[0]
            for box in result.boxes:
                label = str(result.names[int(box.cls[0])]).lower()
                label = "ball" if label in {"sports ball", "ball", "basketball"} else label
                if label not in {"ball", "rim"}:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                predictions[name].append({"image": path.relative_to(dataset).as_posix(), "label": label,
                                           "box": [x1 / width, y1 / height, x2 / width, y2 / height],
                                           "confidence": float(box.conf[0])})
        elapsed[name] = time.perf_counter() - started
    report = {"dataset_hash": manifest["dataset_hash"], "split": split, "image_count": len(paths),
              "settings": settings, "device": device, "elapsed_seconds": elapsed,
              "weights_hashes": {"baseline": file_hash(baseline_weights), "fine_tuned": file_hash(weights)},
              "baseline": detection_metrics(gold, predictions["baseline"], iou_threshold),
              "fine_tuned": detection_metrics(gold, predictions["fine_tuned"], iou_threshold),
              "limitations": "COCO baseline has no rim class; compare ball counts fairly, report trained rim separately. "
                             "Fixed-threshold precision/recall, not mAP. Runtime includes first-call warmup."}
    report["baseline"]["rim_supported"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report
