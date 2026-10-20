"""Bounded detector/classifier training, with no test-set model selection."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from playlab.dataset import canonical_hash, file_hash, validate_splits, verify_dataset
from playlab.features import FEATURE_NAMES, feature_vector, geometry_prediction


def code_version() -> dict:
    source = Path(__file__).parent
    digest = canonical_hash({p.name: file_hash(p) for p in sorted(source.glob("*.py"))})
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True,
                                  text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "uncommitted"
    return {"revision": revision, "source_hash": digest}


def _read_rows(source):
    dataset_manifest = None
    if isinstance(source, (str, Path)):
        source = Path(source)
        if source.is_dir():
            dataset_manifest = verify_dataset(source)
            rows = json.loads((source / "records.json").read_text(encoding="utf-8"))
            rows = [r for r in rows if r.get("kind") == "shot"]
        elif source.suffix.lower() == ".csv":
            with source.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        else:
            rows = json.loads(source.read_text(encoding="utf-8"))
    else:
        rows = list(source)
    if not isinstance(rows, list):
        raise ValueError("Features must be a JSON list, CSV rows or an exported dataset.")
    normalized = []
    for row in rows:
        row = dict(row)
        row["reviewed"] = row.get("reviewed") is True or str(row.get("reviewed")).lower() in {"true", "1"}
        if isinstance(row.get("features"), str):
            row["features"] = json.loads(row["features"])
        normalized.append(row)
    return normalized, dataset_manifest


def _validate_classifier_groups(rows, manifest):
    if manifest:
        validate_splits(rows, manifest["games"])
        return
    # Raw feature tables cannot silently opt into same-game blocked splits.
    games, possessions, replays = {}, {}, {}
    for r in rows:
        if r.get("split") not in {"train", "val", "test"} or not r.get("game_id") or not r.get("possession_id"):
            raise ValueError("Feature rows need game_id, possession_id and train/val/test split.")
        keys = [(games, str(r["game_id"])), (possessions, (str(r["game_id"]), str(r["possession_id"])))]
        if r.get("replay_group"):
            keys.append((replays, str(r["replay_group"])))
        for lookup, key in keys:
            if key in lookup and lookup[key] != r["split"]:
                raise ValueError("Feature groups cross splits. Use a validated dataset for declared blocked splits.")
            lookup[key] = r["split"]


def train_classifier(rows, output: str | Path) -> Path:
    """Fit L2 logistic C={0.1,1,10}; tune abstention on validation only.

    A held-out test row is never read for features, results or model selection.
    The fitted imputer/scaler are exclusively trained on training rows.
    """
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    all_rows, manifest = _read_rows(rows)
    reviewed = [r for r in all_rows if r.get("reviewed") is True]
    _validate_classifier_groups(reviewed, manifest)
    development = [r for r in reviewed if r["split"] in {"train", "val"}
                   and r.get("result") in {"made", "missed"}]
    training = [r for r in development if r["split"] == "train"]
    validation = [r for r in development if r["split"] == "val"]
    for split, subset in (("training", training), ("validation", validation)):
        if {r["result"] for r in subset} != {"made", "missed"}:
            raise ValueError(f"Need human-reviewed made and missed shots in {split} data.")
        if split == "training" and any(sum(r["result"] == label for r in subset) < 2 for label in ("made", "missed")):
            raise ValueError("Need at least two training examples of each shot result; this is still only a pilot.")
        if any(not any(name in r.get("features", r) for name in FEATURE_NAMES) for r in subset):
            raise ValueError("Shot features missing. Analyze its possession, match release within 0.3 s, then re-export.")
    x_train = np.asarray([feature_vector(r.get("features", r)) for r in training])
    y_train = np.asarray([r["result"] for r in training])
    x_val = np.asarray([feature_vector(r.get("features", r)) for r in validation])
    y_val = np.asarray([r["result"] for r in validation])
    if not np.isfinite(x_train).any():
        raise ValueError("Training shots have no finite visual features.")
    best = None
    comparisons = []
    for strength in (0.1, 1.0, 10.0):
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(), LogisticRegression(C=strength, random_state=42,
                                                                    max_iter=1000))
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_val)
        prediction = model.classes_[probabilities.argmax(axis=1)]
        scores = probabilities.max(axis=1)
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
            evidence = x_val[:, 5] >= 4
            eligible = np.isfinite(x_val[:, 1]) & (x_val[:, 1] <= 2) & evidence
            emitted = (scores >= threshold) & eligible
            coverage = float(emitted.mean())
            correct = int(((prediction == y_val) & emitted).sum())
            accuracy = correct / int(emitted.sum()) if emitted.any() else None
            report = {"C": strength, "threshold": threshold, "coverage": coverage,
                      "correct": correct, "emitted": int(emitted.sum()), "eligible": len(validation),
                      "accuracy_when_emitted": accuracy}
            comparisons.append(report)
            # Enforce >=50% validation coverage before optimizing selective accuracy.
            key = (coverage >= 0.5, correct / len(validation),
                   accuracy if accuracy is not None else -1, coverage, -strength, -threshold)
            if best is None or key > best[0]:
                best = (key, model, threshold, report)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output is not empty; choose a new run directory.")
    output.mkdir(parents=True, exist_ok=True)
    baseline = [geometry_prediction(r.get("features", r)) for r in validation]
    baseline_emitted = sum(p != "unknown" for p in baseline)
    baseline_correct = sum(p == y for p, y in zip(baseline, y_val))
    baseline_key = (baseline_emitted / len(validation) >= 0.5, baseline_correct / len(validation),
                    baseline_correct / baseline_emitted if baseline_emitted else -1,
                    baseline_emitted / len(validation))
    # Same denominator and evidence pool; keep the simpler baseline on ties.
    selected_method = "geometry" if baseline_key >= best[0][:4] else "classifier"
    provenance = {"model": "l2_logistic", "feature_names": list(FEATURE_NAMES), "seed": 42,
                  "code": code_version(), "dataset_hash": manifest["dataset_hash"] if manifest else None,
                  "development_hash": canonical_hash(development), "train_count": len(training),
                  "validation_count": len(validation), "test_used_for_selection": False,
                  "selected_method": selected_method,
                  "selection": best[3], "comparisons": comparisons,
                  "baseline_validation": {"eligible": len(validation), "emitted": baseline_emitted,
                                          "correct": baseline_correct,
                                          "coverage": baseline_emitted / len(validation)},
                  "limitations": "Small pilot; scores are uncalibrated; no accuracy promise."}
    bundle = {"model": best[1], "feature_names": FEATURE_NAMES, "threshold": best[2],
              "selected_method": selected_method, "provenance": provenance}
    destination = output / "classifier.joblib"
    joblib.dump(bundle, destination)
    (output / "training.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def train_detector(dataset: str | Path, output: str | Path, device: str = "cpu", resume=None,
                   max_hours: float = 2, epochs: int = 40, stop_after_epochs: int | None = None,
                   image_size: int = 960) -> Path:
    """Fine tune a two-class ball/rim YOLO head, never evaluating the test set.

    stop_after_epochs=1 intentionally pauses after a recoverable model save. Resume
    that last.pt to exercise real optimizer/RNG recovery before an overnight run.
    Requires optional vision dependencies; importing this module does not.
    """
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    manifest = verify_dataset(dataset)
    if not 0 < max_hours <= 2 or not 1 <= epochs <= 1000:
        raise ValueError("Choose a positive training limit (<=2 hours, <=1000 epochs).")
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise ValueError("stop_after_epochs must be positive.")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or not 32 <= image_size <= 2048 or image_size % 32:
        raise ValueError("image_size must be a multiple of 32 between 32 and 2048.")
    # Enumerate only verified images. Directory globs allow unmanifested files
    # to enter training/validation, even though the dataset hash is unchanged.
    images = {}
    for split in ("train", "val"):
        images[split] = sorted(dataset / relative for relative in manifest["files"]
                               if Path(relative).parts[:2] == ("images", split)
                               and Path(relative).suffix.lower() == ".png")
        if not images[split]:
            raise ValueError(f"No reviewed {split} images. Export positive boxes and/or explicit negative frames.")
        for image in images[split]:
            label = Path("labels") / split / image.relative_to(dataset / "images" / split).with_suffix(".txt")
            if label.as_posix() not in manifest["files"]:
                raise ValueError(f"Reviewed image has no verified label file: {image.name}")
    provenance = {"dataset_hash": manifest["dataset_hash"], "code": code_version(),
                  "seed": 42, "epochs": epochs, "max_hours": max_hours, "device": device,
                  "image_size": image_size, "image_counts": {key: len(value) for key, value in images.items()},
                  "model": "yolo11n", "classes": ["ball", "rim"], "test_used_for_selection": False}
    previous = output / "training.json"
    elapsed_before = 0.0
    if resume:
        if not previous.exists():
            raise ValueError("Resume requires the original run directory and its training.json.")
        saved = json.loads(previous.read_text(encoding="utf-8"))
        if saved.get("status") == "completed":
            raise ValueError("Completed training checkpoints are finalized. Start a new run; resume an interrupted run.")
        if saved["dataset_hash"] != manifest["dataset_hash"] or saved["code"] != provenance["code"]:
            raise ValueError("Dataset or code changed; start a new run rather than resuming.")
        if saved.get("epochs") != epochs or saved.get("image_size", 960) != image_size:
            raise ValueError("Epoch limit or image size changed; resume the original configuration or start a new run.")
        expected_checkpoint = output / "detector" / "weights" / "last.pt"
        if Path(resume).resolve() != expected_checkpoint:
            raise ValueError("Resume must use this run's detector/weights/last.pt checkpoint.")
        if not Path(resume).is_file() or not Path(str(resume) + ".rng.pt").is_file():
            raise ValueError("Resume needs both last.pt and its saved RNG sidecar.")
        for path, key in ((Path(resume), "last_checkpoint_sha256"),
                          (Path(str(resume) + ".rng.pt"), "rng_sidecar_sha256")):
            if saved.get(key) and file_hash(path) != saved[key]:
                raise ValueError("Checkpoint or RNG sidecar changed; resume the recorded original files.")
        elapsed_before = float(saved.get("elapsed_seconds", 0))
        max_hours = min(max_hours, float(saved["max_hours"]))
        if elapsed_before >= max_hours * 3600:
            raise ValueError("This run exhausted its compute time budget. Stop and review.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Training output is not empty. Resume or choose a new run directory.")
    model_directory = Path(os.environ.get("PLAYLAB_MODEL_DIR", Path(__file__).parents[1] / "artifacts" / "models"))
    model_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(model_directory.parent / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(model_directory.parent / "matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the optional vision dependencies or use the Colab notebook.") from exc
    import random
    import sqlite3

    runtime_versions = {}
    for package in ("torch", "ultralytics", "numpy"):
        try:
            runtime_versions[package] = version(package)
        except PackageNotFoundError:
            runtime_versions[package] = "unavailable"
    if resume and saved.get("runtime_versions") != runtime_versions:
        raise ValueError("Training runtime versions changed; resume the original runtime or start a new run.")
    provenance["runtime_versions"] = runtime_versions
    initial_weights = model_directory / "yolo11n.pt"
    provenance["initial_weights_sha256"] = (saved.get("initial_weights_sha256") if resume else
                                             file_hash(initial_weights) if initial_weights.is_file() else None)
    output.mkdir(parents=True, exist_ok=True)
    # Atomic, bounded attempt registry shared by sibling run directories. A
    # checkpoint resume belongs to its existing attempt, not a new candidate.
    with sqlite3.connect(output.parent / "training-budget.sqlite", timeout=30) as budget:
        budget.execute("CREATE TABLE IF NOT EXISTS attempts (dataset TEXT, run TEXT PRIMARY KEY)")
        budget.execute("BEGIN IMMEDIATE")
        existing = budget.execute("SELECT run FROM attempts WHERE run=?", (str(output),)).fetchone()
        count = budget.execute("SELECT COUNT(*) FROM attempts WHERE dataset=?", (manifest["dataset_hash"],)).fetchone()[0]
        if not resume and not existing and count >= 3:
            raise ValueError("Three detector candidates have already been attempted for this dataset here. Stop and review.")
        if not existing:
            budget.execute("INSERT INTO attempts VALUES (?,?)", (manifest["dataset_hash"], str(output)))
    # The list files live with the run, preserving the immutable dataset bundle.
    # Ultralytics maps each listed /images/ path to its corresponding /labels/.
    for split, paths in images.items():
        (output / f"{split}-images.txt").write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")
    runtime_yaml = output / "train-data.yaml"
    runtime_yaml.write_text(f"path: {json.dumps(str(dataset))}\n"
                            f"train: {json.dumps(str(output / 'train-images.txt'))}\n"
                            f"val: {json.dumps(str(output / 'val-images.txt'))}\n"
                            "names:\n  0: ball\n  1: rim\n", encoding="utf-8")
    provenance.update(status="running", elapsed_seconds=elapsed_before, max_hours=max_hours)
    (output / "training.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    model = YOLO(str(Path(resume).resolve()) if resume else str(model_directory / "yolo11n.pt"))

    class Paused(Exception):
        pass

    saved_epochs = 0
    started = time.perf_counter()

    def after_save(trainer):
        nonlocal saved_epochs
        state = {"python": random.getstate(), "numpy": np.random.get_state(),
                 "torch": torch.get_rng_state(), "epoch": trainer.epoch,
                 "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(state, str(trainer.last) + ".rng.pt")
        provenance["elapsed_seconds"] = elapsed_before + time.perf_counter() - started
        provenance.update(last_saved_epoch=int(trainer.epoch),
                          last_checkpoint_sha256=file_hash(trainer.last),
                          rng_sidecar_sha256=file_hash(str(trainer.last) + ".rng.pt"))
        (output / "training.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        saved_epochs += 1
        if stop_after_epochs is not None and saved_epochs >= stop_after_epochs:
            raise Paused("Intentionally paused after durable checkpoint save.")

    def restore_rng(_trainer):
        if resume:
            state = torch.load(str(resume) + ".rng.pt", map_location="cpu", weights_only=False)
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if state["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda"])
            provenance["rng_restored_from_epoch"] = int(state["epoch"])
            if hasattr(_trainer, "start_epoch"):
                provenance["resume_start_epoch"] = int(_trainer.start_epoch)
            if hasattr(_trainer, "optimizer"):
                provenance["resume_optimizer_state_entries"] = len(_trainer.optimizer.state)

    def cap_epochs(trainer):
        # Ultralytics time mode otherwise increases its epoch target after each
        # epoch. Keep both the requested epoch ceiling and its timed stop active.
        trainer.epochs = min(trainer.epochs, epochs)
        trainer.args.epochs = min(trainer.args.epochs, epochs)

    model.add_callback("on_model_save", after_save)
    model.add_callback("on_train_start", restore_rng)
    model.add_callback("on_train_epoch_start", cap_epochs)
    status = "completed"
    try:
        model.train(data=str(runtime_yaml), epochs=epochs, time=max_hours - elapsed_before / 3600, device=device,
                    project=str(output), name="detector", exist_ok=True, resume=bool(resume),
                    imgsz=image_size, batch=2, workers=0, seed=42, deterministic=True,
                    save=True, save_period=1, plots=False, cache=False, close_mosaic=0, patience=8)
    except Paused:
        status = "paused"
    except BaseException:
        provenance.update(status="interrupted", elapsed_seconds=elapsed_before + time.perf_counter() - started)
        (output / "training.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        raise
    last = output / "detector" / "weights" / "last.pt"
    best = output / "detector" / "weights" / "best.pt"
    if not last.exists():
        raise RuntimeError("Training ended without a recoverable checkpoint. Inspect the cloud run logs.")
    provenance.update(status=status, elapsed_seconds=elapsed_before + time.perf_counter() - started,
                      last_checkpoint=str(last), best_checkpoint=str(best),
                      last_checkpoint_sha256=file_hash(last),
                      optimizer_rng_resume="YOLO checkpoint plus Python/NumPy/Torch RNG sidecar; verify smoke run")
    (output / "training.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return last if status == "paused" else (best if best.exists() else last)
