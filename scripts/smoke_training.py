"""Real CPU checkpoint smoke on synthetic shapes; never basketball accuracy.

Run with the optional vision runtime, for example:
uv run --extra vision python scripts/smoke_training.py artifacts/smoke/training-01
The output must be new. No reviewed footage, held-out test data or GPU is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from playlab.dataset import canonical_hash, file_hash, verify_dataset
from playlab.features import FEATURE_NAMES
from playlab.training import train_classifier, train_detector


def synthetic_dataset(destination: Path) -> Path:
    """Create disjoint artificial images/labels solely for pipeline execution."""
    destination.mkdir(parents=True)
    files = {}
    for split, count in (("train", 4), ("val", 2)):
        for index in range(count):
            rng = np.random.default_rng(index + (100 if split == "val" else 0))
            pixels = rng.integers(20, 70, size=(96, 96, 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image)
            draw.ellipse((24, 52, 36, 64), fill="orange")
            draw.rectangle((56, 20, 80, 26), outline="white", width=2)
            image_rel = Path("images") / split / f"synthetic-{index}.png"
            label_rel = Path("labels") / split / f"synthetic-{index}.txt"
            (destination / image_rel).parent.mkdir(parents=True, exist_ok=True)
            (destination / label_rel).parent.mkdir(parents=True, exist_ok=True)
            image.save(destination / image_rel)
            (destination / label_rel).write_text(
                "0 0.3125 0.60416667 0.125 0.125\n1 0.70833333 0.23958333 0.25 0.0625\n",
                encoding="utf-8")
            for relative in (image_rel, label_rel):
                files[relative.as_posix()] = file_hash(destination / relative)
    manifest = {"schema": 1, "kind": "synthetic_software_smoke_only", "files": files,
                "limitations": "Artificial shapes, not human-reviewed basketball labels or accuracy evidence."}
    manifest["dataset_hash"] = canonical_hash(manifest)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_dataset(destination)
    return destination


def classifier_rows() -> list[dict]:
    rows = []
    for split in ("train", "val"):
        for index in range(8):
            made = index % 2 == 0
            features = dict(zip(FEATURE_NAMES, [0.1 if made else 1.8, 0.1, 1.0,
                                               0.1 if made else 1.7, 0.8, 8.0]))
            features["geometry_result"] = "made" if made else "missed"
            rows.append({"id": f"{split}-{index}", "game_id": f"synthetic-{split}",
                         "possession_id": f"{split}-p-{index}", "release": 10 + index * 10,
                         "split": split, "reviewed": True, "result": "made" if made else "missed",
                         "features": features, "synthetic": True})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Choose a new smoke output directory; existing results are never overwritten.")
    output.mkdir(parents=True)
    import torch

    torch.set_num_threads(4)
    started = time.perf_counter()
    dataset = synthetic_dataset(output / "synthetic-dataset")
    run = output / "detector-run"
    checkpoint = train_detector(dataset, run, device="cpu", epochs=3, image_size=96,
                                max_hours=0.05, stop_after_epochs=1)
    paused = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sidecar = torch.load(str(checkpoint) + ".rng.pt", map_location="cpu", weights_only=False)
    assert paused["epoch"] == sidecar["epoch"] == 0
    assert paused["optimizer"] and paused["optimizer"]["state"], "Paused checkpoint lacks optimizer state"
    paused_hash = file_hash(checkpoint)
    del paused, sidecar
    final = train_detector(dataset, run, device="cpu", resume=checkpoint, epochs=3,
                           image_size=96, max_hours=0.05)
    provenance = json.loads((run / "training.json").read_text())
    assert provenance["status"] == "completed"
    assert provenance["rng_restored_from_epoch"] == 0
    assert provenance["resume_start_epoch"] == 1
    assert provenance["resume_optimizer_state_entries"] > 0
    final_sidecar = torch.load(str(checkpoint) + ".rng.pt", map_location="cpu", weights_only=False)
    assert final_sidecar["epoch"] == 2
    with (run / "detector" / "results.csv").open() as stream:
        results = list(csv.DictReader(stream))
    epochs = [int(float(row["epoch"])) for row in results]
    assert epochs == [1, 2, 3], epochs
    classifier = train_classifier(classifier_rows(), output / "classifier-run")
    classifier_provenance = json.loads((classifier.parent / "training.json").read_text())
    report = {"kind": "synthetic_software_smoke_only", "device": "cpu", "gpu_used": False,
              "elapsed_seconds": time.perf_counter() - started,
              "dataset_hash": verify_dataset(dataset)["dataset_hash"],
              "detector": {"paused_epoch": 0, "paused_checkpoint_sha256": paused_hash,
                           "resumed_start_epoch": provenance["resume_start_epoch"],
                           "optimizer_state_entries_restored": provenance["resume_optimizer_state_entries"],
                           "rng_restored_from_epoch": provenance["rng_restored_from_epoch"],
                           "completed_epochs": epochs, "model_path": str(final), "model_sha256": file_hash(final)},
              "classifier": {"model_path": str(classifier), "model_sha256": file_hash(classifier),
                             "train_count": classifier_provenance["train_count"],
                             "validation_count": classifier_provenance["validation_count"],
                             "selected_method": classifier_provenance["selected_method"]},
              "test_used_for_selection": False,
              "limitations": "Real libraries/optimizer/RNG executed on artificial data. No basketball accuracy or GPU recovery claim."}
    (output / "smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
