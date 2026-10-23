# Basketball Play Lab: fine-tune yolo11s into a two-class ball/rim detector.
# Output: Drive/PlayLab/models/ball_rim.pt -> copy to artifacts/models/ball_rim.pt locally.
# Runtime: GPU (T4 is fine). Needs a free Roboflow API key saved as the Colab secret
# ROBOFLOW_API_KEY (key icon in the left bar). The key is never printed.
# Your own Label Studio frames (scripts/label_studio_frames.py convert -> zip in Drive)
# are mixed in via OWN_ZIPS; their validation frames are also scored on their own, which
# is the number that matters for your footage. Public-data scores are not.
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

DATASETS = [  # (workspace, project, version); check each license on its Universe page.
    ("dm-vision", "ballv2-hoopv2", 1),        # ~1.7k images, ball + hoop, CC BY 4.0
    ("eagle-eye", "basketball-1zhpe", 1),     # basketball + rim, CC BY 4.0
]
# The PlayLab folder holding your labeled frames (a copied folder may be named "PlayLab (1)").
DRIVE = next((p for p in (Path("/content/drive/MyDrive/PlayLab"), Path("/content/drive/MyDrive/PlayLab (1)"))
              if (p / "my-frames-v1.zip").is_file()), Path("/content/drive/MyDrive/PlayLab"))
# Each new training run needs a NEW RUN_NAME. Run 01: public data only. Run 02: own frames
# mixed in, cut off after 2 of 43 epochs. Run 03: that mix trained to completion (46 epochs).
# Run 04 adds 100 labeled frames from a second gym (St. Mark's vs Dallas, training split).
RUN_NAME = "ball-rim-yolo11s-04"
CONTINUE_FROM = next((p for p in (DRIVE / "runs/ball-rim-yolo11s-03/weights/best.pt",
                                  DRIVE / "runs/ball-rim-yolo11s-02/weights/best.pt",
                                  DRIVE / "runs/ball-rim-yolo11s-01/weights/best.pt") if p.is_file()), None)
# (zip in Drive, train repeat). Repeats weight your footage ~3x against public images.
# Only my-frames-v1 has val frames, so own_val_* stays comparable with run 03.
OWN_ZIPS = [(DRIVE / "my-frames-v1.zip", 3), (DRIVE / "my-frames-v2.zip", 3)]
EPOCHS, IMAGE_SIZE, MAX_HOURS = 80, 960, 1.5  # broadcast balls are ~15px: keep 960
MIN_EPOCHS = 15  # never export a run that stopped before this many epochs
ULTRALYTICS = "ultralytics==8.4.183"          # matches the local vision runtime
CLASSES = ("ball", "rim")                     # names playlab.detectors already maps


def target_class(name):
    """Map a source class name to ball/rim; anything else (players, 'made', ...) is dropped."""
    key = re.sub(r"[\s_\-]+", "", str(name).lower())
    if key in {"ball", "basketball", "sportsball", "basketballball"}:
        return 0
    if key in {"rim", "hoop", "basket", "basketballrim", "basketballhoop", "hooprim"}:
        return 1
    return None


def merge(sources, merged):
    """Remap YOLO label files from several exports into one train/val tree.

    Sources are (slug, root) or (slug, root, train_repeat, own). Validation images of
    `own` sources are also listed in own_val.txt so they can be scored separately.
    """
    counts = {split: [0, 0, 0] for split in ("train", "val")}  # images, balls, rims
    own_val = []
    for source in sources:
        slug, root, repeat, own = (*source, 1, False) if len(source) == 2 else source
        names = yaml.safe_load((root / "data.yaml").read_text())["names"]
        names = names if isinstance(names, list) else [names[i] for i in sorted(names)]
        mapping = {i: target_class(n) for i, n in enumerate(names)}
        print(slug, {n: (CLASSES[mapping[i]] if mapping[i] is not None else "dropped")
                     for i, n in enumerate(names)})
        if {0, 1} - set(mapping.values()):
            print(f"  warning: {slug} lacks a ball or rim class; only its mapped boxes are used")
        for split, out in (("train", "train"), ("valid", "val"), ("test", "val")):
            images = root / split / "images"
            if not images.is_dir():
                continue
            for image in sorted(images.iterdir()):
                label = root / split / "labels" / (image.stem + ".txt")
                rows = []
                for line in (label.read_text().splitlines() if label.is_file() else []):
                    parts = line.split()
                    if len(parts) != 5:  # skip polygon/segment rows rather than misread them
                        continue
                    cls = mapping.get(int(parts[0]))
                    if cls is not None:
                        rows.append(" ".join([str(cls), *parts[1:]]))
                        counts[out][1 + cls] += repeat if out == "train" else 1
                (merged / out / "images").mkdir(parents=True, exist_ok=True)
                (merged / out / "labels").mkdir(parents=True, exist_ok=True)
                for copy in range(repeat if out == "train" else 1):
                    stem = f"{slug}__{image.stem}" + (f"__r{copy}" if copy else "")
                    shutil.copy2(image, merged / out / "images" / (stem + image.suffix))
                    (merged / out / "labels" / (stem + ".txt")).write_text("\n".join(rows))
                    counts[out][0] += 1
                    if own and out == "val":
                        own_val.append(str(merged / out / "images" / (stem + image.suffix)))
    for split, (images, balls, rims) in counts.items():
        print(f"{split}: {images} images, {balls} ball boxes, {rims} rim boxes")
    assert counts["train"][1] and counts["train"][2], "Training data needs both ball and rim boxes."
    assert counts["val"][2], "Validation has no rim boxes; add a dataset with labelled rims."
    data = merged / "data.yaml"
    data.write_text(f"path: {merged}\ntrain: train/images\nval: val/images\nnames:\n  0: ball\n  1: rim\n")
    if own_val:
        (merged / "own_val.txt").write_text("\n".join(own_val) + "\n")
        (merged / "own_val.yaml").write_text(f"path: {merged}\ntrain: train/images\nval: own_val.txt\n"
                                             "names:\n  0: ball\n  1: rim\n")
    return data, counts


# --- Colab run ---
if "google.colab" in sys.modules:
    from google.colab import drive, userdata

    # Remount so files copied into Drive after an earlier mount are visible.
    drive.mount("/content/drive", force_remount=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", ULTRALYTICS, "roboflow"], check=True)
    import torch
    from roboflow import Roboflow
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "Switch to a GPU runtime (Runtime > Change runtime type)."
    print(torch.cuda.get_device_name(0))
    workspace = Path("/content/ball-rim")
    raw, merged = workspace / "raw", workspace / "merged"
    shutil.rmtree(merged, ignore_errors=True)
    rf = Roboflow(api_key=userdata.get("ROBOFLOW_API_KEY"))
    sources = []
    for ws, project, ver in DATASETS:
        location = raw / f"{ws}-{project}-{ver}"
        if not (location / "data.yaml").is_file():
            rf.workspace(ws).project(project).version(ver).download("yolov8", location=str(location))
        sources.append((f"{ws}-{project}-{ver}", location))
    for archive, repeat in OWN_ZIPS:
        assert archive.is_file(), f"Upload {archive.name} to {archive.parent} first (or set OWN_ZIPS = [])."
        location = raw / archive.stem
        shutil.rmtree(location, ignore_errors=True)
        shutil.unpack_archive(archive, location)
        sources.append((archive.stem, location, repeat, True))
    if CONTINUE_FROM:
        assert Path(CONTINUE_FROM).is_file(), f"CONTINUE_FROM not found: {CONTINUE_FROM}"
    data, counts = merge(sources, merged)

    runs = DRIVE / "runs"
    last = runs / RUN_NAME / "weights" / "last.pt"
    # Ultralytics marks a completed run's checkpoint with epoch -1.
    finished = last.is_file() and torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1) == -1
    if last.is_file() and not finished:  # runtime expired mid-run: continue the same candidate
        YOLO(str(last)).train(resume=True)
    elif finished:
        print(f"{RUN_NAME} already finished; re-exporting its best.pt. "
              "To train longer, set CONTINUE_FROM and a new RUN_NAME.")
    else:
        YOLO(str(CONTINUE_FROM) if CONTINUE_FROM else "yolo11s.pt").train(
            data=str(data), epochs=EPOCHS, time=MAX_HOURS, imgsz=IMAGE_SIZE, batch=-1,
            patience=20, project=str(runs), name=RUN_NAME, exist_ok=True, seed=0,
            plots=True, close_mosaic=10)
    # Export only a run that ended on its own (all epochs, early stop or time budget) and
    # trained long enough. A runtime that dies mid-run leaves last.pt with epoch >= 0.
    results_csv = runs / RUN_NAME / "results.csv"
    epochs_done = max(0, len(results_csv.read_text().strip().splitlines()) - 1) if results_csv.is_file() else 0
    finished = last.is_file() and torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1) == -1
    print(f"{RUN_NAME}: {epochs_done} epochs trained, finished={finished}")
    if not finished:
        raise SystemExit("Training was interrupted. Rerun this cell to resume; nothing was exported.")
    if epochs_done < MIN_EPOCHS:
        raise SystemExit(f"Only {epochs_done} epochs before the run ended; not exporting. Set a new RUN_NAME, "
                         f"CONTINUE_FROM = {runs / RUN_NAME / 'weights' / 'best.pt'} and a larger MAX_HOURS.")
    best = runs / RUN_NAME / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(data), imgsz=IMAGE_SIZE, split="val", verbose=False)
    per_class = {CLASSES[i]: round(float(metrics.box.maps[i]), 4) for i in range(len(CLASSES))}
    own = {}
    if (merged / "own_val.yaml").is_file():  # the score that reflects your footage
        mine = YOLO(str(best)).val(data=str(merged / "own_val.yaml"), imgsz=IMAGE_SIZE, split="val", verbose=False)
        own = {"own_val_mAP50": round(float(mine.box.map50), 4),
               "own_val_mAP50_95_per_class": {CLASSES[i]: round(float(mine.box.maps[i]), 4)
                                              for i in range(len(CLASSES))}}
    (DRIVE / "models").mkdir(parents=True, exist_ok=True)
    output = DRIVE / "models" / "ball_rim.pt"
    shutil.copy2(best, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    provenance = {"datasets": DATASETS, "own_zips": [[str(a), r] for a, r in OWN_ZIPS], **own,
                  "counts": counts, "ultralytics": ULTRALYTICS,
                  "base": str(CONTINUE_FROM or "yolo11s.pt"), "run": RUN_NAME, "epochs_trained": epochs_done,
                  "image_size": IMAGE_SIZE, "val_mAP50_95_per_class": per_class,
                  "val_mAP50": round(float(metrics.box.map50), 4), "sha256": digest,
                  "note": "own_val_* is scored on your held-out val-block frames; other scores mix public data."}
    (DRIVE / "models" / "ball_rim.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))
    print(f"Download {output} to artifacts/models/ball_rim.pt in Basketball-Play-Lab.")
