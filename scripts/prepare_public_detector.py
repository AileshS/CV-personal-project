"""Prepare a pinned public ball/rim pilot; never touch the local review Store.

Publisher: Eagle Eye Basketball v1 (CC BY 4.0), mirrored in Swish-Vision.
Only publisher train/valid are downloaded. Published labels are not local gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from playlab.dataset import canonical_hash, file_hash, verify_dataset

REPO = "ahmedsilat44/Swish-Vision"
REVISION = "b22265d1a62bbfd5cc1c0d3aea63be38950869a4"
SOURCE = "https://universe.roboflow.com/eagle-eye/basketball-1zhpe/dataset/1"


def fetch(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Basketball-Play-Lab"})
            with urllib.request.urlopen(request, timeout=40) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def original(path):
    return Path(path).name.split(".rf.")[0]


def download(raw, train_count=300, val_count=60):
    if train_count < 1 or val_count < 1:
        raise ValueError("Choose positive train and validation sample counts")
    raw = Path(raw).resolve()
    raw.mkdir(parents=True, exist_ok=True)
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("Incomplete source tree")
    entries = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    groups = {split: {original(p) for p in entries if p.startswith(f"Basketball-1/{split}/images/")}
              for split in ("train", "valid", "test")}
    selection = []
    for split, count in (("train", train_count), ("valid", val_count)):
        blocked = groups["test"] | (groups["valid"] if split == "train" else set())
        candidates = sorted((p for p in entries if p.startswith(f"Basketball-1/{split}/images/")
                             and Path(p).suffix.lower() in {".jpg", ".png"} and original(p) not in blocked),
                            key=lambda p: hashlib.sha256(p.encode()).hexdigest())
        seen = set()
        for p in candidates:
            if original(p) in seen:
                continue
            seen.add(original(p))
            selection.extend([p, (Path(p).parent.parent / "labels" / (Path(p).stem + ".txt")).as_posix()])
            if len(seen) == count:
                break
        if len(seen) < count:
            raise ValueError("Insufficient disjoint source images")
    selection += [f"Basketball-1/{name}" for name in ("data.yaml", "README.dataset.txt", "README.roboflow.txt")]

    def pull(relative):
        entry = entries[relative]
        destination = raw / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = destination.read_bytes() if destination.is_file() else fetch(
            f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{relative}")
        digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if digest != entry["sha"]:
            raise ValueError(f"Source blob hash mismatch: {relative}")
        if not destination.exists():
            destination.write_bytes(content)
        return relative

    with ThreadPoolExecutor(max_workers=10) as pool:
        paths = list(pool.map(pull, selection))
    report = {"repository": REPO, "revision": REVISION, "source": SOURCE,
              "license": "CC BY 4.0", "paths": paths, "test_downloaded": False}
    (raw / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def convert(raw, destination):
    raw, destination = Path(raw).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Choose a new dataset version; destination exists")
    selection = json.loads((raw / "selection.json").read_text())
    if selection["revision"] != REVISION:
        raise ValueError("Unexpected public dataset revision")
    config = (raw / "Basketball-1/data.yaml").read_text()
    if "- basketball\n- rim\n- sports ball\n" not in config.replace("\r\n", "\n"):
        raise ValueError("Publisher class definitions changed")
    if "License: CC BY 4.0" not in (raw / "Basketball-1/README.dataset.txt").read_text():
        raise ValueError("Publisher license missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".public-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        files, records, seen = {}, [], set()
        counts = {"train": {"images": 0, "ball": 0, "rim": 0}, "val": {"images": 0, "ball": 0, "rim": 0}}
        skipped = {"ambiguous_sports_ball": 0, "empty_labels": 0, "duplicate_pixels": 0}
        # Validation gets priority when exact decoded duplicates cross source splits.
        paths = sorted(selection["paths"], key=lambda p: ("/valid/" not in p, p))
        for relative in paths:
            if "/images/" not in relative:
                continue
            path = Path(relative)
            source_split = path.parts[1]
            if source_split not in {"train", "valid"}:
                raise ValueError("Only external train/valid are permitted")
            split = "val" if source_split == "valid" else "train"
            labels = raw / path.parent.parent / "labels" / (path.stem + ".txt")
            parsed = []
            for line in labels.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError("Expected YOLO class and four box coordinates")
                category = int(fields[0])
                x, y, width, height = map(float, fields[1:])
                if (category not in {0, 1, 2} or not all(math.isfinite(v) for v in (x, y, width, height))
                        or width <= 0 or height <= 0):
                    raise ValueError("Invalid source annotation")
                box = [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
                if any(v < -1e-6 or v > 1 + 1e-6 for v in box):
                    raise ValueError("Source box exceeds image")
                parsed.append((category, [max(0., min(1., v)) for v in box]))
            if any(category == 2 for category, _ in parsed):
                skipped["ambiguous_sports_ball"] += 1
                continue
            if not parsed:
                skipped["empty_labels"] += 1
                continue
            with Image.open(raw / relative) as source:
                image = source.convert("RGB")
            pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if pixels in seen:
                skipped["duplicate_pixels"] += 1
                continue
            seen.add(pixels)
            identifier = hashlib.sha256(relative.encode()).hexdigest()[:24]
            image_rel = f"images/{split}/{identifier}.png"
            label_rel = f"labels/{split}/{identifier}.txt"
            for name in (image_rel, label_rel):
                (staging / name).parent.mkdir(parents=True, exist_ok=True)
            image.save(staging / image_rel)
            lines = []
            for index, (category, box) in enumerate(parsed):
                x1, y1, x2, y2 = box
                label = "ball" if category == 0 else "rim"
                lines.append(f"{category} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}")
                counts[split][label] += 1
                records.append({"id": f"{identifier}-{index}", "kind": "box", "split": split,
                                "image": image_rel, "label": label, "box": box,
                                "source": "external_publisher_labels", "reviewed": False,
                                "source_image": relative})
            (staging / label_rel).write_text("\n".join(lines) + "\n")
            for name in (image_rel, label_rel):
                files[name] = file_hash(staging / name)
            counts[split]["images"] += 1
        if (not counts["train"]["ball"] or not counts["train"]["rim"] or not counts["val"]["ball"]):
            raise ValueError("Need positive ball/rim training examples and ball validation in both external splits")
        provenance = {**selection, "counts": counts, "skipped": skipped,
                      "limitations": ["Publisher labels, not locally verified human gold.",
                                      "416px source preprocessing; may not transfer to small balls in the user's game.",
                                      "Source game/replay grouping unavailable; source splits are not proof of independence.",
                                      "No verified negative frames, shot outcomes, possession or defender labels.",
                                      "Hoop/rim annotation extent must be checked before geometry calibration."]}
        if not counts["val"]["rim"]:
            provenance["limitations"].append(
                "Publisher validation has no rim labels: rim recall and rim model selection are unsupported.")
        texts = {"records.json": json.dumps(records, indent=2),
                 "provenance.json": json.dumps(provenance, indent=2),
                 "data.yaml": "train: images/train\nval: images/val\nnames:\n  0: ball\n  1: rim\n",
                 "ATTRIBUTION.txt": f"Eagle Eye Basketball v1. CC BY 4.0. {SOURCE}\nMirror: {REPO}@{REVISION}\n"
                                    "Converted to PNG; ambiguous sports-ball and empty frames excluded; no test downloaded.\n"
                                    "https://creativecommons.org/licenses/by/4.0/\n"}
        for name, content in texts.items():
            (staging / name).write_text(content, encoding="utf-8")
            files[name] = file_hash(staging / name)
        manifest = {"schema": 1, "kind": "external_labeled_detection", "games": [],
                    "record_count": len(records), "files": files, "provenance": provenance,
                    "split_policy": "publisher_train_valid_original_names_and_exact_pixels_disjoint_no_test"}
        manifest["dataset_hash"] = canonical_hash(manifest)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        verify_dataset(staging)
        staging.rename(destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("--raw", default="artifacts/external/swish-vision")
    parser.add_argument("--train-count", type=int, default=300)
    parser.add_argument("--val-count", type=int, default=60)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not args.offline:
        download(args.raw, args.train_count, args.val_count)
    manifest = convert(args.raw, args.destination)
    print(json.dumps({"dataset_hash": manifest["dataset_hash"], "counts": manifest["provenance"]["counts"],
                      "limitations": manifest["provenance"]["limitations"]}, indent=2))
