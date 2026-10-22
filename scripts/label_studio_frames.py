"""Pull frames from your own game for Label Studio, then turn the labels into a YOLO dataset.

export: scans the game's declared train/val blocks (never test, 10 s guards), keeps a
        mix of "ball near rim" frames and evenly spread frames, and pre-labels them
        with the current ball/rim model so you mostly correct rather than draw.
serve:  hosts the exported images on http://localhost:8081 for Label Studio.
convert: turns a Label Studio JSON export into train/valid YOLO folders plus a zip
        for the Colab training cell. Frames you submitted with no boxes become
        negatives (that teaches the model shoes and floor spots are not balls).
Pre-labels are unreviewed suggestions; only submitted annotations become labels.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("ball", "rim")
GUARD_SECONDS = 10.0
LABEL_CONFIG = """<View>
  <Header value="Box every visible basketball and rim. Submit with no boxes if neither is visible."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image">
    <Label value="ball" background="#f5a623" hotkey="1"/>
    <Label value="rim" background="#00b3d6" hotkey="2"/>
  </RectangleLabels>
</View>
"""


def allowed_windows(game, splits=("train", "val"), guard=GUARD_SECONDS):
    """Declared non-test blocks shrunk by guards; whole-game splits use the full source."""
    blocks = game.get("split_blocks") or []
    if blocks:
        return [(float(b["start"]) + guard, float(b["end"]) - guard, b["split"]) for b in blocks
                if b["split"] in splits and float(b["end"]) - float(b["start"]) > 2 * guard]
    if game.get("split") in splits:
        return [(0.0, float(game["duration"]), game["split"])]
    raise ValueError("Assign this game's train/val split (or split blocks) before exporting frames.")


def near_rim(balls, rims, ball_conf=.1, rim_conf=.25, reach=3.0):
    """Best ball confidence within `reach` rim widths of a rim, or 0."""
    best = 0.0
    for rim in (r for r in rims if r["confidence"] >= rim_conf):
        rx1, ry1, rx2, ry2 = rim["bbox"]
        width, cx, cy = rx2 - rx1, (rx1 + rx2) / 2, (ry1 + ry2) / 2
        for ball in (b for b in balls if b["confidence"] >= ball_conf):
            bx1, by1, bx2, by2 = ball["bbox"]
            if width > 0 and ((bx1 + bx2) / 2 - cx) ** 2 + ((by1 + by2) / 2 - cy) ** 2 <= (reach * width) ** 2:
                best = max(best, ball["confidence"])
    return best


def choose_times(scan, count, shot_share=.5, min_gap=3.0):
    """Half the frames where a ball is near a rim (strongest first), the rest spread evenly."""
    chosen = []

    def free(t):
        return all(abs(t - c) >= min_gap for c in chosen)

    for row in sorted((r for r in scan if r["near_rim"] > 0), key=lambda r: -r["near_rim"]):
        if len(chosen) >= round(count * shot_share):
            break
        if free(row["time"]):
            chosen.append(row["time"])
    rest = [r["time"] for r in sorted(scan, key=lambda r: r["time"]) if free(r["time"])]
    need = count - len(chosen)
    if need > 0 and rest:
        step = len(rest) / need
        for i in range(min(need, len(rest))):
            t = rest[int(i * step)]
            if free(t):
                chosen.append(t)
    return sorted(chosen)


def prediction(detections, width, height, min_conf=.35):
    """Label Studio pre-annotation (percent coordinates) from detector boxes."""
    result = []
    for key, label in (("balls", "ball"), ("rims", "rim")):
        for i, box in enumerate(b for b in detections[key] if b["confidence"] >= min_conf):
            x1, y1, x2, y2 = box["bbox"]
            result.append({"id": f"{label}{i}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                           "original_width": width, "original_height": height, "score": round(box["confidence"], 3),
                           "value": {"x": 100 * x1 / width, "y": 100 * y1 / height, "width": 100 * (x2 - x1) / width,
                                     "height": 100 * (y2 - y1) / height, "rotation": 0,
                                     "rectanglelabels": [label]}})
    return result


def _detect(model, frame):
    import cv2

    result = model.predict(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), conf=.05, imgsz=960, verbose=False)[0]
    output = {"balls": [], "rims": []}
    for box in result.boxes:
        name = str(result.names[int(box.cls[0])]).lower()
        key = "balls" if name == "ball" else "rims" if name == "rim" else None
        if key:
            output[key].append({"bbox": [float(v) for v in box.xyxy[0].tolist()], "confidence": float(box.conf[0])})
    return output


def export(args):
    sys.path.insert(0, str(ROOT))
    from PIL import Image
    from ultralytics import YOLO

    from playlab.analysis import default_ball_rim_weights
    from playlab.dataset import file_hash
    from playlab.media import frame_at, iter_frames
    from playlab.store import Store

    store = Store(args.data)
    games = store.list("game")
    game = next((g for g in games if g["id"].startswith(args.game or "")), None) if args.game or len(games) == 1 else None
    if game is None:
        raise SystemExit("Pass --game with one of: " + ", ".join(f"{g['id'][:8]} ({g['name']})" for g in games))
    weights = Path(args.weights) if args.weights else default_ball_rim_weights()
    if not weights or not Path(weights).is_file():
        raise SystemExit("Ball/rim weights are needed for pre-labels: artifacts/models/ball_rim.pt or --weights.")
    model = YOLO(str(weights))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new folder so earlier labels are never overwritten.")
    (out / "images").mkdir(parents=True)
    counts = {"train": args.train, "val": args.val}
    tasks, chosen_all = [], []
    for start, end, split in allowed_windows(game):
        quota = counts[split]
        if quota <= 0:
            continue
        scan = []
        print(f"Scanning {split} {start:.0f}-{end:.0f}s every {args.scan_every:g}s…", flush=True)
        for at, frame in iter_frames(game["source_path"], start=start, end=end,
                                     sample_fps=1 / args.scan_every, max_width=1280):
            found = _detect(model, frame)
            scan.append({"time": float(at), "near_rim": near_rim(found["balls"], found["rims"])})
            if len(scan) % 100 == 0:
                print(f"  {split}: scanned {len(scan)} frames (at {at:.0f}s)", flush=True)
        chosen = choose_times(scan, quota)
        counts[split] = quota - len(chosen)  # several blocks of one split share the quota
        shotlike = sum(1 for r in scan if r["time"] in chosen and r["near_rim"] > 0)
        print(f"  {split}: chose {len(chosen)} frames ({shotlike} with a ball near a rim)", flush=True)
        for at in chosen:
            frame = frame_at(game["source_path"], at)
            name = f"{split}-{at:08.2f}.jpg"
            Image.fromarray(frame).save(out / "images" / name, quality=92)
            height, width = frame.shape[:2]
            tasks.append({"data": {"image": f"{args.base_url.rstrip('/')}/images/{name}", "split": split,
                                   "source_seconds": round(at, 2)},
                          "predictions": [{"model_version": weights.name, "result":
                                           prediction(_detect(model, frame), width, height)}]})
            chosen_all.append({"file": name, "split": split, "time": at})
    (out / "tasks.json").write_text(json.dumps(tasks, indent=1))
    (out / "label_config.xml").write_text(LABEL_CONFIG)
    (out / "manifest.json").write_text(json.dumps({
        "game_id": game["id"], "source_path": game["source_path"], "source_url": game.get("source_url"),
        "split_blocks": game.get("split_blocks"), "guard_seconds": GUARD_SECONDS,
        "prelabel_weights": str(weights), "prelabel_weights_sha256": file_hash(weights),
        "frames": chosen_all, "note": "Pre-labels are unreviewed model suggestions, not labels."}, indent=1))
    print(f"Wrote {len(tasks)} tasks to {out}")


def _image_name(task):
    return Path(unquote(urlparse(task["data"]["image"]).path)).name


def convert(args):
    """Label Studio JSON export -> YOLO train/valid folders, data.yaml and a zip."""
    export_rows = json.loads(Path(args.export_json).read_text(encoding="utf-8"))
    images, out = Path(args.images), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new dataset folder.")
    counts = {split: {"images": 0, "negatives": 0, "ball": 0, "rim": 0, "skipped": 0} for split in ("train", "valid")}
    for task in export_rows:
        name = _image_name(task)
        split = "valid" if name.startswith("val-") else "train" if name.startswith("train-") else None
        if split is None:
            raise SystemExit(f"Unexpected image name {name}: frames must come from the export command.")
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not done:  # unlabeled or skipped frames are never treated as negatives
            counts[split]["skipped"] += 1
            continue
        annotation = max(done, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        rows = []
        for item in annotation.get("result", []):
            if item.get("type") != "rectanglelabels":
                continue
            value = item["value"]
            label = value["rectanglelabels"][0].lower()
            if label not in CLASSES:
                continue
            if abs(value.get("rotation", 0)) > 1e-6:
                raise SystemExit(f"{name}: rotated boxes are not supported; redraw it without rotation.")
            x, y, w, h = (max(0., min(100., value[k])) for k in ("x", "y", "width", "height"))
            w, h = min(w, 100 - x), min(h, 100 - y)
            if w <= 0 or h <= 0:
                continue
            rows.append(f"{CLASSES.index(label)} {(x + w / 2) / 100:.6f} {(y + h / 2) / 100:.6f} "
                        f"{w / 100:.6f} {h / 100:.6f}")
            counts[split][label] += 1
        for folder in ("images", "labels"):
            (out / split / folder).mkdir(parents=True, exist_ok=True)
        shutil.copy2(images / name, out / split / "images" / name)
        (out / split / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(rows))
        counts[split]["images"] += 1
        counts[split]["negatives"] += not rows
    (out / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\nnc: 2\nnames:\n- ball\n- rim\n")
    (out / "README.txt").write_text("Human-reviewed Label Studio frames from your own game (train/val blocks only).\n"
                                    + json.dumps(counts, indent=1))
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    print(json.dumps(counts, indent=1))
    print(f"Dataset: {out}\nUpload for Colab: {archive}")


def serve(args):
    """Serve the export folder to Label Studio (localhost only, with CORS for its image viewer)."""
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=args.folder))
    print(f"Serving {args.folder} at http://localhost:{args.port}/ (Ctrl+C to stop)", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("export", help="Pull and pre-label frames for Label Studio")
    one.add_argument("--out", default=str(ROOT / "artifacts/labeling/my-frames-v1"))
    one.add_argument("--data", default=str(ROOT / "data"))
    one.add_argument("--game", help="Game id prefix (needed only with several games)")
    one.add_argument("--weights", help="Ball/rim weights for pre-labels (default: artifacts/models/ball_rim.pt)")
    one.add_argument("--train", type=int, default=300)
    one.add_argument("--val", type=int, default=75)
    one.add_argument("--scan-every", type=float, default=2.0, help="Seconds between scanned frames")
    one.add_argument("--base-url", default="http://localhost:8081", help="Where the image folder is served")
    two = commands.add_parser("convert", help="Label Studio JSON export -> YOLO dataset zip")
    two.add_argument("export_json")
    two.add_argument("--images", default=str(ROOT / "artifacts/labeling/my-frames-v1/images"))
    two.add_argument("--out", default=str(ROOT / "artifacts/datasets/my-frames-v1"))
    three = commands.add_parser("serve", help="Serve exported images for Label Studio")
    three.add_argument("--folder", default=str(ROOT / "artifacts/labeling/my-frames-v1"))
    three.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    {"export": export, "convert": convert, "serve": serve}[args.command](args)


if __name__ == "__main__":
    main()
