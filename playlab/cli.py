"""Identical dataset/model commands on local CPU or a cloud GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from playlab.store import Store


def parser():
    p = argparse.ArgumentParser(description="Basketball Play Lab: review, train, test")
    p.add_argument("--data", default=os.environ.get("PLAYLAB_DATA", "data"), help="Workspace directory")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    imp = sub.add_parser("import", help="Import a local video or YouTube URL")
    imp.add_argument("source")
    imp.add_argument("--name")
    imp.add_argument("--queue", action="store_true")
    scan = sub.add_parser("index")
    scan.add_argument("game_id")
    scan.add_argument("--queue", action="store_true")
    split = sub.add_parser("split-game", help="Assign a whole game to a dataset partition")
    split.add_argument("game_id")
    split.add_argument("split", choices=["train", "val", "test"])
    block = sub.add_parser("split-blocks", help="Declare same-game partitions with exclusion gaps")
    block.add_argument("game_id")
    block.add_argument("blocks_json", help="JSON list of {start,end,split} continuous blocks")
    add = sub.add_parser("possession")
    add.add_argument("game_id")
    add.add_argument("start", type=float)
    add.add_argument("end", type=float)
    add.add_argument("--split", choices=["unassigned", "train", "val", "test"], default="unassigned")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("possession_id")
    analyze.add_argument("--device", default="cpu")
    analyze.add_argument("--weights")
    analyze.add_argument("--model-bundle")
    analyze.add_argument("--mode", choices=["automatic", "assisted"], default="assisted",
                         help="Automatic ignores all human shot anchors; assisted uses release anchors")
    analyze.add_argument("--queue", action="store_true")
    worker = sub.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--recover", action="store_true", help="Only after stopping the previous worker")
    sub.add_parser("status")
    resume = sub.add_parser("resume")
    resume.add_argument("job_id")
    demo = sub.add_parser("fixture", help="Own synthetic media for mechanics, not basketball accuracy")
    demo.add_argument("--seconds", type=float, default=9)
    exp = sub.add_parser("export")
    exp.add_argument("destination")
    batch = sub.add_parser("prepare-label-batch", help="Sample split-safe frames and unreviewed suggestions")
    batch.add_argument("destination")
    batch.add_argument("--frames", type=int, default=300)
    batch.add_argument("--possessions", type=int, default=50, help="Review items, not verified possessions")
    batch.add_argument("--split", dest="splits", action="append", choices=["train", "val", "test"])
    batch.add_argument("--game", dest="game_ids", action="append")
    batch.add_argument("--model-assisted", action="store_true")
    batch.add_argument("--weights")
    batch.add_argument("--device", default="cpu")
    batch.add_argument("--allow-review-windows", action="store_true")
    batch.add_argument("--window-seconds", type=float, default=30)
    train = sub.add_parser("train-detector")
    train.add_argument("dataset")
    train.add_argument("output")
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume")
    train.add_argument("--max-hours", type=float, default=2)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--image-size", type=int, default=960)
    cls = sub.add_parser("train-classifier")
    cls.add_argument("rows")
    cls.add_argument("output")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("dataset")
    freeze.add_argument("output")
    freeze.add_argument("artifacts", nargs="+")
    ev = sub.add_parser("evaluate")
    ev.add_argument("dataset")
    ev.add_argument("predictions")
    ev.add_argument("output")
    ev.add_argument("--freeze", help="Path to freeze.json when it is outside the report directory")
    sub.add_parser("notify-test", help="Send a test only to your configured PLAYLAB_NTFY_URL")
    attention = sub.add_parser("attention")
    attention.add_argument("--resolve")
    return p


def execute(args):
    store = Store(args.data)
    command = args.command
    if command == "init":
        return {"workspace": str(store.root), "status": "ready"}
    if command == "fixture":
        from playlab.media import import_game, make_fixture
        path = make_fixture(store.root / "fixtures" / "synthetic.mp4", args.seconds)
        game = import_game(store, path, "Synthetic mechanics fixture")
        game = store.save("game", {**game, "synthetic": True})
        return game
    if command == "import":
        if args.queue:
            return {"job_id": store.enqueue("import", {"source": args.source, "name": args.name})}
        from playlab.media import import_game
        return import_game(store, args.source, args.name)
    if command == "index":
        if args.queue:
            return {"job_id": store.enqueue("index", {"game_id": args.game_id})}
        from playlab.segmentation import propose_possessions
        return propose_possessions(store, args.game_id)
    if command == "split-game":
        from playlab.domain import assign_game_split
        assign_game_split(store, args.game_id, args.split)
        return store.get("game", args.game_id)
    if command == "split-blocks":
        from playlab.domain import assign_split_blocks
        return assign_split_blocks(store, args.game_id, json.loads(Path(args.blocks_json).read_text()))
    if command == "possession":
        from playlab.domain import save_possession
        return save_possession(store, args.game_id, args.start, args.end, reviewed=True, split=args.split)
    if command == "analyze":
        payload = {"possession_id": args.possession_id, "device": args.device, "weights": args.weights,
                   "model_bundle": args.model_bundle, "mode": args.mode}
        if args.queue:
            return {"job_id": store.enqueue("analyze", payload)}
        from playlab.analysis import analyze_possession
        return analyze_possession(store, **payload)
    if command == "worker":
        from playlab.worker import run_worker
        if args.recover:
            # User explicitly confirms the old worker is stopped. Never auto-recover active work.
            lock = store.root / ".worker.lock"
            if lock.exists():
                lock.unlink()
            store.recover_interrupted()
        run_worker(store, once=args.once)
        return {"status": "worker exited"}
    if command == "status":
        return {"games": store.list("game"), "jobs": store.jobs(),
                "attention": store.list("attention", resolved=False)}
    if command == "resume":
        store.resume_job(args.job_id)
        return {"job_id": args.job_id, "status": "queued"}
    if command == "export":
        from playlab.dataset import export_dataset
        return {"dataset": str(export_dataset(store, args.destination))}
    if command == "prepare-label-batch":
        from playlab.labeling import prepare_label_batch
        path = prepare_label_batch(store, args.destination, frame_count=args.frames,
                                   possession_count=args.possessions, splits=args.splits or ("train", "val"),
                                   game_ids=args.game_ids, model_assisted=args.model_assisted,
                                   weights=args.weights, device=args.device,
                                   allow_review_windows=args.allow_review_windows, window_seconds=args.window_seconds)
        return {"batch": str(path), "reviewed": False}
    if command == "train-detector":
        from playlab.training import train_detector
        path = train_detector(args.dataset, args.output, device=args.device, resume=args.resume,
                              max_hours=args.max_hours, epochs=args.epochs, image_size=args.image_size)
        return {"run": str(path)}
    if command == "train-classifier":
        from playlab.training import train_classifier
        return {"bundle": str(train_classifier(args.rows, args.output))}
    if command == "freeze":
        from playlab.evaluation import freeze_run
        return {"freeze": str(freeze_run(args.dataset, args.artifacts, args.output))}
    if command == "evaluate":
        from playlab.evaluation import evaluate_locked
        return evaluate_locked(args.dataset, args.predictions, args.output, freeze=args.freeze)
    if command == "notify-test":
        from playlab.attention import notify
        return notify("Play Lab notification test. Confirm receipt in Codex; no footage or labels included.")
    if command == "attention":
        from playlab.attention import resolve_attention
        if args.resolve:
            resolve_attention(store, args.resolve)
        return store.list("attention", resolved=False)
    raise ValueError("Unsupported command")


def main():
    args = parser().parse_args()
    try:
        result = execute(args)
        print(json.dumps(result, indent=2, default=str, allow_nan=False))
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser().exit(2, f"Play Lab: {error}\n")


if __name__ == "__main__":
    main()
