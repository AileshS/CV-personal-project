"""Prepare source-partitioned frames and optional unreviewed neural box suggestions."""
from __future__ import annotations

import argparse

from playlab.labeling import prepare_label_batch
from playlab.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("--data", default="data")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--possessions", type=int, default=50, help="Review items; windows are not claimed possessions")
    parser.add_argument("--split", dest="splits", action="append", choices=["train", "val", "test"])
    parser.add_argument("--game", dest="game_ids", action="append")
    parser.add_argument("--model-assisted", action="store_true")
    parser.add_argument("--weights", help="Optional fine-tuned ball/rim weights; COCO baseline has no rim class")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-review-windows", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=30)
    args = parser.parse_args()
    print(prepare_label_batch(Store(args.data), args.destination, frame_count=args.frames,
                              possession_count=args.possessions, splits=args.splits or ("train", "val"),
                              game_ids=args.game_ids, model_assisted=args.model_assisted, weights=args.weights,
                              device=args.device, allow_review_windows=args.allow_review_windows,
                              window_seconds=args.window_seconds))


if __name__ == "__main__":
    main()
