"""Usage: uv run python scripts/benchmark_detector.py DATASET WEIGHTS REPORT.json"""
import argparse
import json

from playlab.benchmark import benchmark_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("weights")
    parser.add_argument("output")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline-weights")
    parser.add_argument("--freeze")
    parser.add_argument("--config", help="JSON confidence,image_size,iou_threshold; freeze this for test")
    args = parser.parse_args()
    print(json.dumps(benchmark_detector(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
