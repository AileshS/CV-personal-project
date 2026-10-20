"""Export saved analysis predictions, excluding human outcomes, for evaluation."""
import argparse

from playlab.benchmark import predict_dataset
from playlab.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--data", default="data")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    print(predict_dataset(Store(args.data), args.dataset, args.output, args.split))


if __name__ == "__main__":
    main()
