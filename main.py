"""Command-line entry point for running the baseline experiment."""

import argparse

from baseline import MajorityClassifier
from dataset_loader import load_csv_dataset
from utils import accuracy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="path to a CSV dataset")
    parser.add_argument("--target-column", required=True, help="CSV column used as the target")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_csv_dataset(args.dataset, args.target_column)
    predictions = MajorityClassifier().fit(dataset.features, dataset.targets).predict(dataset.features)
    print(f"rows={len(dataset)}")
    print(f"accuracy={accuracy(dataset.targets, predictions):.6f}")


if __name__ == "__main__":
    main()
