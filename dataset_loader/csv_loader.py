"""CSV dataset loading with explicit target-column validation."""

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CsvDataset:
    """A tabular dataset represented as feature mappings and target values."""

    features: list[dict[str, str]]
    targets: list[str]

    def __post_init__(self) -> None:
        if len(self.features) != len(self.targets):
            raise ValueError("features and targets must contain the same number of rows")

    def __len__(self) -> int:
        return len(self.targets)


def load_csv_dataset(path: str | Path, target_column: str) -> CsvDataset:
    """Load a CSV file and separate one named column as the target."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"dataset has no header: {csv_path}")
        if target_column not in fieldnames:
            raise ValueError(
                f"target column {target_column!r} is not present; "
                f"available columns: {', '.join(fieldnames)}"
            )

        features: list[dict[str, str]] = []
        targets: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            if any(value is None for value in row.values()):
                raise ValueError(f"row {row_number} has more fields than the header")
            target = row[target_column]
            if target is None or target == "":
                raise ValueError(f"row {row_number} has an empty target")
            targets.append(target)
            features.append({key: value for key, value in row.items() if key != target_column})

    return CsvDataset(features=features, targets=targets)
