#!/usr/bin/env python3
"""Validate complete retrieval and deterministic QA artifacts for the final grid."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


BASELINES = ("bm25", "dense", "lightmem", "hipporag2", "mem0")
TASKS = {
    "SH-Doc_QA": (100, 1, ("substring_exact_match",)),
    "MH-Doc_QA": (100, 1, ("substring_exact_match",)),
    "FactConsolidation-SH": (100, 1, ("substring_exact_match",)),
    "FactConsolidation-MH": (100, 1, ("substring_exact_match",)),
    "LoCoMo": (1986, 10, ("f1",)),
    "2WikiMultiHopQA": (
        1000,
        1,
        ("passage_recall_at_5", "passage_precision_at_5", "answer_f1"),
    ),
}
EVALUATORS = (
    "Qwen_Qwen3.5-9B",
    "Qwen_Qwen3.5-4B",
    "Qwen_Qwen3.5-2B",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def read_json_object(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"non-object JSON at {path}")
    return value


def validate_retrieval(cell_dir: Path, expected_cases: int, expected_groups: int) -> list[str]:
    errors = []
    try:
        retrieval_rows = read_jsonl(cell_dir / "retrieval.jsonl")
        efficiency_rows = read_jsonl(cell_dir / "efficiency.jsonl")
    except ValueError as error:
        return [str(error)]

    if len(retrieval_rows) != expected_cases:
        errors.append(f"{cell_dir}: expected {expected_cases} retrieval rows, found {len(retrieval_rows)}")
    if len(efficiency_rows) != expected_groups:
        errors.append(f"{cell_dir}: expected {expected_groups} efficiency rows, found {len(efficiency_rows)}")

    case_keys = []
    for row_number, row in enumerate(retrieval_rows, start=1):
        case = row.get("case")
        retrieved = row.get("retrieved")
        if not isinstance(case, dict) or not case.get("case_id"):
            errors.append(f"{cell_dir}: retrieval row {row_number} has no case_id")
        else:
            case_keys.append((row.get("group_id"), case["case_id"]))
        if not isinstance(retrieved, list) or len(retrieved) != 5:
            found = len(retrieved) if isinstance(retrieved, list) else "non-list"
            errors.append(f"{cell_dir}: retrieval row {row_number} expected top-5, found {found}")
    if len(set(case_keys)) != len(case_keys):
        errors.append(f"{cell_dir}: duplicate (group_id, case_id) retrieval rows")

    group_ids = []
    for row_number, row in enumerate(efficiency_rows, start=1):
        group_ids.append(row.get("group_id"))
        if row.get("status") != "completed" or row.get("error") is not None:
            errors.append(
                f"{cell_dir}: efficiency row {row_number} is not completed: "
                f"status={row.get('status')!r}, error={row.get('error')!r}"
            )
    if len(set(group_ids)) != len(group_ids):
        errors.append(f"{cell_dir}: duplicate efficiency group rows")
    return errors


def validate_evaluation(
    evaluation_dir: Path,
    expected_cases: int,
    required_metrics: tuple[str, ...],
) -> list[str]:
    errors = []
    try:
        rows = read_jsonl(evaluation_dir / "predictions.jsonl")
        summary = read_json_object(evaluation_dir / "summary.json")
    except ValueError as error:
        return [str(error)]

    if len(rows) != expected_cases:
        errors.append(f"{evaluation_dir}: expected {expected_cases} predictions, found {len(rows)}")
    case_keys = []
    values: dict[str, list[float]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=1):
        case_keys.append((row.get("group_id"), row.get("case_id")))
        retrieved = row.get("retrieved")
        if not isinstance(retrieved, list) or len(retrieved) != 5:
            found = len(retrieved) if isinstance(retrieved, list) else "non-list"
            errors.append(f"{evaluation_dir}: prediction row {row_number} expected top-5, found {found}")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{evaluation_dir}: prediction row {row_number} has invalid metrics")
            continue
        for metric in required_metrics:
            value = metrics.get(metric)
            if not isinstance(value, (int, float)):
                errors.append(f"{evaluation_dir}: prediction row {row_number} lacks numeric {metric}")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                values[metric].append(float(value))
    if any(group_id is None or case_id is None for group_id, case_id in case_keys):
        errors.append(f"{evaluation_dir}: prediction row has no group_id or case_id")
    if len(set(case_keys)) != len(case_keys):
        errors.append(f"{evaluation_dir}: duplicate (group_id, case_id) prediction rows")

    for metric in required_metrics:
        if metric not in summary:
            errors.append(f"{evaluation_dir}: summary lacks {metric}")
    for metric, metric_values in values.items():
        expected = sum(metric_values) / len(metric_values)
        actual = summary.get(metric)
        if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > 1e-12:
            errors.append(f"{evaluation_dir}: summary value for {metric} does not match predictions")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    experiment_dir = args.output_root / args.experiment_id
    errors = []
    for baseline in BASELINES:
        for task, (expected_cases, expected_groups, required_metrics) in TASKS.items():
            cell_dir = experiment_dir / baseline / task
            errors.extend(validate_retrieval(cell_dir, expected_cases, expected_groups))
            if not args.retrieval_only:
                for evaluator in EVALUATORS:
                    errors.extend(
                        validate_evaluation(
                            cell_dir / "evaluations" / evaluator,
                            expected_cases,
                            required_metrics,
                        )
                    )

    if errors:
        print("Experiment validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    retrieval_cells = len(BASELINES) * len(TASKS)
    if args.retrieval_only:
        print(f"Validated {retrieval_cells} complete retrieval cells in {experiment_dir}")
    else:
        print(
            f"Validated {retrieval_cells} retrieval cells and "
            f"{retrieval_cells * len(EVALUATORS)} evaluation cells in {experiment_dir}"
        )


if __name__ == "__main__":
    main()
