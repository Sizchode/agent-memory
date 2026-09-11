"""Summarize deterministic per-question failures from completed benchmark artifacts."""

import argparse
import json
from pathlib import Path
from typing import Any


LOCOMO_CATEGORY_LABELS = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-evaluators", type=int, default=3)
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def case_key(record: dict[str, Any]) -> tuple[str, str]:
    case_id = record.get("case_id") or record["case"]["case_id"]
    return str(record["group_id"]), str(case_id)


def primary_metric(metrics: dict[str, float]) -> str:
    for name in ("substring_exact_match", "f1", "answer_f1"):
        if name in metrics:
            return name
    raise ValueError(f"no benchmark answer metric in {sorted(metrics)}")


def question_record(
    key: tuple[str, str],
    prediction: dict[str, Any],
    retrieval: dict[tuple[str, str], dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    case = retrieval[key]["case"]
    return {
        "group_id": key[0],
        "case_id": key[1],
        "question": case["question"],
        "answers": case["answers"],
        "prediction": prediction["prediction"],
        "score": prediction["metrics"][metric],
    }


def consistent_failure_record(
    key: tuple[str, str],
    predictions: dict[str, dict[tuple[str, str], dict[str, Any]]],
    retrieval: dict[tuple[str, str], dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    case = retrieval[key]["case"]
    return {
        "group_id": key[0],
        "case_id": key[1],
        "question": case["question"],
        "answers": case["answers"],
        "retrieved": retrieval[key]["retrieved"],
        "evaluations": {
            evaluator: {
                "prediction": records[key]["prediction"],
                "score": records[key]["metrics"][metric],
            }
            for evaluator, records in predictions.items()
        },
    }


def benchmark_category_scores(
    task_dir: Path,
    retrieval: dict[tuple[str, str], dict[str, Any]],
    predictions: dict[str, dict[tuple[str, str], dict[str, Any]]],
    metric: str,
) -> dict[str, Any]:
    """Aggregate only categories explicitly supplied by the benchmark data."""

    categories: dict[int, list[tuple[str, str]]] = {}
    for key, record in retrieval.items():
        category = record["case"].get("category")
        if category is not None:
            categories.setdefault(int(category), []).append(key)
    if not categories:
        return {}

    labels = LOCOMO_CATEGORY_LABELS if task_dir.name == "LoCoMo" else {}
    result: dict[str, Any] = {}
    for category, keys in sorted(categories.items()):
        evaluator_means = {
            evaluator: sum(float(records[key]["metrics"][metric]) for key in keys) / len(keys)
            for evaluator, records in predictions.items()
        }
        result[str(category)] = {
            "label": labels.get(category),
            "case_count": len(keys),
            "mean_score_by_evaluator": evaluator_means,
        }
    return result


def analyze_task(task_dir: Path, expected_evaluators: int) -> dict[str, Any]:
    retrieval_path = task_dir / "retrieval.jsonl"
    prediction_paths = sorted((task_dir / "evaluations").glob("*/predictions.jsonl"))
    if not retrieval_path.is_file():
        return {"status": "missing retrieval"}
    retrieval_rows = read_jsonl(retrieval_path)
    retrieval = {case_key(record): record for record in retrieval_rows}
    if len(retrieval) != len(retrieval_rows):
        raise ValueError(f"duplicate (group_id, case_id) in {retrieval_path}")

    result: dict[str, Any] = {
        "status": "complete" if len(prediction_paths) == expected_evaluators else "incomplete",
        "retrieval_case_count": len(retrieval),
        "evaluator_count": len(prediction_paths),
        "evaluators": {},
    }
    zero_sets: list[set[tuple[str, str]]] = []
    consistently_zero: set[tuple[str, str]] = set()
    prediction_maps: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    metric_name: str | None = None
    for path in prediction_paths:
        evaluator = path.parent.name
        rows = read_jsonl(path)
        predictions = {case_key(record): record for record in rows}
        if len(predictions) != len(rows):
            raise ValueError(f"duplicate (group_id, case_id) in {path}")
        if set(predictions) != set(retrieval):
            result["status"] = "incomplete"
        missing_keys = set(retrieval) - set(predictions)
        unexpected_keys = set(predictions) - set(retrieval)
        if missing_keys or unexpected_keys:
            result["status"] = "incomplete"
            result["evaluators"][evaluator] = {
                "case_count": len(rows),
                "missing_case_count": len(missing_keys),
                "unexpected_case_count": len(unexpected_keys),
            }
            continue
        prediction_maps[evaluator] = predictions
        if not rows:
            result["evaluators"][evaluator] = {"case_count": 0}
            continue
        current_metric = primary_metric(rows[0]["metrics"])
        if metric_name is not None and current_metric != metric_name:
            raise ValueError(f"answer metric differs across evaluators in {task_dir}")
        metric_name = current_metric
        scores = {key: float(record["metrics"][current_metric]) for key, record in predictions.items()}
        zero = {key for key, score in scores.items() if score == 0.0}
        partial = {key for key, score in scores.items() if 0.0 < score < 1.0}
        full = {key for key, score in scores.items() if score == 1.0}
        zero_sets.append(zero)
        result["evaluators"][evaluator] = {
            "case_count": len(rows),
            "mean_score": sum(scores.values()) / len(scores),
            "zero_score_count": len(zero),
            "partial_score_count": len(partial),
            "full_score_count": len(full),
            "zero_score_cases": [question_record(key, predictions[key], retrieval, current_metric) for key in sorted(zero)],
        }

    result["answer_metric"] = metric_name
    if result["status"] == "complete" and len(zero_sets) == expected_evaluators:
        consistently_zero = set.intersection(*zero_sets)
        result["zero_score_for_every_evaluator"] = [
            consistent_failure_record(key, prediction_maps, retrieval, metric_name) for key in sorted(consistently_zero)
        ]
        category_scores = benchmark_category_scores(task_dir, retrieval, prediction_maps, metric_name)
        if category_scores:
            result["benchmark_category_scores"] = category_scores

    if prediction_maps and retrieval and metric_name is not None:
        exemplar_predictions = next(iter(prediction_maps.values()))
        recall_name = next(
            (name for name in exemplar_predictions[next(iter(exemplar_predictions))]["metrics"] if name.startswith("passage_recall_at_")),
            None,
        )
        if recall_name is not None:
            recall_groups: dict[str, list[tuple[tuple[str, str], float]]] = {"none": [], "partial": [], "all": []}
            for key, record in exemplar_predictions.items():
                recall = float(record["metrics"][recall_name])
                group = "none" if recall == 0.0 else "all" if recall == 1.0 else "partial"
                recall_groups[group].append((key, recall))
            result["gold_passage_retrieval"] = {
                "metric": recall_name,
                **{
                    name: {
                        "count": len(records),
                        "mean_answer_score_by_evaluator": {
                            evaluator: (
                                sum(float(predictions[key]["metrics"][metric_name]) for key, _ in records) / len(records)
                                if records else None
                            )
                            for evaluator, predictions in prediction_maps.items()
                        },
                        "zero_score_for_every_evaluator_count": (
                            sum(key in consistently_zero for key, _ in records)
                            if result["status"] == "complete" and len(zero_sets) == expected_evaluators
                            else None
                        ),
                        "cases": [
                            {"group_id": key[0], "case_id": key[1], "recall": recall}
                            for key, recall in records
                        ],
                    }
                    for name, records in recall_groups.items()
                },
            }
    return result


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_evaluators <= 0:
        raise SystemExit("--expected-evaluators must be positive")
    report: dict[str, Any] = {}
    for method_dir in sorted(path for path in args.results_root.iterdir() if path.is_dir()):
        method_report: dict[str, Any] = {}
        for task_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            if (task_dir / "retrieval.jsonl").exists():
                method_report[task_dir.name] = analyze_task(task_dir, args.expected_evaluators)
        if method_report:
            report[method_dir.name] = method_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
