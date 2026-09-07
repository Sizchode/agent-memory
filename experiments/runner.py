"""Benchmark-neutral execution loop for memory retrieval experiments."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, Protocol

from baseline.base import MemoryBaseline, RetrievedItem
from dataset_loader import BenchmarkSample, HippoRAGQuery, LoCoMoConversation
from utils.hipporag_metrics import gold_passage_recall_at_k, hipporag_answer_f1
from utils.locomo_metrics import locomo_bleu1, locomo_token_f1
from utils.metrics import substring_exact_match
from utils.prompts import memorize_prompt, query_prompt


MetricKind = Literal["subem", "locomo", "hipporag"]


class AnswerModel(Protocol):
    def answer(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    answers: tuple[str, ...]
    metric: MetricKind
    prompt_source: str | None = None
    gold_passages: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryGroup:
    group_id: str
    memory_items: tuple[str, ...]
    cases: tuple[EvaluationCase, ...]
    native_sample: Any = None


@dataclass(frozen=True)
class CaseResult:
    group_id: str
    case_id: str
    prediction: str
    retrieved: tuple[RetrievedItem, ...]
    metrics: dict[str, float]


def memory_agent_bench_groups(samples: Sequence[BenchmarkSample]) -> Iterable[MemoryGroup]:
    for index, sample in enumerate(samples):
        yield MemoryGroup(
            group_id=f"mab-{index}",
            memory_items=tuple(memorize_prompt(sample.source, chunk) for chunk in sample.chunks),
            cases=tuple(
                EvaluationCase(
                    case_id=pair.qa_pair_id or f"{index}-{question_index}",
                    question=pair.question,
                    answers=pair.answers,
                    metric="subem",
                    prompt_source=sample.source,
                )
                for question_index, pair in enumerate(sample.question_answers)
            ),
            native_sample=sample,
        )


def locomo_groups(conversations: Sequence[LoCoMoConversation]) -> Iterable[MemoryGroup]:
    for conversation in conversations:
        yield MemoryGroup(
            group_id=f"locomo-{conversation.sample_id}",
            memory_items=conversation.sessions,
            cases=tuple(
                EvaluationCase(
                    case_id=f"{conversation.sample_id}-{index}",
                    question=item.question,
                    answers=(item.answer,),
                    metric="locomo",
                )
                for index, item in enumerate(conversation.questions)
            ),
            native_sample=conversation,
        )


def hipporag_groups(queries: Sequence[HippoRAGQuery]) -> Iterable[MemoryGroup]:
    if not queries:
        return
    dataset = queries[0].dataset
    corpus = queries[0].paragraphs
    if any(query.dataset != dataset or query.paragraphs != corpus for query in queries):
        raise ValueError("a HippoRAG run must contain queries from one shared corpus")
    yield MemoryGroup(
        group_id=f"hipporag-{dataset}",
        memory_items=corpus,
        cases=tuple(
            EvaluationCase(query.query_id, query.question, query.answers, "hipporag", gold_passages=query.gold_passages)
            for query in queries
        ),
        native_sample=queries,
    )


def run_groups(
    groups: Iterable[MemoryGroup],
    *,
    create_baseline: Callable[[MemoryGroup], MemoryBaseline],
    answer_model: AnswerModel,
    top_k: int,
    output_dir: str | Path,
    system_prompt: str,
) -> dict[str, float]:
    """Build memory once per group, answer every associated question, and persist protocol outputs."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "predictions.jsonl"
    metric_values: dict[str, list[float]] = {}
    with result_path.open("w", encoding="utf-8") as stream:
        for group in groups:
            baseline = create_baseline(group)
            try:
                baseline.build(group.memory_items)
                for case in group.cases:
                    retrieved = tuple(baseline.retrieve(case.question, top_k))
                    prediction = answer_model.answer(system_prompt, _answer_prompt(case, retrieved))
                    result = CaseResult(group.group_id, case.case_id, prediction, retrieved, _score(case, prediction, retrieved, top_k))
                    for name, value in result.metrics.items():
                        metric_values.setdefault(name, []).append(value)
                    stream.write(json.dumps(_json_record(result), ensure_ascii=False) + "\n")
            finally:
                baseline.close()
    summary = {name: sum(values) / len(values) for name, values in metric_values.items() if values}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _answer_prompt(case: EvaluationCase, retrieved: Sequence[RetrievedItem]) -> str:
    evidence = "\n\n".join(f"[Evidence {index + 1}]\n{item.text}" for index, item in enumerate(retrieved))
    instruction = query_prompt(case.prompt_source, case.question) if case.prompt_source else (
        "Answer the question using only the supplied evidence. Return only the answer.\n\n"
        f"Question: {case.question}\nAnswer:"
    )
    return f"Retrieved memory:\n{evidence}\n\n{instruction}"


def _score(case: EvaluationCase, prediction: str, retrieved: Sequence[RetrievedItem], top_k: int) -> dict[str, float]:
    if case.metric == "subem":
        return {"substring_exact_match": float(substring_exact_match(prediction, case.answers))}
    if case.metric == "locomo":
        return {
            "token_f1": locomo_token_f1(prediction, case.answers[0]),
            "bleu_1": locomo_bleu1(prediction, case.answers[0]),
        }
    return {
        f"passage_recall_at_{top_k}": gold_passage_recall_at_k(case.gold_passages, [item.text for item in retrieved], top_k),
        "answer_f1": hipporag_answer_f1(prediction, case.answers),
    }


def _json_record(result: CaseResult) -> dict[str, Any]:
    return {
        "group_id": result.group_id,
        "case_id": result.case_id,
        "prediction": result.prediction,
        "metrics": result.metrics,
        "retrieved": [asdict(item) for item in result.retrieved],
    }
