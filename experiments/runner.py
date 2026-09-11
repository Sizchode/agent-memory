"""Benchmark-neutral execution loop for memory retrieval experiments."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Literal, Protocol

from baseline.base import MemoryBaseline, RetrievedItem
from dataset_loader import BenchmarkSample, HippoRAGQuery, LoCoMoConversation
from utils.hipporag_metrics import gold_passage_precision_at_k, gold_passage_recall_at_k, hipporag_answer_f1
from utils.locomo_metrics import locomo_qa_f1
from utils.metrics import substring_exact_match
from utils.models import release_accelerator_memory
from utils.prompts import SYSTEM_MESSAGE, hipporag_qa_messages, query_prompt


MetricKind = Literal["subem", "locomo", "hipporag"]


class AnswerModel(Protocol):
    def answer(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    answers: tuple[str, ...]
    metric: MetricKind
    prompt_source: str | None = None
    gold_passages: tuple[str, ...] = ()
    category: int | None = None
    adversarial_answer: str | None = None


@dataclass(frozen=True)
class MemoryGroup:
    group_id: str
    memory_items: tuple[str, ...]
    cases: tuple[EvaluationCase, ...]
    native_sample: Any = None
    memory_timestamps: tuple[str | None, ...] = ()
    dialogue_turns: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CaseResult:
    group_id: str
    case_id: str
    prediction: str
    retrieved: tuple[RetrievedItem, ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class RetrievedCase:
    group_id: str
    case: EvaluationCase
    retrieved: tuple[RetrievedItem, ...]
    top_k: int
    retrieval_seconds: float | None = None


def memory_agent_bench_groups(
    samples: Sequence[BenchmarkSample],
    *,
    start_index: int = 0,
) -> Iterable[MemoryGroup]:
    for index, sample in enumerate(samples, start=start_index):
        yield MemoryGroup(
            group_id=f"mab-{index}",
            # Generator and retrieval backbones consume the original document
            # chunks.  The released MemoryAgentBench prompt is used only for
            # downstream QA; wrapping every chunk as a completed dialogue
            # makes document extractors answer the embedded assistant turn.
            memory_items=sample.chunks,
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
            memory_items=conversation.memory_items,
            cases=tuple(
                EvaluationCase(
                    case_id=f"{conversation.sample_id}-{index}",
                    question=item.question,
                    answers=(item.answer,),
                    metric="locomo",
                    category=item.category,
                    adversarial_answer=item.answer if item.category == 5 else None,
                )
                for index, item in enumerate(conversation.questions)
            ),
            native_sample=conversation,
            memory_timestamps=conversation.memory_timestamps,
            dialogue_turns=conversation.turns,
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
    seed: int = 42,
) -> dict[str, float]:
    """Convenience path for small local runs; HPC uses the two explicit phases."""

    destination = Path(output_dir)
    retrieval_path = destination / "retrieval.jsonl"
    retrieve_groups(groups, create_baseline=create_baseline, top_k=top_k, output_path=retrieval_path)
    return evaluate_retrieval(
        retrieval_path,
        answer_model=answer_model,
        output_dir=destination,
        seed=seed,
    )


def retrieve_groups(
    groups: Iterable[MemoryGroup],
    *,
    create_baseline: Callable[[MemoryGroup], MemoryBaseline],
    top_k: int,
    output_path: str | Path,
) -> None:
    """Build a method and write a benchmark-neutral retrieval JSONL artifact."""

    _retrieve_groups(
        groups,
        create_baseline=create_baseline,
        top_k=top_k,
        output_path=output_path,
        build_memory=True,
    )


def retrieve_existing_memory_groups(
    groups: Iterable[MemoryGroup],
    *,
    create_baseline: Callable[[MemoryGroup], MemoryBaseline],
    top_k: int,
    output_path: str | Path,
) -> None:
    """Load a generated memory store and retrieve without rebuilding it."""

    _retrieve_groups(
        groups,
        create_baseline=create_baseline,
        top_k=top_k,
        output_path=output_path,
        build_memory=False,
    )


def _retrieve_groups(
    groups: Iterable[MemoryGroup],
    *,
    create_baseline: Callable[[MemoryGroup], MemoryBaseline],
    top_k: int,
    output_path: str | Path,
    build_memory: bool,
) -> None:

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    result_path = Path(output_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    efficiency_path = result_path.with_name("efficiency.jsonl")
    with result_path.open("w", encoding="utf-8") as stream, efficiency_path.open("w", encoding="utf-8") as efficiency_stream:
        for group in groups:
            baseline = create_baseline(group)
            build_seconds: float | None = None
            status = "completed"
            error: str | None = None
            before_build: dict[str, Any] | None = None
            after_build: dict[str, Any] | None = None
            after_retrieval: dict[str, Any] | None = None
            try:
                before_build = _efficiency_metrics(baseline)
                if build_memory:
                    build_start = perf_counter()
                    baseline.build(group.memory_items)
                    build_seconds = perf_counter() - build_start
                after_build = _efficiency_metrics(baseline)
                for case in group.cases:
                    retrieval_start = perf_counter()
                    retrieved = tuple(baseline.retrieve(case.question, top_k))
                    retrieval_seconds = perf_counter() - retrieval_start
                    stream.write(json.dumps(_retrieval_record(RetrievedCase(group.group_id, case, retrieved, top_k, retrieval_seconds)), ensure_ascii=False) + "\n")
                    stream.flush()
                after_retrieval = _efficiency_metrics(baseline)
            except BaseException as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                after_build = _efficiency_metrics(baseline)
                after_retrieval = after_build
                raise
            finally:
                try:
                    efficiency_stream.write(json.dumps({
                        "group_id": group.group_id,
                        "status": status,
                        "error": error,
                        "build_memory": build_memory,
                        "input_memory_items": len(group.memory_items),
                        "evaluation_cases": len(group.cases),
                        "build_seconds": build_seconds,
                        "generator_statistics_before_build": before_build,
                        "generator_statistics_after_build": after_build,
                        "generator_statistics_after_retrieval": after_retrieval,
                    }, ensure_ascii=False) + "\n")
                    efficiency_stream.flush()
                finally:
                    try:
                        baseline.close()
                    finally:
                        del baseline
                        release_accelerator_memory()


def evaluate_retrieval(
    retrieval_path: str | Path,
    *,
    answer_model: AnswerModel,
    output_dir: str | Path,
    seed: int = 42,
) -> dict[str, float]:
    """Answer and score an existing retrieval artifact in the evaluator environment."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "predictions.jsonl"
    metric_values: dict[str, list[float]] = {}
    choice_random = random.Random(seed)
    with result_path.open("w", encoding="utf-8") as stream:
        for item in _read_retrieval_records(retrieval_path):
            messages, answer_key = _answer_prompt(item.case, item.retrieved, choice_random)
            generation = _official_generation(item.case)
            prediction = answer_model.answer(messages, **generation)
            if item.case.metric == "hipporag" and "Answer:" in prediction:
                prediction = prediction.split("Answer:", 1)[1].strip()
            if answer_key is not None:
                prediction = _locomo_category_5_answer(prediction, answer_key)
            result = CaseResult(
                item.group_id,
                item.case.case_id,
                prediction,
                item.retrieved,
                _score(item.case, prediction, item.retrieved, item.top_k),
            )
            for name, value in result.metrics.items():
                metric_values.setdefault(name, []).append(value)
            stream.write(json.dumps(_json_record(result), ensure_ascii=False) + "\n")
            stream.flush()
    summary = {name: sum(values) / len(values) for name, values in metric_values.items() if values}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _retrieval_record(item: RetrievedCase) -> dict[str, Any]:
    return {
        "group_id": item.group_id,
        "case": asdict(item.case),
        "retrieved": [asdict(retrieved) for retrieved in item.retrieved],
        "top_k": item.top_k,
        "retrieval_seconds": item.retrieval_seconds,
    }


def _efficiency_metrics(baseline: MemoryBaseline) -> dict[str, Any] | None:
    metrics = getattr(baseline, "efficiency_metrics", None)
    return metrics() if callable(metrics) else None


def _read_retrieval_records(path: str | Path) -> Iterable[RetrievedCase]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                case_payload = payload["case"]
                case = EvaluationCase(
                    case_id=str(case_payload["case_id"]),
                    question=str(case_payload["question"]),
                    answers=tuple(case_payload["answers"]),
                    metric=case_payload["metric"],
                    prompt_source=case_payload.get("prompt_source"),
                    gold_passages=tuple(case_payload.get("gold_passages", [])),
                    category=case_payload.get("category"),
                    adversarial_answer=case_payload.get("adversarial_answer"),
                )
                retrieved = tuple(RetrievedItem(**record) for record in payload["retrieved"])
                yield RetrievedCase(str(payload["group_id"]), case, retrieved, int(payload["top_k"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid retrieval record at {path}:{line_number}") from exc


def _answer_prompt(
    case: EvaluationCase,
    retrieved: Sequence[RetrievedItem],
    choice_random: random.Random,
) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    evidence = "\n\n".join(f"[Evidence {index + 1}]\n{item.text}" for index, item in enumerate(retrieved))
    answer_key = None
    if case.prompt_source:
        instruction = query_prompt(case.prompt_source, case.question)
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": f"Retrieved memory:\n{evidence}\n\n{instruction}"}]
    elif case.metric == "locomo":
        question = case.question
        if case.category == 2:
            question += " Use DATE of CONVERSATION to answer with an approximate date."
        if case.category == 5:
            if case.adversarial_answer is None:
                raise ValueError("LoCoMo category 5 requires adversarial_answer")
            unavailable = "No information available"
            if choice_random.random() < 0.5:
                options = (unavailable, case.adversarial_answer)
                answer_key = {"a": unavailable, "b": case.adversarial_answer}
            else:
                options = (case.adversarial_answer, unavailable)
                answer_key = {"a": case.adversarial_answer, "b": unavailable}
            question += f" (a) {options[0]} (b) {options[1]}. Select the correct answer by writing (a) or (b)."
        instruction = (
            "Based on the above conversations, write a short answer for the following question in a few words. "
            "Do not write complete and lengthy sentences. Answer with exact words from the conversations whenever possible.\n\n"
            f"Question: {question}"
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful, respectful and honest assistant whose job is to understand the following conversation and answer questions based on the conversation. If you don't know the answer to a question, please don't share false information.",
            },
            {"role": "user", "content": f"Retrieved memory:\n{evidence}\n\n{instruction}"},
        ]
    else:
        messages = hipporag_qa_messages([item.text for item in retrieved], case.question)
    return messages, answer_key


def _locomo_category_5_answer(prediction: str, answer_key: dict[str, str]) -> str:
    """Map the released LoCoMo category-5 option format back to answer text."""

    normalized = prediction.strip().lower()
    if normalized in {"a", "(a)"}:
        return answer_key["a"]
    if normalized in {"b", "(b)"}:
        return answer_key["b"]
    return prediction


def _official_generation(case: EvaluationCase) -> dict[str, float | int | None]:
    """Released per-question decoding settings for each benchmark protocol."""

    if case.metric == "locomo":
        return {"max_tokens": 50, "temperature": 0.4, "top_k": 10, "top_p": 0.9}
    if case.metric == "hipporag":
        return {"max_tokens": 2048, "temperature": 0.0}
    if case.prompt_source is None:
        raise ValueError("MemoryAgentBench cases require prompt_source")
    if case.prompt_source.startswith("ruler_qa"):
        max_tokens = 50
    elif case.prompt_source.startswith("eventqa_"):
        max_tokens = 40
    elif case.prompt_source.startswith("factconsolidation_"):
        max_tokens = 10
    else:
        raise ValueError(f"no official generation limit for {case.prompt_source!r}")
    return {"max_tokens": max_tokens, "temperature": 0.7}


def _score(case: EvaluationCase, prediction: str, retrieved: Sequence[RetrievedItem], top_k: int) -> dict[str, float]:
    if case.metric == "subem":
        return {"substring_exact_match": float(substring_exact_match(prediction, case.answers))}
    if case.metric == "locomo":
        if case.category is None:
            raise ValueError("LoCoMo cases require a category")
        score = locomo_qa_f1(prediction, case.answers[0], case.category)
        return {"f1": score, f"f1_category_{case.category}": score}
    return {
        f"passage_recall_at_{top_k}": gold_passage_recall_at_k(case.gold_passages, [item.text for item in retrieved], top_k),
        f"passage_precision_at_{top_k}": gold_passage_precision_at_k(case.gold_passages, [item.text for item in retrieved], top_k),
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
