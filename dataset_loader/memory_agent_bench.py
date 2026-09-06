"""Loader for the Hugging Face MemoryAgentBench repository.

The benchmark stores one complete context and many QA pairs per row.  This
module keeps that shape explicit and prepares chunks once per context so a
runner can build memory once and issue all queries against it.
"""

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Iterable


class TaskName(str, Enum):
    SH_DOC_QA = "SH-Doc QA"
    MH_DOC_QA = "MH-Doc QA"
    EVENT_QA = "EventQA"
    FACT_CONSOLIDATION_SH = "FactConsolidation-SH"
    FACT_CONSOLIDATION_MH = "FactConsolidation-MH"


@dataclass(frozen=True)
class QuestionAnswer:
    question: str
    answers: tuple[str, ...]
    qa_pair_id: str | None


@dataclass(frozen=True)
class BenchmarkSample:
    task: TaskName
    source: str
    context: str
    chunks: tuple[str, ...]
    question_answers: tuple[QuestionAnswer, ...]
    metadata: dict[str, Any]


_TASK_SOURCES: dict[TaskName, tuple[str, tuple[str, ...]]] = {
    TaskName.SH_DOC_QA: ("Accurate_Retrieval", ("ruler_qa1",)),
    TaskName.MH_DOC_QA: ("Accurate_Retrieval", ("ruler_qa2",)),
    TaskName.EVENT_QA: ("Accurate_Retrieval", ("eventqa_",)),
    TaskName.FACT_CONSOLIDATION_SH: ("Conflict_Resolution", ("factconsolidation_sh_",)),
    TaskName.FACT_CONSOLIDATION_MH: ("Conflict_Resolution", ("factconsolidation_mh_",)),
}


def load_memory_agent_bench(
    task: TaskName | str,
    *,
    chunk_size: int = 4096,
    tokenizer_model: str = "gpt-4o-mini",
    max_contexts: int | None = None,
    revision: str = "main",
) -> list[BenchmarkSample]:
    """Load and expand one of the five requested benchmark task families."""

    task_name = _coerce_task(task)
    split_name, source_prefixes = _TASK_SOURCES[task_name]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_contexts is not None and max_contexts <= 0:
        raise ValueError("max_contexts must be positive when provided")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the 'datasets' dependency to load MemoryAgentBench") from exc

    raw_dataset = load_dataset("ai-hyz/MemoryAgentBench", split=split_name, revision=revision)
    samples: list[BenchmarkSample] = []
    for row in raw_dataset:
        metadata = _as_mapping(row.get("metadata", {}))
        source = str(metadata.get("source", ""))
        if not _matches_source(source, source_prefixes):
            continue
        samples.append(_build_sample(task_name, row, metadata, source, chunk_size, tokenizer_model))
        if max_contexts is not None and len(samples) >= max_contexts:
            break
    return samples


def _coerce_task(task: TaskName | str) -> TaskName:
    try:
        return task if isinstance(task, TaskName) else TaskName(task)
    except ValueError as exc:
        valid = ", ".join(task_name.value for task_name in TaskName)
        raise ValueError(f"unknown task {task!r}; choose one of: {valid}") from exc


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    raise TypeError("metadata must be a mapping or a JSON object string")


def _matches_source(source: str, prefixes: Iterable[str]) -> bool:
    return any(source == prefix.rstrip("_") or source.startswith(prefix) for prefix in prefixes)


def _build_sample(
    task: TaskName,
    row: dict[str, Any],
    metadata: dict[str, Any],
    source: str,
    chunk_size: int,
    tokenizer_model: str,
) -> BenchmarkSample:
    questions = _as_list(row.get("questions"))
    answer_groups = _as_list(row.get("answers"))
    qa_ids = _as_list(metadata.get("qa_pair_ids", []))
    if not questions:
        raise ValueError(f"{source} row has no questions")
    if len(questions) != len(answer_groups):
        raise ValueError(f"{source} row has mismatched questions and answers lengths")

    question_answers = tuple(
        QuestionAnswer(
            question=str(question),
            answers=tuple(_as_list(answer_group)),
            qa_pair_id=str(qa_ids[index]) if index < len(qa_ids) else None,
        )
        for index, (question, answer_group) in enumerate(zip(questions, answer_groups, strict=True))
    )
    context = str(row["context"])
    return BenchmarkSample(
        task=task,
        source=source,
        context=context,
        chunks=tuple(chunk_text_into_sentences(context, tokenizer_model, chunk_size)),
        question_answers=question_answers,
        metadata=metadata,
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def chunk_text_into_sentences(text: str, model_name: str = "gpt-4o-mini", chunk_size: int = 4096) -> list[str]:
    """Match the official sentence-preserving token chunking protocol."""

    try:
        import nltk
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("install 'nltk' and 'tiktoken' to chunk MemoryAgentBench contexts") from exc

    nltk.download("punkt", quiet=True)
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    sentences = nltk.sent_tokenize(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if current and current_tokens + sentence_tokens > chunk_size:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks
