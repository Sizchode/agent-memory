"""All benchmark loaders and official-protocol dataset preparation."""

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
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


@dataclass(frozen=True)
class LoCoMoQuestion:
    question: str
    answer: str
    category: int
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class LoCoMoConversation:
    sample_id: str
    sessions: tuple[str, ...]
    questions: tuple[LoCoMoQuestion, ...]


@dataclass(frozen=True)
class HippoRAGQuery:
    dataset: str
    query_id: str
    question: str
    answers: tuple[str, ...]
    gold_passages: tuple[str, ...]
    paragraphs: tuple[str, ...]


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
    """Load one MemoryAgentBench task family and expand each context's QA list."""

    task_name = _coerce_task(task)
    split_name, source_prefixes = _TASK_SOURCES[task_name]
    if chunk_size <= 0 or max_contexts is not None and max_contexts <= 0:
        raise ValueError("chunk_size and max_contexts must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install datasets to load MemoryAgentBench") from exc

    samples: list[BenchmarkSample] = []
    for row in load_dataset("ai-hyz/MemoryAgentBench", split=split_name, revision=revision):
        metadata = _as_mapping(row.get("metadata", {}))
        source = str(metadata.get("source", ""))
        if not _matches_source(source, source_prefixes):
            continue
        samples.append(_make_memory_sample(task_name, row, metadata, source, chunk_size, tokenizer_model))
        if max_contexts is not None and len(samples) == max_contexts:
            break
    return samples


def load_locomo(path: str | Path) -> list[LoCoMoConversation]:
    """Load the official ``data/locomo10.json`` conversational QA format."""

    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError("LoCoMo data must be a JSON list")
    conversations: list[LoCoMoConversation] = []
    for sample in payload:
        conversation = sample["conversation"]
        keys = sorted(
            (key for key in conversation if key.startswith("session_") and key[8:].isdigit()),
            key=lambda key: int(key[8:]),
        )
        sessions = tuple("\n".join(str(turn["text"]) for turn in conversation[key]) for key in keys)
        questions = tuple(
            LoCoMoQuestion(str(item["question"]), str(item["answer"]), int(item["category"]), tuple(item.get("evidence", [])))
            for item in sample["qa"]
        )
        conversations.append(LoCoMoConversation(str(sample["sample_id"]), sessions, questions))
    return conversations


def load_hipporag2_dataset(root: str | Path, dataset: str, *, max_queries: int = 1000) -> list[HippoRAGQuery]:
    """Load HippoRAG 2's released 1,000-query multi-hop evaluation format."""

    if dataset not in {"musique", "2wikimultihopqa", "hotpotqa"}:
        raise ValueError("dataset must be musique, 2wikimultihopqa, or hotpotqa")
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")
    root_path = Path(root)
    corpus = _read_json(root_path / f"{dataset}_corpus.json")
    corpus_by_title = {str(item["title"]): str(item["text"]) for item in corpus}
    paragraphs = tuple(f"{title}\n{text}" for title, text in corpus_by_title.items())
    queries: list[HippoRAGQuery] = []
    for sample in _read_json(root_path / f"{dataset}.json")[:max_queries]:
        gold_passages = tuple(
            f"{title}\n{corpus_by_title[title]}" for title in _gold_titles(sample) if title in corpus_by_title
        )
        answers = [str(sample["answer"]), *(str(item) for item in sample.get("answer_aliases", []))]
        queries.append(HippoRAGQuery(dataset, str(sample["id"]), str(sample["question"]), tuple(dict.fromkeys(answers)), gold_passages, paragraphs))
    return queries


def chunk_text_into_sentences(text: str, model_name: str = "gpt-4o-mini", chunk_size: int = 4096) -> list[str]:
    """Use MemoryAgentBench's official sentence-preserving token chunking."""

    try:
        import nltk
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("install nltk and tiktoken to chunk MemoryAgentBench contexts") from exc
    nltk.download("punkt", quiet=True)
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    chunks, current, token_count = [], [], 0
    for sentence in nltk.sent_tokenize(text):
        sentence_tokens = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if current and token_count + sentence_tokens > chunk_size:
            chunks.append(" ".join(current))
            current, token_count = [], 0
        current.append(sentence)
        token_count += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _coerce_task(task: TaskName | str) -> TaskName:
    try:
        return task if isinstance(task, TaskName) else TaskName(task)
    except ValueError as exc:
        raise ValueError(f"unknown MemoryAgentBench task: {task!r}") from exc


def _make_memory_sample(task: TaskName, row: dict[str, Any], metadata: dict[str, Any], source: str, chunk_size: int, tokenizer_model: str) -> BenchmarkSample:
    questions, answers = _as_list(row.get("questions")), _as_list(row.get("answers"))
    ids = _as_list(metadata.get("qa_pair_ids", []))
    if not questions or len(questions) != len(answers):
        raise ValueError(f"invalid questions/answers in MemoryAgentBench source {source}")
    pairs = tuple(
        QuestionAnswer(str(question), tuple(_as_list(answer)), str(ids[index]) if index < len(ids) else None)
        for index, (question, answer) in enumerate(zip(questions, answers, strict=True))
    )
    context = str(row["context"])
    return BenchmarkSample(task, source, context, tuple(chunk_text_into_sentences(context, tokenizer_model, chunk_size)), pairs, metadata)


def _gold_titles(sample: dict[str, Any]) -> tuple[str, ...]:
    if "supporting_facts" in sample:
        return tuple(dict.fromkeys(str(item[0]) for item in sample["supporting_facts"]))
    if "contexts" in sample:
        return tuple(str(item["title"]) for item in sample["contexts"] if item["is_supporting"])
    return tuple(str(item["title"]) for item in sample["paragraphs"] if item.get("is_supporting", True))


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and isinstance(decoded := json.loads(value), dict):
        return decoded
    raise TypeError("metadata must be a mapping or JSON object string")


def _as_list(value: Any) -> list[Any]:
    return [] if value is None else value if isinstance(value, list) else [value]


def _matches_source(source: str, prefixes: Iterable[str]) -> bool:
    return any(source == prefix.rstrip("_") or source.startswith(prefix) for prefix in prefixes)


def _read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
