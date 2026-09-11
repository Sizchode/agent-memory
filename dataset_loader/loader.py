"""All benchmark loaders and official-protocol dataset preparation."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any


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
    memory_items: tuple[str, ...]
    memory_timestamps: tuple[str, ...]
    turns: tuple["LoCoMoTurn", ...]
    questions: tuple[LoCoMoQuestion, ...]


@dataclass(frozen=True)
class LoCoMoTurn:
    text: str
    speaker_name: str
    speaker_id: str
    role: str
    timestamp: str
    blip_caption: str


@dataclass(frozen=True)
class HippoRAGQuery:
    dataset: str
    query_id: str
    question: str
    answers: tuple[str, ...]
    gold_passages: tuple[str, ...]
    paragraphs: tuple[str, ...]


_TASK_SOURCES: dict[TaskName, tuple[str, str]] = {
    # These are the exact sub-datasets used by the official main-experiment
    # configuration.  In particular, do not pool the length ablations.
    TaskName.SH_DOC_QA: ("Accurate_Retrieval", "ruler_qa1_197K"),
    TaskName.MH_DOC_QA: ("Accurate_Retrieval", "ruler_qa2_421K"),
    TaskName.EVENT_QA: ("Accurate_Retrieval", "eventqa_full"),
    TaskName.FACT_CONSOLIDATION_SH: ("Conflict_Resolution", "factconsolidation_sh_262k"),
    TaskName.FACT_CONSOLIDATION_MH: ("Conflict_Resolution", "factconsolidation_mh_262k"),
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
    split_name, expected_source = _TASK_SOURCES[task_name]
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
        if source.casefold() != expected_source.casefold():
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
        speaker_a = str(conversation["speaker_a"])
        speaker_b = str(conversation["speaker_b"])
        keys = sorted(
            (key for key in conversation if key.startswith("session_") and key[8:].isdigit()),
            key=lambda key: int(key[8:]),
        )
        memory_items = tuple(
            item
            for key in keys
            for item in _format_locomo_session(conversation, key)
        )
        memory_timestamps = tuple(
            _normalize_locomo_timestamp(str(conversation[f"{key}_date_time"]))
            for key in keys
            for _ in conversation[key]
        )
        turns = tuple(
            LoCoMoTurn(
                text=str(turn["text"]),
                speaker_name=str(turn["speaker"]),
                speaker_id="speaker_a" if str(turn["speaker"]) == speaker_a else "speaker_b",
                role="user" if str(turn["speaker"]) == speaker_a else "assistant",
                timestamp=_normalize_locomo_timestamp(str(conversation[f"{key}_date_time"])),
                blip_caption=str(turn.get("blip_caption", "")),
            )
            for key in keys
            for turn in conversation[key]
        )
        questions = tuple(
            LoCoMoQuestion(
                str(item["question"]),
                str(item.get("answer", item.get("adversarial_answer", ""))),
                int(item["category"]),
                tuple(item.get("evidence", [])),
            )
            for item in sample["qa"]
        )
        conversations.append(
            LoCoMoConversation(
                str(sample["sample_id"]),
                memory_items,
                memory_timestamps,
                turns,
                questions,
            )
        )
    return conversations


def load_hipporag2_dataset(root: str | Path, dataset: str, *, max_queries: int = 1000) -> list[HippoRAGQuery]:
    """Load HippoRAG 2's released 1,000-query 2Wiki evaluation format."""

    if dataset != "2wikimultihopqa":
        raise ValueError("dataset must be 2wikimultihopqa")
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")
    root_path = Path(root)
    corpus = _read_json(root_path / f"{dataset}_corpus.json")
    paragraphs = tuple(_format_passage(item["title"], item["text"]) for item in corpus)
    passages_by_title: dict[str, list[str]] = {}
    for item in corpus:
        passages_by_title.setdefault(str(item["title"]), []).append(_format_passage(item["title"], item["text"]))
    queries: list[HippoRAGQuery] = []
    for sample in _read_json(root_path / f"{dataset}.json")[:max_queries]:
        gold_passages = _gold_passages(sample, passages_by_title)
        answers = [str(sample["answer"]), *(str(item) for item in sample.get("answer_aliases", []))]
        query_id = sample.get("id", sample.get("_id"))
        if query_id is None:
            raise ValueError(f"{dataset} query is missing both 'id' and '_id'")
        queries.append(HippoRAGQuery(dataset, str(query_id), str(sample["question"]), tuple(dict.fromkeys(answers)), gold_passages, paragraphs))
    return queries


def chunk_text_into_sentences(text: str, model_name: str = "gpt-4o-mini", chunk_size: int = 4096) -> list[str]:
    """Use MemoryAgentBench's official sentence-preserving token chunking."""

    try:
        import nltk
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("install nltk and tiktoken to chunk MemoryAgentBench contexts") from exc
    # NLTK >= 3.9 split the language tables into punkt_tab.  Downloading both
    # mirrors the official implementation while making its current dependency
    # work on a clean compute node.
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
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


def _gold_passages(sample: dict[str, Any], passages_by_title: dict[str, list[str]]) -> tuple[str, ...]:
    if "supporting_facts" not in sample:
        raise ValueError("query is missing official supporting passage annotations")
    titles = tuple(dict.fromkeys(str(item[0]) for item in sample["supporting_facts"]))
    missing_titles = [title for title in titles if title not in passages_by_title]
    if missing_titles:
        raise ValueError(f"gold passage titles are absent from the corpus: {missing_titles}")
    ambiguous_titles = [title for title in titles if len(passages_by_title[title]) != 1]
    if ambiguous_titles:
        raise ValueError(f"gold passage titles are ambiguous in the corpus: {ambiguous_titles}")
    supporting_titles = set(titles)
    passages = tuple(
        _format_passage(title, " ".join(sentences))
        for title, sentences in sample["context"]
        if str(title) in supporting_titles
    )
    if set(passages) != {passages_by_title[title][0] for title in titles}:
        raise ValueError("official supporting contexts do not match the released corpus")
    return tuple(dict.fromkeys(passages))


def _format_passage(title: Any, text: Any) -> str:
    return f"{title}\n{text}"


def _format_locomo_session(conversation: dict[str, Any], session_key: str) -> tuple[str, ...]:
    """Render LoCoMo's released RAG dialog records, including their dates."""

    date_time = str(conversation.get(f"{session_key}_date_time", ""))
    turns: list[str] = []
    for turn in conversation[session_key]:
        rendered = f'{turn["speaker"]} said, "{turn["text"]}"'
        if turn.get("blip_caption"):
            rendered += f' and shared {turn["blip_caption"]}'
        turns.append(f"{date_time}: {rendered}")
    return tuple(turns)


def _normalize_locomo_timestamp(value: str) -> str:
    """Apply LightMem's released LoCoMo timestamp conversion exactly."""

    return datetime.strptime(value.strip("()"), "%I:%M %p on %d %B, %Y").strftime("%Y-%m-%d %H:%M:%S")


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and isinstance(decoded := json.loads(value), dict):
        return decoded
    raise TypeError("metadata must be a mapping or JSON object string")


def _as_list(value: Any) -> list[Any]:
    return [] if value is None else value if isinstance(value, list) else [value]


def _read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
