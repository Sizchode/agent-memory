"""Load the two external benchmark groups using their released data formats."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


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


def load_locomo(path: str | Path) -> list[LoCoMoConversation]:
    """Load the official ``data/locomo10.json`` format."""

    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError("LoCoMo data must be a JSON list")
    conversations: list[LoCoMoConversation] = []
    for sample in payload:
        conversation = sample["conversation"]
        session_keys = sorted(
            (key for key in conversation if key.startswith("session_") and key[8:].isdigit()),
            key=lambda key: int(key[8:]),
        )
        sessions = tuple(_session_text(conversation, key) for key in session_keys)
        questions = tuple(
            LoCoMoQuestion(
                question=str(item["question"]),
                answer=str(item["answer"]),
                category=int(item["category"]),
                evidence=tuple(str(value) for value in item.get("evidence", [])),
            )
            for item in sample["qa"]
        )
        conversations.append(
            LoCoMoConversation(str(sample["sample_id"]), sessions, questions)
        )
    return conversations


def load_hipporag2_dataset(
    root: str | Path,
    dataset: str,
    *,
    max_queries: int = 1000,
) -> list[HippoRAGQuery]:
    """Load HippoRAG 2's sampled query and corpus JSON files.

    ``root`` must contain ``<dataset>.json`` and ``<dataset>_corpus.json``
    from the official ``HippoRAG2Official/reproduce/dataset`` directory.
    """

    if dataset not in {"musique", "2wikimultihopqa", "hotpotqa"}:
        raise ValueError("dataset must be musique, 2wikimultihopqa, or hotpotqa")
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")
    base = Path(root)
    samples = _read_json(base / f"{dataset}.json")
    corpus = _read_json(base / f"{dataset}_corpus.json")
    corpus_by_title = {str(doc["title"]): str(doc["text"]) for doc in corpus}

    queries: list[HippoRAGQuery] = []
    for sample in samples[:max_queries]:
        gold_titles = _gold_titles(sample)
        gold_passages = tuple(
            f"{title}\n{corpus_by_title[title]}" for title in gold_titles if title in corpus_by_title
        )
        paragraphs = tuple(
            f"{doc['title']}\n{doc['text']}" for doc in corpus
        )
        answers = [str(sample["answer"])]
        answers.extend(str(alias) for alias in sample.get("answer_aliases", []))
        queries.append(
            HippoRAGQuery(
                dataset=dataset,
                query_id=str(sample["id"]),
                question=str(sample["question"]),
                answers=tuple(dict.fromkeys(answers)),
                gold_passages=gold_passages,
                paragraphs=paragraphs,
            )
        )
    return queries


def _read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _session_text(conversation: dict[str, Any], key: str) -> str:
    session = conversation[key]
    if not isinstance(session, list):
        raise ValueError(f"LoCoMo {key} must be a list of turns")
    return "\n".join(str(turn["text"]) for turn in session)


def _gold_titles(sample: dict[str, Any]) -> tuple[str, ...]:
    if "supporting_facts" in sample:
        return tuple(dict.fromkeys(str(item[0]) for item in sample["supporting_facts"]))
    if "contexts" in sample:
        return tuple(str(item["title"]) for item in sample["contexts"] if item["is_supporting"])
    return tuple(
        str(item["title"])
        for item in sample["paragraphs"]
        if item.get("is_supporting", True)
    )
