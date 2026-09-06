"""HippoRAG 2 retrieval and QA metrics."""

from collections import Counter
import re
import string
from collections.abc import Sequence


def hipporag_normalize(answer: str) -> str:
    text = answer.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def gold_passage_recall_at_k(
    gold_passages: Sequence[str], retrieved_passages: Sequence[str], k: int = 5
) -> float:
    """Match HippoRAG 2's set-based Recall@k calculation."""

    if k <= 0:
        raise ValueError("k must be positive")
    gold = set(gold_passages)
    if not gold:
        return 0.0
    return len(set(retrieved_passages[:k]) & gold) / len(gold)


def hipporag_answer_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """Return the maximum normalized token F1 over answer aliases."""

    return max((_answer_f1(prediction, answer) for answer in gold_answers), default=0.0)


def _answer_f1(prediction: str, answer: str) -> float:
    predicted = hipporag_normalize(prediction).split()
    expected = hipporag_normalize(answer).split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    if not common or not predicted or not expected:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)
