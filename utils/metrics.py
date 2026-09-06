"""Evaluation helpers matching MemoryAgentBench's substring exact match."""

from collections.abc import Sequence
import re
import string


def accuracy(expected: Sequence[str], predicted: Sequence[str]) -> float:
    """Return exact-match accuracy for two equally sized label sequences."""

    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        raise ValueError("accuracy is undefined for empty sequences")
    return sum(actual == guess for actual, guess in zip(expected, predicted, strict=True)) / len(expected)


def normalize_answer(answer: str) -> str:
    """Apply the benchmark's lowercase, punctuation, article, and space rules."""

    text = answer.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def substring_exact_match(prediction: str, ground_truths: Sequence[str]) -> bool:
    """Return true when any normalized ground truth is in the prediction."""

    normalized_prediction = normalize_answer(prediction)
    return any(normalize_answer(answer) in normalized_prediction for answer in ground_truths)
