"""Evaluation helpers."""

from collections.abc import Sequence


def accuracy(expected: Sequence[str], predicted: Sequence[str]) -> float:
    """Return exact-match accuracy for two equally sized label sequences."""

    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        raise ValueError("accuracy is undefined for empty sequences")
    return sum(actual == guess for actual, guess in zip(expected, predicted, strict=True)) / len(expected)
