"""A deterministic majority-class classifier baseline."""

from collections import Counter
from typing import Any, Iterable


class MajorityClassifier:
    """Predict the most frequent label observed during fitting.

    Features are accepted to keep the estimator compatible with the dataset
    interface, but this baseline intentionally does not use them.
    """

    def __init__(self) -> None:
        self._label: str | None = None

    def fit(self, features: Iterable[dict[str, Any]], targets: Iterable[str]) -> "MajorityClassifier":
        del features
        labels = list(targets)
        if not labels:
            raise ValueError("cannot fit MajorityClassifier on an empty dataset")
        counts = Counter(labels)
        self._label = min(counts, key=lambda label: (-counts[label], label))
        return self

    def predict(self, features: Iterable[dict[str, Any]]) -> list[str]:
        if self._label is None:
            raise RuntimeError("MajorityClassifier must be fitted before prediction")
        return [self._label for _ in features]
