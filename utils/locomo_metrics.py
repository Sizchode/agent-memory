"""Deterministic LoCoMo QA metric from the released evaluator."""

from collections import Counter
import re
import string


def locomo_tokens(text: str) -> list[str]:
    """Tokenize with the same Unicode word/non-whitespace split policy."""

    return re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE)


def locomo_normalize(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def locomo_token_f1(prediction: str, ground_truth: str) -> float:
    """Official stemmed token F1 for one prediction/reference pair."""
    try:
        from nltk.stem import PorterStemmer
    except ImportError as exc:
        raise RuntimeError("install nltk to compute LoCoMo token F1") from exc
    stemmer = PorterStemmer()
    predicted = [stemmer.stem(token) for token in locomo_normalize(prediction).split()]
    expected = [stemmer.stem(token) for token in locomo_normalize(ground_truth).split()]
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def locomo_qa_f1(prediction: str, ground_truth: str, category: int) -> float:
    """Apply the official category-specific LoCoMo QA scoring policy."""

    if category == 1:
        predictions = [item.strip() for item in prediction.split(",")]
        ground_truths = [item.strip() for item in ground_truth.split(",")]
        return sum(max(locomo_token_f1(candidate, expected) for candidate in predictions) for expected in ground_truths) / len(ground_truths)
    if category == 3:
        return locomo_token_f1(prediction, ground_truth.split(";", 1)[0].strip())
    if category in {2, 4}:
        return locomo_token_f1(prediction, ground_truth)
    if category == 5:
        normalized = prediction.lower()
        return float("no information available" in normalized or "not mentioned" in normalized)
    raise ValueError(f"unsupported LoCoMo category: {category}")
