"""Deterministic LoCoMo token-level metrics from the released evaluator."""

from collections import Counter
import math
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


def locomo_bleu1(prediction: str, ground_truth: str) -> float:
    """Compute LoCoMo's unigram BLEU with brevity penalty."""

    predicted = locomo_normalize(prediction).split()
    expected = locomo_normalize(ground_truth).split()
    if not predicted:
        return 0.0
    if not expected:
        return float(not predicted)
    clipped = sum((Counter(predicted) & Counter(expected)).values())
    precision = clipped / len(predicted)
    if precision == 0:
        return 0.0
    brevity_penalty = 1.0 if len(predicted) > len(expected) else math.exp(1 - len(expected) / len(predicted))
    return brevity_penalty * precision
