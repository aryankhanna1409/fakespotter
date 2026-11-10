"""Sentiment backends.

* ``transformers`` - DistilBERT fine-tuned on SST-2, the model the README describes (default).
* ``lexicon``      - a fast, dependency-free word-list scorer, for machines without PyTorch,
                     quick dry runs and tests. Less accurate; use it to explore, not to publish.
"""

from __future__ import annotations

from collections.abc import Sequence

from fakespotter.models import NEGATIVE, POSITIVE, SentimentResult
from fakespotter.sentiment.base import AnalyzerError, SentimentAnalyzer

BACKENDS = ("transformers", "lexicon")

__all__ = ["BACKENDS", "AnalyzerError", "SentimentAnalyzer", "aggregate", "get_analyzer"]


def get_analyzer(name: str = "transformers", **kwargs) -> SentimentAnalyzer:
    name = name.lower()
    if name in ("transformers", "distilbert", "hf"):
        from fakespotter.sentiment.transformer import TransformerAnalyzer

        return TransformerAnalyzer(**kwargs)
    if name == "lexicon":
        from fakespotter.sentiment.lexicon import LexiconAnalyzer

        return LexiconAnalyzer(**kwargs)
    raise AnalyzerError(f"Unknown sentiment backend {name!r}; choose one of: {', '.join(BACKENDS)}")


def aggregate(
    results: Sequence[SentimentResult], weights: Sequence[float] | None = None
) -> SentimentResult:
    """Combine verdicts for several chunks of one thread into a single verdict.

    Each verdict is turned into P(positive), averaged (weighted, typically by chunk
    length), and turned back into a label + confidence. Errored chunks are ignored; if
    every chunk errored the result is an error.
    """
    weights = list(weights) if weights is not None else [1.0] * len(results)
    if len(weights) != len(results):
        raise ValueError("weights and results must have the same length")
    total = 0.0
    acc = 0.0
    for result, weight in zip(results, weights):
        p = result.positive_probability
        if p is None or weight <= 0:
            continue
        acc += p * weight
        total += weight
    if total == 0:
        return SentimentResult.error()
    p_pos = acc / total
    return (
        SentimentResult(POSITIVE, p_pos) if p_pos >= 0.5 else SentimentResult(NEGATIVE, 1 - p_pos)
    )
