"""DistilBERT (SST-2) through the Hugging Face ``pipeline`` API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fakespotter.models import NEGATIVE, POSITIVE, SentimentResult
from fakespotter.sentiment.base import AnalyzerError, SentimentAnalyzer

#: The checkpoint ``pipeline("sentiment-analysis")`` falls back to, pinned explicitly so
#: results don't change if the library's default ever does.
DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

_LABEL_ALIASES = {
    "POSITIVE": POSITIVE,
    "POS": POSITIVE,
    "LABEL_1": POSITIVE,
    "NEGATIVE": NEGATIVE,
    "NEG": NEGATIVE,
    "LABEL_0": NEGATIVE,
}


class TransformerAnalyzer(SentimentAnalyzer):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: int | str | None = None,
        batch_size: int = 8,
        pipeline: Any = None,
    ) -> None:
        """``pipeline`` lets tests inject any callable with the HF pipeline signature."""
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self._pipeline = pipeline
        self.name = model.rsplit("/", 1)[-1]

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise AnalyzerError(
                    "The 'transformers' backend needs the ML extras: "
                    'pip install -r requirements.txt  (or pip install -e ".[ml]"). '
                    "To try fakespotter without PyTorch, pass --backend lexicon."
                ) from exc
            try:  # keep the terminal clean: no weight-loading bars or advisory warnings
                from transformers.utils import logging as hf_logging

                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()
            except Exception:  # pragma: no cover - logging helpers moved between versions
                pass
            kwargs: dict[str, Any] = {"model": self.model}
            if self.device is not None:
                kwargs["device"] = self.device
            self._pipeline = pipeline("sentiment-analysis", **kwargs)
        return self._pipeline

    def load(self) -> TransformerAnalyzer:
        """Load (and on first use, download) the model now rather than on first score."""
        _ = self.pipeline
        return self

    def analyze(self, texts: Sequence[str]) -> list[SentimentResult]:
        if not texts:
            return []
        # truncation=True is what keeps long threads from failing: the original script
        # trimmed to ~2,000 characters, which is often more than DistilBERT's 512 tokens.
        outputs = self.pipeline(
            list(texts), truncation=True, max_length=512, batch_size=self.batch_size
        )
        return [_to_result(out) for out in outputs]


def _to_result(output: Any) -> SentimentResult:
    if isinstance(output, list):  # top_k / return_all_scores style output
        output = max(output, key=lambda item: item["score"])
    label = _LABEL_ALIASES.get(str(output["label"]).upper())
    if label is None:
        raise AnalyzerError(f"Unexpected label from model: {output['label']!r}")
    return SentimentResult(label, float(output["score"]))
