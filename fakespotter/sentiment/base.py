"""The interface every sentiment backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from fakespotter.models import SentimentResult


class AnalyzerError(RuntimeError):
    """A backend could not be loaded."""


class SentimentAnalyzer(ABC):
    #: Short identifier written to the ``model`` column of the output.
    name: str = "analyzer"

    @abstractmethod
    def analyze(self, texts: Sequence[str]) -> list[SentimentResult]:
        """Score each text. Implementations may raise; the pipeline records that as ERROR."""

    def analyze_one(self, text: str) -> SentimentResult:
        return self.analyze([text])[0]
