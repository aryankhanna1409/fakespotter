"""Plain data types passed between the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
ERROR = "ERROR"
LABELS = (POSITIVE, NEGATIVE, ERROR)


@dataclass(frozen=True)
class SentimentResult:
    """A model verdict: ``label`` is POSITIVE / NEGATIVE / ERROR, ``score`` its confidence."""

    label: str
    score: float

    @classmethod
    def error(cls) -> SentimentResult:
        return cls(ERROR, 0.0)

    @property
    def is_error(self) -> bool:
        return self.label == ERROR

    @property
    def positive_probability(self) -> float | None:
        """P(positive) implied by a binary label + confidence; ``None`` for errors."""
        if self.label == POSITIVE:
            return self.score
        if self.label == NEGATIVE:
            return 1.0 - self.score
        return None

    @property
    def signed(self) -> float | None:
        """Confidence signed by polarity: +0.98 is confidently positive, -0.98 confidently negative."""
        if self.label == POSITIVE:
            return self.score
        if self.label == NEGATIVE:
            return -self.score
        return None


@dataclass
class Post:
    """One Reddit submission, as returned by a :mod:`fakespotter.sources` backend."""

    id: str
    title: str
    selftext: str
    score: int
    num_comments: int
    url: str
    created_utc: float
    subreddit: str
    permalink: str = ""
    author: str | None = None
    comments: list[str] | None = field(default=None, repr=False)

    @property
    def created_date(self) -> datetime:
        return datetime.fromtimestamp(self.created_utc, tz=timezone.utc).replace(tzinfo=None)


@dataclass
class ScoredPost:
    """A post plus the sentiment verdict for its whole thread."""

    post: Post
    sentiment: SentimentResult
    text_chars: int = 0
    error: str | None = None

    def to_row(self, model: str = "") -> dict:
        """The CSV row. Core columns come first, in the order documented in the README."""
        p = self.post
        return {
            "title": p.title,
            "score": p.score,
            "num_comments": p.num_comments,
            "url": p.url,
            "created_utc": float(p.created_utc),
            "sentiment_label": self.sentiment.label,
            "sentiment_score": float(self.sentiment.score),
            "created_date": p.created_date.strftime("%Y-%m-%d %H:%M:%S"),
            # Extra columns, appended after the documented ones.
            "post_id": p.id,
            "subreddit": p.subreddit,
            "permalink": p.permalink,
            "text_chars": self.text_chars,
            "model": model,
            "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
