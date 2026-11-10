"""Shared fixtures: fake Reddit data, fake sources and a deterministic analyzer."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd
import pytest

from fakespotter.models import NEGATIVE, POSITIVE, Post, SentimentResult
from fakespotter.sentiment import SentimentAnalyzer
from fakespotter.sources import Source

FIXTURES = Path(__file__).parent / "fixtures"


def make_post(i: int = 1, **overrides) -> Post:
    values = dict(
        id=f"p{i}",
        title=f"Post number {i}",
        selftext="Body text",
        score=10 * i,
        num_comments=i,
        url=f"https://www.reddit.com/r/Reps/comments/p{i}/post_{i}/",
        created_utc=1748400000.0 + i * 3600,
        subreddit="Reps",
        permalink=f"https://www.reddit.com/r/Reps/comments/p{i}/post_{i}/",
    )
    values.update(overrides)
    return Post(**values)


class FakeSource(Source):
    name = "fake"

    def __init__(
        self,
        posts: dict[str, list[Post]],
        comments: dict[str, list[str]] | None = None,
        fail_comments: set[str] = frozenset(),
        fail_listing: set[str] = frozenset(),
    ):
        self.posts = posts
        self.comments = comments or {}
        self.fail_comments = set(fail_comments)
        self.fail_listing = set(fail_listing)
        self.calls: list[tuple] = []

    def iter_posts(self, subreddit, *, sort="hot", limit=100, time_filter="week") -> Iterator[Post]:
        self.calls.append(("list", subreddit, sort, limit, time_filter))
        if subreddit in self.fail_listing:
            raise RuntimeError(f"cannot list r/{subreddit}")
        yield from self.posts.get(subreddit, [])[:limit]

    def fetch_comments(self, post, *, limit=None):
        self.calls.append(("comments", post.id, limit))
        if post.id in self.fail_comments:
            raise ConnectionError("comment fetch failed")
        bodies = self.comments.get(post.id, [])
        return bodies[:limit] if limit is not None else bodies


class KeywordAnalyzer(SentimentAnalyzer):
    """POSITIVE 0.9 if the text contains 'good', else NEGATIVE 0.8. Raises on 'boom'."""

    name = "keyword-test"

    def __init__(self):
        self.seen: list[str] = []

    def analyze(self, texts: Sequence[str]) -> list[SentimentResult]:
        out = []
        for text in texts:
            self.seen.append(text)
            if "boom" in text:
                raise RuntimeError("model exploded")
            out.append(
                SentimentResult(POSITIVE, 0.9) if "good" in text else SentimentResult(NEGATIVE, 0.8)
            )
        return out


@pytest.fixture
def analyzer() -> KeywordAnalyzer:
    return KeywordAnalyzer()


@pytest.fixture
def listing_json() -> dict:
    return json.loads((FIXTURES / "listing.json").read_text())


@pytest.fixture
def comments_json() -> list:
    return json.loads((FIXTURES / "comments.json").read_text())


@pytest.fixture
def results_frame() -> pd.DataFrame:
    """A small, hand-checkable results table."""
    rows = [
        ("Amazing haul, love it", 50, 10, "POSITIVE", 0.95, "2025-05-20 10:00:00", "a1", "Reps"),
        ("Great QC on this haul", 20, 4, "POSITIVE", 0.85, "2025-05-21 10:00:00", "a2", "Reps"),
        ("Seller scam, avoid", 5, 30, "NEGATIVE", 0.99, "2025-05-21 12:00:00", "a3", "Reps"),
        (
            "Customs seized my haul",
            2,
            12,
            "NEGATIVE",
            0.97,
            "2025-05-22 09:00:00",
            "a4",
            "FashionReps",
        ),
        (
            "Scam warning for agent",
            1,
            8,
            "NEGATIVE",
            0.55,
            "2025-05-23 09:00:00",
            "a5",
            "FashionReps",
        ),
        ("Long thread that failed", 9, 99, "ERROR", 0.0, "2025-05-23 18:00:00", "a6", "Reps"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "title",
            "score",
            "num_comments",
            "sentiment_label",
            "sentiment_score",
            "created_date",
            "post_id",
            "subreddit",
        ],
    )
    frame["url"] = (
        "https://www.reddit.com/r/" + frame["subreddit"] + "/comments/" + frame["post_id"] + "/x/"
    )
    frame["created_date"] = pd.to_datetime(frame["created_date"])
    frame["created_utc"] = frame["created_date"].astype("int64") // 10**9
    return frame
