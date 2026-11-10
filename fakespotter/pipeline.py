"""The scrape -> build text -> score loop."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from fakespotter.models import Post, ScoredPost, SentimentResult
from fakespotter.sentiment import SentimentAnalyzer, aggregate
from fakespotter.sources import Source
from fakespotter.text import DEFAULT_MAX_CHARS, build_thread_text, chunk

STRATEGIES = ("truncate", "chunked")
#: Upper bound on chunks per thread for the chunked strategy, to keep huge threads cheap.
MAX_CHUNKS = 8

#: Called after each post is scored: (scored_post, index_in_subreddit, limit).
ProgressCallback = Callable[[ScoredPost, int, int], None]
#: Called when a subreddit starts: (subreddit, limit).
SubredditCallback = Callable[[str, int], None]


@dataclass
class RunResult:
    """Everything one run produced."""

    scored: list[ScoredPost] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # subreddit -> listing error
    model: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    @property
    def failed(self) -> int:
        return sum(s.sentiment.is_error for s in self.scored)

    def to_frame(self) -> pd.DataFrame:
        from fakespotter.export import COLUMNS

        rows = [s.to_row(self.model) for s in self.scored]
        return pd.DataFrame(rows, columns=COLUMNS)


class Scraper:
    """Fetches posts from a :class:`Source` and scores each thread with an analyzer.

    A thread that cannot be fetched or scored gets the label ``ERROR`` with score ``0.0``,
    exactly like the original script, and the run carries on.
    """

    def __init__(
        self,
        source: Source,
        analyzer: SentimentAnalyzer,
        *,
        delay: float = 2.0,
        max_chars: int = DEFAULT_MAX_CHARS,
        strategy: str = "truncate",
        comment_limit: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, not {strategy!r}")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.source = source
        self.analyzer = analyzer
        self.delay = max(0.0, delay)
        self.max_chars = max_chars
        self.strategy = strategy
        self.comment_limit = comment_limit
        self.sleep = sleep

    # -- one post ---------------------------------------------------------------------
    def score_text(
        self, title: str, body: str, comments: Iterable[str]
    ) -> tuple[SentimentResult, int]:
        """Score a thread. Returns the verdict and how many characters were analysed."""
        if self.strategy == "truncate":
            text = build_thread_text(title, body, comments, max_chars=self.max_chars)
            if not text:
                return SentimentResult.error(), 0
            return self.analyzer.analyze_one(text), len(text)
        full = build_thread_text(title, body, comments, max_chars=None)
        pieces = chunk(full, self.max_chars)[:MAX_CHUNKS]
        if not pieces:
            return SentimentResult.error(), 0
        results = self.analyzer.analyze(pieces)
        return aggregate(results, [len(p) for p in pieces]), sum(len(p) for p in pieces)

    def score_post(self, post: Post) -> ScoredPost:
        try:
            comments = post.comments
            if comments is None:
                comments = self.source.fetch_comments(post, limit=self.comment_limit)
            result, chars = self.score_text(post.title, post.selftext, comments)
            return ScoredPost(post, result, chars, None if not result.is_error else "empty thread")
        except Exception as exc:  # noqa: BLE001 - one bad thread must not end the run
            return ScoredPost(post, SentimentResult.error(), 0, f"{type(exc).__name__}: {exc}")

    # -- a whole run ------------------------------------------------------------------
    def run(
        self,
        subreddits: Iterable[str],
        *,
        sort: str = "hot",
        limit: int = 100,
        time_filter: str = "week",
        skip_ids: Iterable[str] = (),
        on_subreddit: SubredditCallback | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> RunResult:
        result = RunResult(model=self.analyzer.name)
        skip = set(skip_ids)
        first = True
        for subreddit in subreddits:
            if on_subreddit:
                on_subreddit(subreddit, limit)
            try:
                for index, post in enumerate(
                    self.source.iter_posts(
                        subreddit, sort=sort, limit=limit, time_filter=time_filter
                    ),
                    start=1,
                ):
                    if post.id in skip:
                        continue
                    if not first and self.delay:
                        self.sleep(self.delay)  # respect Reddit's rate limits
                    first = False
                    scored = self.score_post(post)
                    result.scored.append(scored)
                    if on_progress:
                        on_progress(scored, index, limit)
            except Exception as exc:  # noqa: BLE001 - report and move to the next subreddit
                result.errors[subreddit] = f"{type(exc).__name__}: {exc}"
        result.finished_at = datetime.now(timezone.utc)
        return result
