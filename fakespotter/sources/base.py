"""The interface every post source implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from fakespotter.models import Post

SORTS = ("hot", "new", "top", "rising", "controversial")
TIME_FILTERS = ("hour", "day", "week", "month", "year", "all")


class SourceError(RuntimeError):
    """A source could not be configured or reached."""


class Source(ABC):
    """Lists posts from a subreddit and fetches the comments of one post."""

    name: str = "source"

    @abstractmethod
    def iter_posts(
        self, subreddit: str, *, sort: str = "hot", limit: int = 100, time_filter: str = "week"
    ) -> Iterator[Post]:
        """Yield up to ``limit`` posts from ``r/<subreddit>`` in the given listing order."""

    @abstractmethod
    def fetch_comments(self, post: Post, *, limit: int | None = None) -> list[str]:
        """Return the bodies of the post's comments (all of them, or the first ``limit``)."""

    @staticmethod
    def validate(sort: str, time_filter: str) -> None:
        if sort not in SORTS:
            raise SourceError(f"Unknown sort {sort!r}; choose one of: {', '.join(SORTS)}")
        if time_filter not in TIME_FILTERS:
            raise SourceError(
                f"Unknown time filter {time_filter!r}; choose one of: {', '.join(TIME_FILTERS)}"
            )
