"""The official Reddit API, through PRAW (what the original script used)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fakespotter.config import RedditCredentials
from fakespotter.models import Post
from fakespotter.sources.base import Source, SourceError

REDDIT = "https://www.reddit.com"


class PrawSource(Source):
    name = "praw"

    def __init__(
        self,
        credentials: RedditCredentials | None = None,
        *,
        reddit: Any = None,
        replace_more: int | None = 0,
    ) -> None:
        """``reddit`` lets tests inject a fake client. ``replace_more`` is passed to
        PRAW's ``comments.replace_more``: 0 (default, like the original script) keeps the
        comments already loaded, ``None`` expands every "load more comments" link."""
        if reddit is None:
            if credentials is None:
                raise SourceError("PrawSource needs credentials or a reddit client")
            try:
                import praw
            except ImportError as exc:  # pragma: no cover - praw is a core dependency
                raise SourceError("praw is not installed: pip install praw") from exc
            reddit = praw.Reddit(
                client_id=credentials.client_id,
                client_secret=credentials.client_secret,
                user_agent=credentials.user_agent,
                check_for_async=False,
            )
            reddit.read_only = True
        self.reddit = reddit
        self.replace_more = replace_more
        self._submissions: dict[str, Any] = {}

    def iter_posts(
        self, subreddit: str, *, sort: str = "hot", limit: int = 100, time_filter: str = "week"
    ) -> Iterator[Post]:
        self.validate(sort, time_filter)
        sub = self.reddit.subreddit(subreddit)
        listing = getattr(sub, sort)
        kwargs = {"limit": limit}
        if sort in ("top", "controversial"):
            kwargs["time_filter"] = time_filter
        for submission in listing(**kwargs):
            post = to_post(submission, fallback_subreddit=subreddit)
            self._submissions[post.id] = submission
            yield post

    def fetch_comments(self, post: Post, *, limit: int | None = None) -> list[str]:
        submission = self._submissions.get(post.id) or self.reddit.submission(id=post.id)
        submission.comments.replace_more(limit=self.replace_more)
        bodies = [getattr(c, "body", "") or "" for c in submission.comments.list()]
        self._submissions.pop(post.id, None)
        return bodies[:limit] if limit is not None else bodies


def to_post(submission: Any, fallback_subreddit: str = "") -> Post:
    """Convert a PRAW ``Submission`` (or anything shaped like one) into a :class:`Post`."""
    permalink = getattr(submission, "permalink", "") or ""
    subreddit = getattr(submission, "subreddit", None)
    author = getattr(submission, "author", None)
    return Post(
        id=str(submission.id),
        title=submission.title or "",
        selftext=getattr(submission, "selftext", "") or "",
        score=int(getattr(submission, "score", 0) or 0),
        num_comments=int(getattr(submission, "num_comments", 0) or 0),
        url=getattr(submission, "url", "") or (REDDIT + permalink),
        created_utc=float(submission.created_utc),
        subreddit=str(getattr(subreddit, "display_name", subreddit) or fallback_subreddit),
        permalink=REDDIT + permalink if permalink.startswith("/") else permalink,
        author=str(author) if author is not None else None,
    )
