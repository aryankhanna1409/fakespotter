"""Reddit's public JSON endpoints: no API keys needed.

Every listing page on reddit.com is also available as JSON by adding ``.json`` to the
URL. This is handy for a first look, but anonymous clients get tighter rate limits and
Reddit blocks them from some networks (cloud hosts in particular). For regular use,
create an API app and use the ``praw`` source.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from fakespotter.config import DEFAULT_USER_AGENT
from fakespotter.models import Post
from fakespotter.sources.base import Source, SourceError

BASE = "https://www.reddit.com"
PAGE_SIZE = 100  # Reddit's maximum per listing request

Fetcher = Callable[[str], Any]


class PublicJsonSource(Source):
    name = "public"

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        fetch: Fetcher | None = None,
        timeout: float = 20.0,
        retries: int = 3,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep
        self._fetch = fetch or self._http_get

    # -- HTTP ---------------------------------------------------------------------------
    def _http_get(self, url: str) -> Any:
        request = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        url = f"{BASE}{path}" + (f"?{query}" if query else "")
        delay = self.backoff
        for attempt in range(self.retries + 1):
            try:
                return self._fetch(url)
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.retries:
                    hint = (
                        " Reddit refuses anonymous requests from some networks; set "
                        "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET and use --source praw."
                        if exc.code in (401, 403)
                        else ""
                    )
                    raise SourceError(f"Reddit returned HTTP {exc.code} for {url}.{hint}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == self.retries:
                    raise SourceError(f"Could not fetch {url}: {exc}") from exc
            self.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

    # -- Source API -----------------------------------------------------------------------
    def iter_posts(
        self, subreddit: str, *, sort: str = "hot", limit: int = 100, time_filter: str = "week"
    ) -> Iterator[Post]:
        self.validate(sort, time_filter)
        after: str | None = None
        yielded = 0
        while yielded < limit:
            params = {
                "limit": min(PAGE_SIZE, limit - yielded),
                "after": after,
                "raw_json": 1,
                "t": time_filter if sort in ("top", "controversial") else None,
            }
            payload = self.get_json(f"/r/{subreddit}/{sort}.json", params)
            data = (payload or {}).get("data") or {}
            children = data.get("children") or []
            if not children:
                return
            for child in children:
                if child.get("kind") != "t3":
                    continue
                yield to_post(child.get("data") or {}, fallback_subreddit=subreddit)
                yielded += 1
                if yielded >= limit:
                    return
            after = data.get("after")
            if not after:
                return

    def fetch_comments(self, post: Post, *, limit: int | None = None) -> list[str]:
        path = post.permalink.replace(BASE, "") if post.permalink else f"/comments/{post.id}"
        payload = self.get_json(f"{path.rstrip('/')}.json", {"limit": 500, "raw_json": 1})
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        bodies = list(flatten_comments(payload[1]))
        return bodies[:limit] if limit is not None else bodies


def flatten_comments(listing: Any) -> Iterator[str]:
    """Depth-first walk of a comment listing, yielding every comment body (``t1`` nodes)."""
    stack = list(reversed(((listing or {}).get("data") or {}).get("children") or []))
    while stack:
        node = stack.pop()
        if node.get("kind") != "t1":  # "more" stubs and anything unexpected
            continue
        data = node.get("data") or {}
        if data.get("body"):
            yield data["body"]
        replies = data.get("replies")
        if isinstance(replies, dict):
            stack.extend(reversed((replies.get("data") or {}).get("children") or []))


def to_post(data: dict[str, Any], fallback_subreddit: str = "") -> Post:
    permalink = data.get("permalink") or ""
    return Post(
        id=str(data.get("id", "")),
        title=data.get("title") or "",
        selftext=data.get("selftext") or "",
        score=int(data.get("score") or 0),
        num_comments=int(data.get("num_comments") or 0),
        url=data.get("url") or (BASE + permalink),
        created_utc=float(data.get("created_utc") or 0.0),
        subreddit=data.get("subreddit") or fallback_subreddit,
        permalink=BASE + permalink if permalink.startswith("/") else permalink,
        author=data.get("author"),
    )
