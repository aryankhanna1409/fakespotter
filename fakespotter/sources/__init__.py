"""Where posts come from.

* ``praw``   - the official Reddit API through PRAW (needs a client ID and secret).
* ``public`` - Reddit's public ``.json`` listings; no credentials, but stricter limits
               and Reddit may refuse anonymous clients on some networks.
* ``auto``   - ``praw`` when credentials are set, otherwise ``public``.
"""

from __future__ import annotations

from fakespotter.config import RedditCredentials, user_agent
from fakespotter.sources.base import SORTS, TIME_FILTERS, Source, SourceError

SOURCES = ("auto", "praw", "public")

__all__ = ["SORTS", "SOURCES", "TIME_FILTERS", "Source", "SourceError", "get_source"]


def get_source(name: str = "auto", **kwargs) -> Source:
    """Build a source by name. Extra keyword arguments go to the source's constructor."""
    name = name.lower()
    if name == "auto":
        name = "praw" if RedditCredentials.from_env() else "public"
    if name == "praw":
        from fakespotter.sources.reddit import PrawSource

        credentials = kwargs.pop("credentials", None) or RedditCredentials.from_env()
        if credentials is None:
            raise SourceError(
                "The 'praw' source needs REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET. "
                "Create a 'script' app at https://www.reddit.com/prefs/apps, export both "
                "variables (or put them in a .env file), or use --source public."
            )
        return PrawSource(credentials, **kwargs)
    if name == "public":
        from fakespotter.sources.public import PublicJsonSource

        kwargs.setdefault("user_agent", user_agent())
        return PublicJsonSource(**kwargs)
    raise SourceError(f"Unknown source {name!r}; choose one of: {', '.join(SOURCES)}")
