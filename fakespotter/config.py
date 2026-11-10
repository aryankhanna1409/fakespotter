"""Settings, read from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_USER_AGENT = "FakeSpotter/1.0 (sentiment research script)"


def load_dotenv(path: str | os.PathLike = ".env", *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=value`` lines from a ``.env`` file into ``os.environ``.

    A tiny, dependency-free subset of python-dotenv: comments, blank lines, an optional
    ``export`` prefix and single/double quotes are understood. Existing environment
    variables win unless ``override`` is set. Returns the pairs that were read.
    """
    file = Path(path)
    if not file.is_file():
        return {}
    loaded: dict[str, str] = {}
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


@dataclass(frozen=True)
class RedditCredentials:
    client_id: str
    client_secret: str
    user_agent: str = DEFAULT_USER_AGENT

    @classmethod
    def from_env(cls) -> RedditCredentials | None:
        """Credentials from ``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET``, or ``None``."""
        client_id = os.environ.get("REDDIT_CLIENT_ID", "").strip()
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            return None
        return cls(client_id, client_secret, user_agent())


def user_agent() -> str:
    return os.environ.get("REDDIT_USER_AGENT", "").strip() or DEFAULT_USER_AGENT
