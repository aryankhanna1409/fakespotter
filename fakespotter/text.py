"""Turning a Reddit thread into the single block of text the model scores."""

from __future__ import annotations

import re
from collections.abc import Iterable

#: Matches the original script: ~512 DistilBERT tokens is roughly 2,000 characters.
DEFAULT_MAX_CHARS = 2048

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_REMOVED = {"[removed]", "[deleted]"}


def clean(text: str | None) -> str:
    """Collapse whitespace and drop bare URLs, which carry no sentiment."""
    if not text:
        return ""
    return _WS.sub(" ", _URL.sub(" ", text)).strip()


def truncate(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Trim ``text`` to at most ``max_chars``, cutting on a word boundary when possible."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut and not text[max_chars].isspace():
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip()


def build_thread_text(
    title: str | None,
    body: str | None,
    comments: Iterable[str] | None = (),
    *,
    max_chars: int | None = DEFAULT_MAX_CHARS,
) -> str:
    """Combine title, body and comments into one text, as the README describes.

    Removed/deleted bodies and comments are skipped. Pass ``max_chars=None`` to keep the
    whole thread (used by the chunked scoring strategy).
    """
    parts = [clean(title), clean(body) if (body or "").strip() not in _REMOVED else ""]
    for comment in comments or ():
        if (comment or "").strip() in _REMOVED:
            continue
        parts.append(clean(comment))
    text = " ".join(p for p in parts if p)
    return truncate(text, max_chars) if max_chars else text


def chunk(text: str, size: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split ``text`` into consecutive word-aligned pieces of at most ``size`` characters."""
    if size <= 0:
        raise ValueError("size must be positive")
    pieces: list[str] = []
    rest = text.strip()
    while rest:
        piece = truncate(rest, size)
        if not piece:  # a single "word" longer than size: hard cut
            piece = rest[:size]
        pieces.append(piece)
        rest = rest[len(piece) :].lstrip()
    return pieces
