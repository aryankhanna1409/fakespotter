"""Reading and writing result files (CSV or JSON Lines)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: The documented columns, in the README's order (and the order of the sample CSV).
CORE_COLUMNS = [
    "title",
    "score",
    "num_comments",
    "url",
    "created_utc",
    "sentiment_label",
    "sentiment_score",
    "created_date",
]
EXTRA_COLUMNS = ["post_id", "subreddit", "permalink", "text_chars", "model", "scraped_at"]
COLUMNS = CORE_COLUMNS + EXTRA_COLUMNS
FORMATS = ("csv", "jsonl")


def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return "jsonl" if suffix in (".jsonl", ".ndjson", ".json") else "csv"


def read_results(path: str | Path) -> pd.DataFrame:
    """Load a results file written by fakespotter (any version, including the samples)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if detect_format(path) == "jsonl":
        frame = pd.read_json(path, lines=True, dtype={"post_id": str})
    else:
        frame = pd.read_csv(path, dtype={"post_id": str})
    return normalize(frame)


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill in derived columns so older files (e.g. the 30-post sample) can be analysed."""
    frame = frame.copy()
    if "created_date" not in frame and "created_utc" in frame:
        frame["created_date"] = pd.to_datetime(frame["created_utc"], unit="s")
    if "created_date" in frame:
        frame["created_date"] = pd.to_datetime(frame["created_date"], errors="coerce")
    if "post_id" not in frame and "url" in frame:
        frame["post_id"] = (
            frame["url"].astype(str).str.extract(r"/comments/([a-z0-9]+)", expand=False)
        )
    if "subreddit" not in frame and "url" in frame:
        found = frame["url"].astype(str).str.extract(r"/r/([^/]+)/", expand=False)
        known = found.dropna().unique()
        # Older files have no subreddit column; gallery/image URLs don't name one. If every
        # URL that does name a subreddit agrees, the file came from that one subreddit.
        frame["subreddit"] = found.fillna(known[0]) if len(known) == 1 else found
    for column in ("score", "num_comments"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    if "sentiment_score" in frame:
        frame["sentiment_score"] = pd.to_numeric(frame["sentiment_score"], errors="coerce").fillna(
            0.0
        )
    if "sentiment_label" in frame:
        frame["sentiment_label"] = frame["sentiment_label"].fillna("ERROR").astype(str).str.upper()
    return frame


def merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Combine two result sets, keeping the newest row for each post.

    Rows are matched on ``post_id`` (falling back to ``url``). A fresh ERROR never
    overwrites an earlier successful verdict for the same post.
    """
    existing = existing.copy()
    new = new.copy()
    existing["_order"] = 0
    new["_order"] = 1
    combined = pd.concat([existing, new], ignore_index=True)
    key = (
        combined["post_id"].astype("string")
        if "post_id" in combined
        else pd.Series(pd.NA, index=combined.index, dtype="string")
    )
    key = key.fillna(combined["url"].astype("string"))
    combined["_key"] = key
    combined["_ok"] = (combined["sentiment_label"] != "ERROR").astype(int)
    combined = combined.sort_values(["_key", "_ok", "_order"], kind="stable")
    combined = combined.drop_duplicates("_key", keep="last")
    combined = combined.drop(columns=["_key", "_ok", "_order"])
    if "created_utc" in combined:
        combined = combined.sort_values("created_utc", ascending=False, kind="stable")
    ordered = [c for c in COLUMNS if c in combined] + [c for c in combined if c not in COLUMNS]
    return combined[ordered].reset_index(drop=True)


def write_results(
    frame: pd.DataFrame, path: str | Path, *, fmt: str | None = None, append: bool = False
) -> pd.DataFrame:
    """Write ``frame`` to ``path``; with ``append`` merge into whatever is already there.

    Returns the frame that was written, normalised (parsed dates etc.) for analysis.
    """
    path = Path(path)
    fmt = fmt or detect_format(path)
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}; choose one of {FORMATS}")
    if append and path.exists():
        frame = merge(read_results(path), frame)
    out = frame.copy()
    if "created_date" in out and pd.api.types.is_datetime64_any_dtype(out["created_date"]):
        out["created_date"] = out["created_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        out.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        out.to_csv(path, index=False)
    return normalize(frame)


def existing_ids(path: str | Path) -> set[str]:
    """Post IDs already present in a results file that were scored successfully."""
    path = Path(path)
    if not path.exists():
        return set()
    frame = read_results(path)
    if "post_id" not in frame:
        return set()
    ok = frame[frame.get("sentiment_label", "ERROR") != "ERROR"]
    return set(ok["post_id"].dropna().astype(str))
