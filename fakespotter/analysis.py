"""Turning a results table into findings: shares, trends, drivers and standout threads."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from fakespotter.models import ERROR, NEGATIVE, POSITIVE

#: Verdicts below this confidence are reported as "uncertain".
UNCERTAIN_BELOW = 0.6

STOPWORDS = set(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its itself just me more
    most my no nor not now of off on once only or other our out over own same she should so some
    such than that the their them then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your yours
    s t m re ve ll d got get one new my me im it's i'm just like
    """.split()
)
_WORD = re.compile(r"[a-z][a-z0-9'\-]{1,}")


class NoSentimentError(ValueError):
    """The file has no sentiment columns (e.g. the 30-post metadata-only sample)."""


def require_sentiment(frame: pd.DataFrame) -> None:
    missing = {"sentiment_label", "sentiment_score"} - set(frame.columns)
    if missing:
        raise NoSentimentError(
            "This file has no sentiment columns (" + ", ".join(sorted(missing)) + "). "
            "Score posts first with `fakespotter scrape`."
        )


def with_signed(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``signed`` (+conf for POSITIVE, -conf for NEGATIVE, NaN for ERROR)."""
    require_sentiment(frame)
    frame = frame.copy()
    sign = frame["sentiment_label"].map({POSITIVE: 1.0, NEGATIVE: -1.0})
    frame["signed"] = sign * frame["sentiment_score"]
    return frame


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    """Spearman's rho without SciPy: Pearson on ranks. ``None`` if undefined."""
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return None
    ranked = pair.rank()
    if (ranked.nunique() < 2).any():  # a constant column has no correlation
        return None
    rho = ranked.iloc[:, 0].corr(ranked.iloc[:, 1])
    return None if pd.isna(rho) else float(rho)


@dataclass
class Summary:
    total: int
    scored: int
    counts: dict[str, int]
    net_sentiment: float | None  # (pos - neg) / scored, in [-1, 1]
    weighted_net_sentiment: float | None  # same, weighted by engagement
    mean_confidence: float | None
    confidence_by_label: dict[str, float]
    uncertain: int
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None
    subreddits: list[str]
    corr_upvotes: float | None
    corr_comments: float | None
    models: list[str] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.counts.get(ERROR, 0) / self.total if self.total else 0.0

    def share(self, label: str) -> float:
        """Share of *scored* threads with this label (errors excluded)."""
        if label == ERROR:
            return self.error_rate
        return self.counts.get(label, 0) / self.scored if self.scored else 0.0

    @property
    def mood(self) -> str:
        """A plain-words reading of the net sentiment index."""
        n = self.net_sentiment
        if n is None:
            return "no scored threads"
        if n <= -0.6:
            return "strongly negative"
        if n <= -0.2:
            return "mostly negative"
        if n < 0.2:
            return "mixed"
        if n < 0.6:
            return "mostly positive"
        return "strongly positive"


def summarize(frame: pd.DataFrame) -> Summary:
    frame = with_signed(frame)
    labels = frame["sentiment_label"]
    counts = {label: int((labels == label).sum()) for label in (POSITIVE, NEGATIVE, ERROR)}
    scored_rows = frame[labels != ERROR]
    scored = len(scored_rows)
    net = (counts[POSITIVE] - counts[NEGATIVE]) / scored if scored else None
    weighted = None
    if scored:
        weights = (
            scored_rows.get("score", 0).clip(lower=0)
            + scored_rows.get("num_comments", 0).clip(lower=0)
            + 1
        ).astype(float)
        polarity = scored_rows["signed"].apply(lambda v: 1.0 if v > 0 else -1.0)
        weighted = float((polarity * weights).sum() / weights.sum())
    dates = (
        frame["created_date"].dropna()
        if "created_date" in frame
        else pd.Series(dtype="datetime64[ns]")
    )
    subs = sorted(frame["subreddit"].dropna().astype(str).unique()) if "subreddit" in frame else []
    models = (
        sorted(m for m in frame["model"].dropna().astype(str).unique() if m)
        if "model" in frame
        else []
    )
    return Summary(
        total=len(frame),
        scored=scored,
        counts=counts,
        net_sentiment=net,
        weighted_net_sentiment=weighted,
        mean_confidence=float(scored_rows["sentiment_score"].mean()) if scored else None,
        confidence_by_label={
            label: float(frame.loc[labels == label, "sentiment_score"].mean())
            for label in (POSITIVE, NEGATIVE)
            if counts[label]
        },
        uncertain=int((scored_rows["sentiment_score"] < UNCERTAIN_BELOW).sum()),
        date_min=dates.min() if len(dates) else None,
        date_max=dates.max() if len(dates) else None,
        subreddits=subs,
        corr_upvotes=_spearman(frame["signed"], frame["score"]) if "score" in frame else None,
        corr_comments=_spearman(frame["signed"], frame["num_comments"])
        if "num_comments" in frame
        else None,
        models=models,
    )


def pick_frequency(frame: pd.DataFrame) -> str:
    """Day, week or month buckets depending on how much time the data spans."""
    dates = frame["created_date"].dropna()
    if dates.empty:
        return "D"
    span = (dates.max() - dates.min()).days
    if span <= 21:
        return "D"
    if span <= 210:
        return "W"
    return "M"


def timeline(frame: pd.DataFrame, freq: str | None = None) -> pd.DataFrame:
    """Threads per period by label, plus the period's net sentiment.

    Columns: ``period`` (start timestamp), POSITIVE, NEGATIVE, ERROR, ``net``.
    """
    require_sentiment(frame)
    data = frame.dropna(subset=["created_date"])
    columns = ["period", POSITIVE, NEGATIVE, ERROR, "net"]
    if data.empty:
        return pd.DataFrame(columns=columns)
    freq = freq or pick_frequency(data)
    alias = {"D": "D", "W": "W-MON", "M": "MS"}.get(freq, freq)
    periods = (
        data["created_date"]
        .dt.to_period({"W-MON": "W-SUN", "MS": "M"}.get(alias, alias))
        .dt.start_time
    )
    table = pd.crosstab(periods, data["sentiment_label"])
    for label in (POSITIVE, NEGATIVE, ERROR):
        if label not in table:
            table[label] = 0
    table = table[[POSITIVE, NEGATIVE, ERROR]]
    # Fill empty periods so gaps show as gaps rather than being skipped.
    full = pd.date_range(table.index.min(), table.index.max(), freq=alias)
    table = table.reindex(table.index.union(full), fill_value=0)
    scored = table[POSITIVE] + table[NEGATIVE]
    table["net"] = ((table[POSITIVE] - table[NEGATIVE]) / scored.where(scored > 0)).astype(float)
    table.index.name = "period"
    return table.reset_index()[columns]


def top_threads(frame: pd.DataFrame, label: str, n: int = 5) -> pd.DataFrame:
    """The ``n`` most confident threads with ``label``, highest confidence first."""
    require_sentiment(frame)
    subset = frame[frame["sentiment_label"] == label]
    return subset.sort_values(["sentiment_score", "num_comments"], ascending=False).head(n)


def most_discussed(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return frame.sort_values(["num_comments", "score"], ascending=False).head(n)


def by_subreddit(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per subreddit: threads, label counts, net sentiment, mean confidence."""
    frame = with_signed(frame)
    if "subreddit" not in frame:
        frame = frame.assign(subreddit="(unknown)")
    rows = []
    for sub, group in frame.groupby(frame["subreddit"].fillna("(unknown)")):
        pos = int((group["sentiment_label"] == POSITIVE).sum())
        neg = int((group["sentiment_label"] == NEGATIVE).sum())
        rows.append(
            {
                "subreddit": sub,
                "threads": len(group),
                POSITIVE: pos,
                NEGATIVE: neg,
                ERROR: int((group["sentiment_label"] == ERROR).sum()),
                "net": (pos - neg) / (pos + neg) if pos + neg else float("nan"),
                "mean_confidence": group.loc[
                    group["sentiment_label"] != ERROR, "sentiment_score"
                ].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("threads", ascending=False).reset_index(drop=True)


def tokenize_title(title: str) -> list[str]:
    words = _WORD.findall(str(title).lower())
    return [
        w.strip("'-")
        for w in words
        if w not in STOPWORDS and not w.isdigit() and len(w.strip("'-")) > 1
    ]


@dataclass(frozen=True)
class Term:
    term: str
    z: float  # > 0 leans positive, < 0 leans negative
    positive: int
    negative: int


def distinctive_terms(
    frame: pd.DataFrame, n: int = 8, min_count: int = 2
) -> tuple[list[Term], list[Term]]:
    """Title words that most separate positive from negative threads.

    Uses the weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi
    & Quinn, 2008), which is robust for small samples where raw frequencies mislead.
    Returns ``(leans_positive, leans_negative)``, strongest first.
    """
    require_sentiment(frame)
    pos_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()
    for title, label in zip(frame["title"], frame["sentiment_label"]):
        tokens = set(tokenize_title(title))  # document frequency: one vote per thread
        if label == POSITIVE:
            pos_counts.update(tokens)
        elif label == NEGATIVE:
            neg_counts.update(tokens)
    prior = pos_counts + neg_counts
    vocab = [w for w, c in prior.items() if c >= min_count]
    if not vocab or not pos_counts or not neg_counts:
        return [], []
    a0 = sum(prior[w] for w in vocab)
    n_pos = sum(pos_counts[w] for w in vocab)
    n_neg = sum(neg_counts[w] for w in vocab)
    terms = []
    for w in vocab:
        a = prior[w]
        yp, yn = pos_counts[w], neg_counts[w]
        delta = math.log((yp + a) / (n_pos + a0 - yp - a)) - math.log(
            (yn + a) / (n_neg + a0 - yn - a)
        )
        variance = 1.0 / (yp + a) + 1.0 / (yn + a)
        terms.append(Term(w, delta / math.sqrt(variance), yp, yn))
    positive = sorted((t for t in terms if t.z > 0), key=lambda t: -t.z)[:n]
    negative = sorted((t for t in terms if t.z < 0), key=lambda t: t.z)[:n]
    return positive, negative


def frequent_terms(frame: pd.DataFrame, n: int = 10) -> list[tuple[str, int]]:
    """Most common title words across all threads (what the community talks about)."""
    counts: Counter[str] = Counter()
    for title in frame["title"]:
        counts.update(set(tokenize_title(title)))
    return counts.most_common(n)
