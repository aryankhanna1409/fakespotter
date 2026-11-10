"""The terminal summary printed by ``fakespotter report`` and at the end of a scrape."""

from __future__ import annotations

import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fakespotter.analysis import (
    UNCERTAIN_BELOW,
    by_subreddit,
    distinctive_terms,
    frequent_terms,
    most_discussed,
    summarize,
    timeline,
    top_threads,
    with_signed,
)
from fakespotter.models import ERROR, NEGATIVE, POSITIVE

STYLE = {POSITIVE: "bold blue", NEGATIVE: "bold red", ERROR: "dim"}
SPARK = "▁▂▃▄▅▆▇█"


def _bar(counts: dict[str, int], total: int, width: int = 40) -> Text:
    text = Text()
    used = 0
    labels = [lb for lb in (POSITIVE, NEGATIVE, ERROR) if counts.get(lb)]
    for i, label in enumerate(labels):
        n = counts[label]
        cells = width - used if i == len(labels) - 1 else round(n / total * width)
        text.append("█" * cells, style=STYLE[label].replace("bold ", ""))
        used += cells
    return text


def _spark(values: list[float]) -> str:
    if not values:
        return ""
    hi = max(values) or 1
    return "".join(SPARK[min(len(SPARK) - 1, int(v / hi * (len(SPARK) - 1)))] for v in values)


def _short(title: str, n: int = 70) -> str:
    title = " ".join(str(title).split())
    return title if len(title) <= n else title[: n - 1] + "…"


def _threads_table(title: str, rows: pd.DataFrame) -> Table:
    table = Table(
        title=title, title_justify="left", box=box.SIMPLE_HEAD, expand=True, title_style="bold"
    )
    table.add_column("Thread", ratio=6, overflow="ellipsis", no_wrap=True)
    table.add_column("Verdict", ratio=2, no_wrap=True)
    table.add_column("▲", justify="right", ratio=1)
    table.add_column("💬", justify="right", ratio=1)
    for _, r in rows.iterrows():
        label = r["sentiment_label"]
        verdict = "error" if label == ERROR else f"{label.lower()} {r['sentiment_score']:.0%}"
        table.add_row(
            _short(r["title"]),
            Text(verdict, style=STYLE.get(label, "")),
            str(r["score"]),
            str(r["num_comments"]),
        )
    return table


def print_summary(
    frame: pd.DataFrame, console: Console | None = None, *, top: int = 5, source_name: str = ""
) -> None:
    console = console or Console()
    frame = with_signed(frame)
    s = summarize(frame)
    subs = ", ".join(f"r/{x}" for x in s.subreddits) or "Reddit"
    span = (
        f"{s.date_min:%d %b %Y} – {s.date_max:%d %b %Y}"
        if s.date_min is not None and not pd.isna(s.date_min)
        else "undated"
    )

    head = Table.grid(padding=(0, 2))
    head.add_column(style="dim")
    head.add_column()
    net_style = (
        "red" if (s.net_sentiment or 0) < -0.2 else "blue" if (s.net_sentiment or 0) > 0.2 else ""
    )
    head.add_row("Threads", f"{s.total}  ({s.scored} scored, {s.counts[ERROR]} errors)")
    head.add_row("Posted", span)
    head.add_row(
        "Net sentiment",
        Text.assemble(
            (
                f"{s.net_sentiment:+.2f}" if s.net_sentiment is not None else "–",
                f"bold {net_style}",
            ),
            f"  {s.mood}  (−1 all negative … +1 all positive)",
        ),
    )
    if s.weighted_net_sentiment is not None:
        head.add_row("Engagement-weighted", f"{s.weighted_net_sentiment:+.2f}")
    if s.mean_confidence is not None:
        head.add_row(
            "Mean confidence",
            f"{s.mean_confidence:.0%}  ({s.uncertain} below {UNCERTAIN_BELOW:.0%})",
        )
    for name, rho in (("ρ vs upvotes", s.corr_upvotes), ("ρ vs comments", s.corr_comments)):
        if rho is not None:
            head.add_row(name, f"{rho:+.2f}")

    legend = Text.assemble(
        ("■ ", "blue"),
        f"positive {s.counts[POSITIVE]}   ",
        ("■ ", "red"),
        f"negative {s.counts[NEGATIVE]}   ",
        ("■ ", "dim"),
        f"error {s.counts[ERROR]}",
    )
    parts = [head, Text(), _bar(s.counts, s.total or 1), legend]

    tl = timeline(frame)
    if len(tl) > 1:
        totals = (tl[POSITIVE] + tl[NEGATIVE] + tl[ERROR]).tolist()
        parts += [
            Text(),
            Text.assemble(("Volume  ", "dim"), _spark(totals), ("  per period", "dim")),
        ]

    pos_terms, neg_terms = distinctive_terms(frame, n=6)
    if pos_terms or neg_terms:
        parts += [
            Text(),
            Text.assemble(
                ("Leans negative  ", "dim"), (", ".join(t.term for t in neg_terms) or "–", "red")
            ),
            Text.assemble(
                ("Leans positive  ", "dim"), (", ".join(t.term for t in pos_terms) or "–", "blue")
            ),
        ]
    else:
        common = frequent_terms(frame, n=8)
        if common:
            parts += [
                Text(),
                Text.assemble(
                    ("Common words  ", "dim"), ", ".join(f"{w} ({c})" for w, c in common)
                ),
            ]

    console.print(
        Panel(
            Group(*parts),
            title=f"[bold]fakespotter[/] · {subs}",
            subtitle=source_name or None,
            expand=True,
        )
    )

    if len(s.subreddits) > 1:
        subs_table = Table(
            title="By subreddit", title_justify="left", box=box.SIMPLE_HEAD, title_style="bold"
        )
        for col in ("Subreddit", "Threads", "Positive", "Negative", "Errors", "Net"):
            subs_table.add_column(col, justify="left" if col == "Subreddit" else "right")
        for _, r in by_subreddit(frame).iterrows():
            net = "–" if pd.isna(r["net"]) else f"{r['net']:+.2f}"
            subs_table.add_row(
                f"r/{r['subreddit']}",
                str(r["threads"]),
                str(r[POSITIVE]),
                str(r[NEGATIVE]),
                str(r[ERROR]),
                net,
            )
        console.print(subs_table)

    if top > 0:
        if s.counts[NEGATIVE]:
            console.print(_threads_table("Most negative", top_threads(frame, NEGATIVE, top)))
        if s.counts[POSITIVE]:
            console.print(_threads_table("Most positive", top_threads(frame, POSITIVE, top)))
        console.print(_threads_table("Most discussed", most_discussed(frame, top)))
    if s.error_rate >= 0.2:
        console.print(
            f"[yellow]![/] {s.error_rate:.0%} of threads have no verdict. In files from the original script this "
            "was most likely threads longer than DistilBERT's 512-token limit; re-scraping now scores them."
        )
