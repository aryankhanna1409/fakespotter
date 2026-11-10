"""A self-contained HTML report: one file, no external requests, works offline.

Charts are inline SVG generated here; a few lines of vanilla JS add hover tooltips and a
searchable, sortable thread table. Light and dark themes follow the viewer's OS setting.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from fakespotter import __version__
from fakespotter.analysis import (
    UNCERTAIN_BELOW,
    Summary,
    distinctive_terms,
    frequent_terms,
    most_discussed,
    summarize,
    timeline,
    top_threads,
    with_signed,
)
from fakespotter.models import ERROR, NEGATIVE, POSITIVE

LABEL_CLASS = {POSITIVE: "pos", NEGATIVE: "neg", ERROR: "err"}
LABEL_NAME = {POSITIVE: "Positive", NEGATIVE: "Negative", ERROR: "Error"}


# ----------------------------------------------------------------------------- helpers
def _e(value: object) -> str:
    return escape("" if value is None else str(value), quote=True)


def _pct(x: float | None, digits: int = 0) -> str:
    return "–" if x is None else f"{x * 100:.{digits}f}%"


def _signed(x: float | None) -> str:
    return "–" if x is None else f"{x:+.2f}"


def _fmt_int(n: float) -> str:
    n = int(n)
    if abs(n) >= 10_000:
        return f"{n / 1000:.0f}k"
    if abs(n) >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _date(ts: pd.Timestamp | None, fmt: str = "%d %b %Y") -> str:
    return "–" if ts is None or pd.isna(ts) else ts.strftime(fmt).lstrip("0")


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1
    exp = 10 ** math.floor(math.log10(value))
    for m in (1, 2, 2.5, 5, 10):
        if value <= m * exp:
            return m * exp
    return 10 * exp


def _nice_scale(value: float, target: int = 5) -> tuple[float, list[float]]:
    """A round axis maximum and evenly spaced tick values (1/2/5 steps, integers when small)."""
    value = max(float(value), 1.0)
    raw = value / target
    exp = 10 ** math.floor(math.log10(raw))
    step = next(m * exp for m in (1, 2, 5, 10) if raw <= m * exp)
    step = max(step, 1.0)
    top = math.ceil(value / step) * step
    ticks = [step * i for i in range(int(round(top / step)) + 1)]
    return top, ticks


# ----------------------------------------------------------------------------- charts
def split_bar(summary: Summary) -> str:
    """One 100% bar: positive / negative / error, with a 2px surface gap between segments."""
    total = summary.total or 1
    width, height, x = 1000, 28, 0.0
    parts = []
    labels = []
    for label in (POSITIVE, NEGATIVE, ERROR):
        count = summary.counts.get(label, 0)
        if not count:
            continue
        w = count / total * width
        tip = f"{LABEL_NAME[label]}: {count} threads ({count / total:.0%} of all)"
        parts.append(
            f'<rect class="m-{LABEL_CLASS[label]}" x="{x + 1:.1f}" y="0" width="{max(w - 2, 1):.1f}" '
            f'height="{height}" rx="4" data-tip="{_e(tip)}"/>'
        )
        if w > 70:
            labels.append(
                f'<text class="seg-label" x="{x + 10:.1f}" y="{height + 22}">{LABEL_NAME[label]} '
                f'<tspan class="strong">{count / total:.0%}</tspan></text>'
            )
        x += w
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height + 30}" role="img" '
        f'aria-label="Share of threads by sentiment label">{"".join(parts)}{"".join(labels)}</svg>'
    )


def timeline_chart(tl: pd.DataFrame) -> str:
    """Stacked bars per period: positive (bottom), negative, error (top)."""
    if tl.empty:
        return '<p class="empty">No dated threads.</p>'
    width, height = 1000, 260
    left, right, top, bottom = 44, 8, 12, 34
    plot_w, plot_h = width - left - right, height - top - bottom
    totals = tl[POSITIVE] + tl[NEGATIVE] + tl[ERROR]
    ymax, yticks = _nice_scale(float(totals.max()))
    n = len(tl)
    slot = plot_w / n
    bar_w = max(2.0, min(48.0, slot * 0.72))
    out = []
    for t in yticks:
        y = top + plot_h - t / ymax * plot_h
        out.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{_fmt_int(t)}</text>'
        )
    label_every = max(1, math.ceil(n / 10))
    freq_days = (tl["period"].iloc[1] - tl["period"].iloc[0]).days if n > 1 else 1
    date_fmt = "%b %Y" if freq_days >= 28 else "%d %b"
    for i, row in tl.iterrows():
        cx = left + slot * i + slot / 2
        y = top + plot_h
        segs = [
            (POSITIVE, int(row[POSITIVE])),
            (NEGATIVE, int(row[NEGATIVE])),
            (ERROR, int(row[ERROR])),
        ]
        nonzero = [s for s in segs if s[1]]
        period = _date(row["period"], date_fmt if freq_days >= 7 else "%a %d %b")
        net = "no scored threads" if pd.isna(row["net"]) else f"net {row['net']:+.2f}"
        tip = (
            f"{'Week of ' if 6 <= freq_days < 28 else ''}{period}: {int(sum(c for _, c in segs))} threads, "
            f"{row[POSITIVE]} positive, {row[NEGATIVE]} negative, {row[ERROR]} errors ({net})"
        )
        for j, (label, count) in enumerate(nonzero):
            h = count / ymax * plot_h
            gap = 2 if j < len(nonzero) - 1 else 0
            rx = 3 if j == len(nonzero) - 1 else 0
            out.append(
                f'<rect class="m-{LABEL_CLASS[label]}" x="{cx - bar_w / 2:.1f}" y="{y - h + gap:.1f}" '
                f'width="{bar_w:.1f}" height="{max(h - gap, 0.5):.1f}" rx="{rx}"/>'
            )
            y -= h
        # Full-height invisible hit target so thin bars are easy to hover.
        out.append(
            f'<rect class="hit" x="{left + slot * i:.1f}" y="{top}" width="{slot:.1f}" '
            f'height="{plot_h}" data-tip="{_e(tip)}"/>'
        )
        if i % label_every == 0:
            out.append(
                f'<text class="tick" x="{cx:.1f}" y="{height - 12}" text-anchor="middle">{_e(period)}</text>'
            )
    out.append(
        f'<line class="axis" x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}"/>'
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Threads per period, stacked by sentiment">{"".join(out)}</svg>'
    )


def histogram(frame: pd.DataFrame, bins: int = 20) -> str:
    """Distribution of signed sentiment, -1 (confidently negative) to +1 (confidently positive)."""
    signed = frame["signed"].dropna()
    if signed.empty:
        return '<p class="empty">No scored threads.</p>'
    width, height = 520, 250
    left, right, top, bottom = 36, 8, 12, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    edges = [-1 + 2 * i / bins for i in range(bins + 1)]
    counts = pd.cut(signed, edges, include_lowest=True).value_counts(sort=False).tolist()
    ymax, yticks = _nice_scale(max(counts))
    out = []
    for t in yticks:
        y = top + plot_h - t / ymax * plot_h
        out.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{_fmt_int(t)}</text>'
        )
    slot = plot_w / bins
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        cls = "neg" if hi <= 0 else "pos"
        if abs((lo + hi) / 2) < UNCERTAIN_BELOW:  # low-confidence zone
            cls += " soft"
        h = count / ymax * plot_h
        x = left + slot * i + 1
        if count:
            out.append(
                f'<rect class="m-{cls}" x="{x:.1f}" y="{top + plot_h - h:.1f}" width="{slot - 2:.1f}" '
                f'height="{h:.1f}" rx="3"/>'
            )
        tip = f"{count} threads with signed sentiment {lo:+.1f} to {hi:+.1f}"
        out.append(
            f'<rect class="hit" x="{left + slot * i:.1f}" y="{top}" width="{slot:.1f}" height="{plot_h}" data-tip="{_e(tip)}"/>'
        )
    for v in (-1, -0.5, 0, 0.5, 1):
        x = left + (v + 1) / 2 * plot_w
        out.append(
            f'<text class="tick" x="{x:.1f}" y="{height - 20}" text-anchor="middle">{(f"{v:+g}" if v else "0")}</text>'
        )
    out.append(
        f'<line class="axis" x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}"/>'
    )
    mid = left + plot_w / 2
    out.append(f'<line class="zero" x1="{mid}" x2="{mid}" y1="{top}" y2="{top + plot_h}"/>')
    out.append(f'<text class="tick" x="{left}" y="{height - 2}">← confidently negative</text>')
    out.append(
        f'<text class="tick" x="{width - right}" y="{height - 2}" text-anchor="end">confidently positive →</text>'
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Histogram of signed sentiment">{"".join(out)}</svg>'
    )


def scatter(frame: pd.DataFrame) -> str:
    """Engagement (comments, log scale) against signed sentiment, one dot per thread."""
    data = frame.dropna(subset=["signed"])
    if data.empty:
        return '<p class="empty">No scored threads.</p>'
    width, height = 520, 290
    left, right, top, bottom = 36, 10, 24, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    cmax = max(1, int(data["num_comments"].max()))
    lmax = math.log10(_nice_max(cmax + 1) + 1)

    def y_of(c: float) -> float:
        return top + plot_h - math.log10(c + 1) / lmax * plot_h

    out = []
    for t in [0, 1, 10, 100, 1000, 10000]:
        if t > _nice_max(cmax + 1):
            break
        y = y_of(t)
        out.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{_fmt_int(t)}</text>'
        )
    mid = left + plot_w / 2
    out.append(f'<line class="zero" x1="{mid}" x2="{mid}" y1="{top}" y2="{top + plot_h}"/>')
    for v in (-1, -0.5, 0, 0.5, 1):
        x = left + (v + 1) / 2 * plot_w
        out.append(
            f'<text class="tick" x="{x:.1f}" y="{height - 20}" text-anchor="middle">{(f"{v:+g}" if v else "0")}</text>'
        )
    out.append(f'<text class="tick" x="{left}" y="{height - 2}">signed sentiment →</text>')
    out.append(f'<text class="tick" x="{left - 30}" y="{top - 10}">comments (log scale)</text>')
    # Draw the larger class first so the rarer one sits on top.
    order = data.assign(_n=data.groupby("sentiment_label")["sentiment_label"].transform("count"))
    for _, row in order.sort_values("_n", ascending=False).iterrows():
        x = left + (row["signed"] + 1) / 2 * plot_w
        y = y_of(row["num_comments"])
        tip = (
            f"{row['title']}\n{LABEL_NAME.get(row['sentiment_label'], row['sentiment_label'])} "
            f"{row['sentiment_score']:.0%} · {row['num_comments']} comments · {row['score']} upvotes"
        )
        out.append(
            f'<circle class="dot m-{LABEL_CLASS.get(row["sentiment_label"], "err")}" cx="{x:.1f}" cy="{y:.1f}" r="5.5" data-tip="{_e(tip)}"/>'
        )
    out.append(
        f'<line class="axis" x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}"/>'
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Comments against signed sentiment, one dot per thread">{"".join(out)}</svg>'
    )


def terms_chart(frame: pd.DataFrame) -> str:
    positive, negative = distinctive_terms(frame, n=6)
    if not positive and not negative:
        common = frequent_terms(frame, n=14)
        if not common:
            return '<p class="empty">Not enough titles to compare.</p>'
        chips = "".join(f'<span class="chip">{_e(t)} <b>{c}</b></span>' for t, c in common)
        return (
            '<p class="note">Too few positive <i>and</i> negative threads to contrast vocabulary; '
            f'most common title words instead:</p><div class="chips">{chips}</div>'
        )
    rows = [(t, "neg") for t in negative[::-1]] + [(t, "pos") for t in positive]
    rows.sort(key=lambda r: r[0].z)
    zmax = max(abs(t.z) for t, _ in rows) or 1
    width, row_h = 1000, 26
    height = row_h * len(rows) + 8
    mid = width / 2
    half = width / 2 - 130
    out = [f'<line class="zero" x1="{mid}" x2="{mid}" y1="0" y2="{height}"/>']
    for i, (term, cls) in enumerate(rows):
        y = i * row_h + 4
        w = max(2.0, abs(term.z) / zmax * half)
        x = mid - w if cls == "neg" else mid
        tip = f"“{term.term}”: in {term.positive} positive and {term.negative} negative titles (z = {term.z:+.2f})"
        out.append(
            f'<rect class="m-{cls}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="{row_h - 8}" rx="3" data-tip="{_e(tip)}"/>'
        )
        if cls == "neg":
            out.append(
                f'<text class="term" x="{x - 8:.1f}" y="{y + 13}" text-anchor="end">{_e(term.term)}</text>'
            )
        else:
            out.append(
                f'<text class="term" x="{x + w + 8:.1f}" y="{y + 13}">{_e(term.term)}</text>'
            )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Title words that lean negative or positive">{"".join(out)}</svg>'
    )


# ----------------------------------------------------------------------------- tables
def _badge(label: str, score: float) -> str:
    cls = LABEL_CLASS.get(label, "err")
    text = "Error" if label == ERROR else f"{LABEL_NAME.get(label, label)} {score:.0%}"
    return f'<span class="badge b-{cls}">{_e(text)}</span>'


def _thread_list(rows: pd.DataFrame, empty: str) -> str:
    if rows.empty:
        return f'<p class="empty">{_e(empty)}</p>'
    items = []
    for _, r in rows.iterrows():
        items.append(
            f'<li><a href="{_e(r["url"])}" target="_blank" rel="noopener noreferrer">{_e(r["title"])}</a>'
            f'<div class="meta">{_badge(r["sentiment_label"], r["sentiment_score"])}'
            f" <span>▲ {_fmt_int(r['score'])}</span><span>💬 {_fmt_int(r['num_comments'])}</span>"
            f"<span>{_e(_date(r.get('created_date')))}</span></div></li>"
        )
    return f'<ol class="threads">{"".join(items)}</ol>'


def _table_json(frame: pd.DataFrame) -> str:
    records = []
    for _, r in frame.iterrows():
        records.append(
            {
                "t": str(r["title"]),
                "u": str(r["url"]),
                "l": str(r["sentiment_label"]),
                "c": round(float(r["sentiment_score"]), 4),
                "s": int(r["score"]),
                "n": int(r["num_comments"]),
                "d": _date(r.get("created_date"), "%Y-%m-%d"),
                "r": str(r.get("subreddit") or ""),
            }
        )
    # Make the JSON safe to embed in a <script> element.
    return json.dumps(records, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")


# ----------------------------------------------------------------------------- page
def _kpis(s: Summary) -> str:
    def tile(label: str, value: str, sub: str, cls: str = "") -> str:
        return (
            f'<div class="kpi {cls}"><div class="kpi-label">{_e(label)}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'
        )

    net_cls = (
        ""
        if s.net_sentiment is None
        else ("neg-text" if s.net_sentiment < -0.2 else "pos-text" if s.net_sentiment > 0.2 else "")
    )
    return "".join(
        [
            tile(
                "Net sentiment",
                _signed(s.net_sentiment),
                _e(s.mood.capitalize()) + " · −1 to +1",
                net_cls,
            ),
            tile(
                "Threads scored",
                f"{s.scored}<small>/{s.total}</small>",
                f"{_pct(s.error_rate)} could not be scored",
            ),
            tile(
                "Positive share",
                _pct(s.share(POSITIVE)),
                f"{s.counts.get(POSITIVE, 0)} positive · {s.counts.get(NEGATIVE, 0)} negative",
            ),
            tile(
                "Mean confidence",
                _pct(s.mean_confidence),
                f"{s.uncertain} below {UNCERTAIN_BELOW:.0%} (uncertain)",
            ),
            tile(
                "Engagement-weighted",
                _signed(s.weighted_net_sentiment),
                "Net sentiment, weighting threads by upvotes + comments",
            ),
        ]
    )


def _findings(s: Summary, frame: pd.DataFrame) -> str:
    """A few sentences a reader can take away without reading the charts."""
    notes = []
    if s.scored:
        notes.append(
            f"Of the {s.scored} threads the model could score, <b>{_pct(s.share(NEGATIVE))} read as negative</b> "
            f"and {_pct(s.share(POSITIVE))} as positive, so the conversation is <b>{_e(s.mood)}</b> "
            f"(net {_signed(s.net_sentiment)})."
        )
    if s.weighted_net_sentiment is not None and s.net_sentiment is not None:
        diff = s.weighted_net_sentiment - s.net_sentiment
        if abs(diff) >= 0.1:
            way = "more positive" if diff > 0 else "more negative"
            notes.append(
                f"The most-engaged threads skew {way} than the average one (weighted net {_signed(s.weighted_net_sentiment)})."
            )
        else:
            notes.append(
                "Popular and quiet threads lean the same way: weighting by engagement barely moves the index."
            )
    for name, rho in (("upvotes", s.corr_upvotes), ("comment count", s.corr_comments)):
        if rho is None:
            continue
        strength = (
            "no meaningful"
            if abs(rho) < 0.1
            else "a weak"
            if abs(rho) < 0.3
            else "a moderate"
            if abs(rho) < 0.5
            else "a strong"
        )
        direction = "" if abs(rho) < 0.1 else (" positive" if rho > 0 else " negative")
        notes.append(
            f"Sentiment shows {strength}{direction} relationship with {name} (Spearman ρ = {rho:+.2f})."
        )
    if s.error_rate >= 0.2:
        notes.append(
            f"<b>{_pct(s.error_rate)} of threads could not be scored.</b> In files from the original script this "
            "was most likely long threads exceeding DistilBERT's 512-token limit; fakespotter now truncates by "
            "tokens, so re-running the scrape scores them."
        )
    if s.uncertain:
        notes.append(
            f"{s.uncertain} verdicts have confidence under {UNCERTAIN_BELOW:.0%}; treat those as mixed."
        )
    return "".join(f"<li>{n}</li>" for n in notes)


def render_html(frame: pd.DataFrame, *, title: str | None = None, source_name: str = "") -> str:
    """Render the full report for a results table and return the HTML text."""
    frame = with_signed(frame)
    s = summarize(frame)
    tl = timeline(frame)
    subs = ", ".join(f"r/{x}" for x in s.subreddits) or "Reddit"
    title = title or f"How {subs} talks about counterfeits"
    span = f"{_date(s.date_min)} – {_date(s.date_max)}" if s.date_min is not None else "undated"
    model = ", ".join(s.models) or "DistilBERT SST-2"
    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC").lstrip("0")
    neg_top = top_threads(frame, NEGATIVE, 5)
    pos_top = top_threads(frame, POSITIVE, 5)
    discussed = most_discussed(frame, 5)
    meta = f"{s.total} threads · {span} · scored with {_e(model)}"
    if source_name:
        meta += f" · <code>{_e(source_name)}</code>"
    return TEMPLATE.format(
        title=_e(title),
        meta=meta,
        kpis=_kpis(s),
        findings=_findings(s, frame),
        split=split_bar(s),
        timeline=timeline_chart(tl),
        histogram=histogram(frame),
        scatter=scatter(frame),
        terms=terms_chart(frame),
        neg_top=_thread_list(neg_top, "No negative threads."),
        pos_top=_thread_list(pos_top, "No positive threads."),
        discussed=_thread_list(discussed, "No threads."),
        table_json=_table_json(frame),
        total=s.total,
        generated=_e(generated),
        version=_e(__version__),
        uncertain=f"{UNCERTAIN_BELOW:.0%}",
    )


def write_html(frame: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(frame, **kwargs), encoding="utf-8")
    return path


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fakespotter report</title>
<meta name="description" content="{title}">
<style>
:root {{
  color-scheme: light;
  --bg: #f6f5f2; --surface: #fcfcfb; --border: #e4e2dc; --grid: #ebe9e4;
  --text: #0b0b0b; --text-2: #52514e; --muted: #7a7974;
  --pos: #2a78d6; --neg: #e34948; --err: #b4b2aa;
  --pos-soft: #86b6ef; --neg-soft: #f0a3a2;
  --pos-ink: #1c5cab; --neg-ink: #b42a29;
  --accent: #eb6834;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --border: #2f2f2c; --grid: #262624;
    --text: #f0efec; --text-2: #c3c2b7; --muted: #8d8c85;
    --pos: #3987e5; --neg: #e66767; --err: #5a5955;
    --pos-soft: #1c5cab; --neg-soft: #8f3434;
    --pos-ink: #86b6ef; --neg-ink: #f29a9a;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --border: #2f2f2c; --grid: #262624;
  --text: #f0efec; --text-2: #c3c2b7; --muted: #8d8c85;
  --pos: #3987e5; --neg: #e66767; --err: #5a5955;
  --pos-soft: #1c5cab; --neg-soft: #8f3434;
  --pos-ink: #86b6ef; --neg-ink: #f29a9a;
}}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
a {{ color: inherit; text-decoration-color: var(--border); text-underline-offset: 3px; }}
a:hover {{ text-decoration-color: currentColor; }}
code {{ font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--text-2); }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }}
header {{ display: flex; gap: 16px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; margin-bottom: 24px; }}
.brand {{ font-weight: 700; letter-spacing: .02em; font-size: 13px; color: var(--muted); text-transform: uppercase; }}
.brand b {{ color: var(--accent); }}
h1 {{ font-size: clamp(24px, 4vw, 34px); line-height: 1.15; margin: 6px 0 8px; letter-spacing: -.01em; }}
.meta-line {{ color: var(--text-2); font-size: 14px; }}
.theme {{ border: 1px solid var(--border); background: var(--surface); color: var(--text-2); border-radius: 999px; padding: 6px 12px; font: inherit; font-size: 13px; cursor: pointer; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 16px; }}
.kpi, .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; }}
.kpi {{ padding: 16px 18px; }}
.kpi-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }}
.kpi-value {{ font-size: 32px; font-weight: 700; letter-spacing: -.02em; margin: 2px 0; font-variant-numeric: tabular-nums; }}
.kpi-value small {{ font-size: 18px; color: var(--muted); font-weight: 600; }}
.kpi-sub {{ font-size: 13px; color: var(--text-2); }}
.neg-text .kpi-value {{ color: var(--neg-ink); }}
.pos-text .kpi-value {{ color: var(--pos-ink); }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 16px; }}
.card {{ padding: 20px; margin-bottom: 16px; min-width: 0; }}
.card h2 {{ font-size: 16px; margin: 0 0 2px; }}
.card .sub {{ font-size: 13px; color: var(--text-2); margin: 0 0 14px; }}
.findings {{ margin: 0; padding-left: 20px; }}
.findings li {{ margin: 6px 0; }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--text-2); margin-bottom: 10px; }}
.legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; background: var(--c); }}
svg.chart {{ width: 100%; height: auto; display: block; overflow: visible; }}
.m-pos {{ fill: var(--pos); }} .m-neg {{ fill: var(--neg); }} .m-err {{ fill: var(--err); }}
.m-pos.soft {{ fill: var(--pos-soft); }} .m-neg.soft {{ fill: var(--neg-soft); }}
.dot {{ stroke: var(--surface); stroke-width: 2; fill-opacity: .9; }}
.dot:hover {{ stroke: var(--text); }}
.hit {{ fill: transparent; }}
.hit:hover {{ fill: var(--text); fill-opacity: .05; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.axis {{ stroke: var(--border); stroke-width: 1.5; }}
.zero {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 4; }}
.tick {{ fill: var(--muted); font-size: 12px; }}
.term {{ fill: var(--text-2); font-size: 14px; }}
.seg-label {{ fill: var(--text-2); font-size: 14px; }}
.seg-label .strong {{ fill: var(--text); font-weight: 700; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px; font-size: 13px; color: var(--text-2); }}
.chip b {{ color: var(--text); }}
.note, .empty {{ color: var(--text-2); font-size: 14px; }}
.threads {{ margin: 0; padding-left: 22px; }}
.threads li {{ margin: 0 0 12px; }}
.threads li a {{ font-weight: 600; }}
.meta {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; font-size: 12.5px; color: var(--muted); margin-top: 3px; }}
.badge {{ border-radius: 6px; padding: 1px 7px; font-weight: 600; font-size: 12px; white-space: nowrap; }}
.b-pos {{ color: var(--pos-ink); background: color-mix(in srgb, var(--pos) 14%, transparent); }}
.b-neg {{ color: var(--neg-ink); background: color-mix(in srgb, var(--neg) 14%, transparent); }}
.b-err {{ color: var(--text-2); background: color-mix(in srgb, var(--err) 30%, transparent); }}
.controls {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }}
.controls input {{ flex: 1 1 220px; min-width: 0; border: 1px solid var(--border); background: var(--bg); color: var(--text); border-radius: 10px; padding: 8px 12px; font: inherit; }}
.seg {{ display: inline-flex; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
.seg button {{ border: 0; background: transparent; color: var(--text-2); padding: 7px 12px; font: inherit; font-size: 13px; cursor: pointer; }}
.seg button[aria-pressed="true"] {{ background: var(--text); color: var(--surface); }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }}
th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); cursor: pointer; user-select: none; white-space: nowrap; }}
th[data-dir="asc"]::after {{ content: " ▲"; }} th[data-dir="desc"]::after {{ content: " ▼"; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }} th.num {{ text-align: right; }}
.count {{ color: var(--muted); font-size: 13px; }}
#tip {{ position: fixed; pointer-events: none; z-index: 10; max-width: 320px; background: var(--text); color: var(--surface); padding: 7px 10px; border-radius: 8px; font-size: 13px; line-height: 1.4; white-space: pre-line; opacity: 0; transition: opacity .08s; }}
.more-wrap {{ text-align: center; margin-top: 12px; }}
.more {{ border: 1px solid var(--border); background: var(--bg); color: var(--text); border-radius: 10px; padding: 8px 16px; font: inherit; font-size: 14px; cursor: pointer; }}
.more[hidden] {{ display: none; }}
footer {{ color: var(--muted); font-size: 13px; margin-top: 28px; }}
@media (max-width: 640px) {{ .wrap {{ padding-top: 20px; }} .kpi-value {{ font-size: 26px; }} .card {{ padding: 16px; }} }}
@media print {{ .theme, .controls {{ display: none; }} body {{ background: #fff; }} }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <div class="brand"><b>●</b> fakespotter</div>
    <h1>{title}</h1>
    <div class="meta-line">{meta}</div>
  </div>
  <button class="theme" id="theme" type="button" aria-label="Toggle light or dark theme">◐ Theme</button>
</header>

<section class="kpis">{kpis}</section>

<section class="card"><h2>Key findings</h2><ul class="findings">{findings}</ul></section>

<section class="card">
  <h2>Verdicts</h2><p class="sub">Every thread's label: positive, negative, or not scorable.</p>
  <div class="legend"><span style="--c:var(--pos)">Positive</span><span style="--c:var(--neg)">Negative</span><span style="--c:var(--err)">Error</span></div>
  {split}
</section>

<section class="card">
  <h2>Threads over time</h2><p class="sub">When the analysed threads were posted, stacked by verdict. Hover a bar for its net sentiment.</p>
  <div class="legend"><span style="--c:var(--pos)">Positive</span><span style="--c:var(--neg)">Negative</span><span style="--c:var(--err)">Error</span></div>
  {timeline}
</section>

<div class="grid2">
  <section class="card">
    <h2>How sure is the model?</h2><p class="sub">Confidence signed by polarity. Lighter bars sit under {uncertain} confidence.</p>
    {histogram}
  </section>
  <section class="card">
    <h2>Does engagement track sentiment?</h2><p class="sub">One dot per scored thread. Hover for the title.</p>
    <div class="legend"><span style="--c:var(--pos)">Positive</span><span style="--c:var(--neg)">Negative</span></div>
    {scatter}
  </section>
</div>

<section class="card">
  <h2>Words that tip the balance</h2><p class="sub">Title words most associated with negative (left) or positive (right) threads, by weighted log-odds.</p>
  {terms}
</section>

<div class="grid2">
  <section class="card"><h2>Most negative</h2><p class="sub">Highest-confidence negative threads.</p>{neg_top}</section>
  <section class="card"><h2>Most positive</h2><p class="sub">Highest-confidence positive threads.</p>{pos_top}</section>
</div>
<section class="card"><h2>Most discussed</h2><p class="sub">Threads with the most comments.</p>{discussed}</section>

<section class="card">
  <h2>All threads</h2><p class="sub">Search, filter and sort every thread in the file.</p>
  <div class="controls">
    <input id="q" type="search" placeholder="Search titles…" aria-label="Search titles">
    <div class="seg" role="group" aria-label="Filter by verdict">
      <button type="button" data-f="ALL" aria-pressed="true">All</button>
      <button type="button" data-f="POSITIVE" aria-pressed="false">Positive</button>
      <button type="button" data-f="NEGATIVE" aria-pressed="false">Negative</button>
      <button type="button" data-f="ERROR" aria-pressed="false">Error</button>
    </div>
    <span class="count" id="count"></span>
  </div>
  <div class="table-wrap"><table>
    <thead><tr>
      <th data-k="t">Thread</th><th data-k="l">Verdict</th>
      <th data-k="s" class="num">Upvotes</th><th data-k="n" class="num">Comments</th><th data-k="d" class="num">Posted</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
  <div class="more-wrap"><button type="button" class="more" id="more">Show more</button></div>
</section>

<footer>
  Generated {generated} by fakespotter {version}. Each thread's title, body and comments are combined and
  scored as one text; a verdict describes the thread's overall tone, not any single comment.
  Reads public discussion only.
</footer>
</div>
<div id="tip" role="tooltip"></div>
<script id="data" type="application/json">{table_json}</script>
<script>
(function () {{
  var tip = document.getElementById("tip");
  document.addEventListener("mousemove", function (e) {{
    var t = e.target.closest && e.target.closest("[data-tip]");
    if (!t) {{ tip.style.opacity = 0; return; }}
    tip.textContent = t.getAttribute("data-tip");
    var x = e.clientX + 14, y = e.clientY + 14, r = tip.getBoundingClientRect();
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px"; tip.style.opacity = 1;
  }});
  var root = document.documentElement, KEY = "fakespotter-theme";
  try {{ var saved = localStorage.getItem(KEY); if (saved) root.dataset.theme = saved; }} catch (e) {{}}
  document.getElementById("theme").onclick = function () {{
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try {{ localStorage.setItem(KEY, root.dataset.theme); }} catch (e) {{}}
  }};

  var data = JSON.parse(document.getElementById("data").textContent);
  var NAMES = {{POSITIVE: "Positive", NEGATIVE: "Negative", ERROR: "Error"}};
  var CLS = {{POSITIVE: "pos", NEGATIVE: "neg", ERROR: "err"}};
  var PAGE = 25;
  var state = {{q: "", f: "ALL", k: "n", dir: "desc", shown: PAGE}};
  var more = document.getElementById("more");
  var body = document.getElementById("rows"), count = document.getElementById("count");
  function esc(s) {{ return String(s).replace(/[&<>"']/g, function (c) {{ return "&#" + c.charCodeAt(0) + ";"; }}); }}
  function safeUrl(u) {{ return /^https?:\\/\\//i.test(u) ? u : "#"; }}
  function num(n) {{ return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n); }}
  function render() {{
    var q = state.q.toLowerCase();
    var rows = data.filter(function (r) {{
      return (state.f === "ALL" || r.l === state.f) && (!q || r.t.toLowerCase().indexOf(q) !== -1);
    }});
    var k = state.k, m = state.dir === "asc" ? 1 : -1;
    rows.sort(function (a, b) {{
      var x = a[k], y = b[k];
      if (k === "l") {{ x = a.l === "POSITIVE" ? a.c : a.l === "NEGATIVE" ? -a.c : -2; y = b.l === "POSITIVE" ? b.c : b.l === "NEGATIVE" ? -b.c : -2; }}
      if (typeof x === "string") return x.localeCompare(y) * m;
      return (x - y) * m;
    }});
    var all = rows.length;
    rows = rows.slice(0, state.shown);
    more.hidden = all <= state.shown;
    more.textContent = "Show " + Math.min(PAGE, all - state.shown) + " more";
    body.innerHTML = rows.map(function (r) {{
      var v = r.l === "ERROR" ? "Error" : NAMES[r.l] + " " + Math.round(r.c * 100) + "%";
      return "<tr><td><a href=\\"" + esc(safeUrl(r.u)) + "\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\">" + esc(r.t) + "</a></td>" +
        "<td><span class=\\"badge b-" + (CLS[r.l] || "err") + "\\">" + esc(v) + "</span></td>" +
        "<td class=\\"num\\">" + num(r.s) + "</td><td class=\\"num\\">" + num(r.n) + "</td><td class=\\"num\\">" + esc(r.d) + "</td></tr>";
    }}).join("");
    count.textContent = (all === data.length ? all : all + " of " + data.length) + " threads" + (all > rows.length ? ", showing " + rows.length : "");
    document.querySelectorAll("th[data-k]").forEach(function (th) {{
      if (th.dataset.k === state.k) th.dataset.dir = state.dir; else delete th.dataset.dir;
    }});
  }}
  more.onclick = function () {{ state.shown += PAGE; render(); }};
  document.getElementById("q").addEventListener("input", function (e) {{ state.q = e.target.value; state.shown = PAGE; render(); }});
  document.querySelectorAll(".seg button").forEach(function (b) {{
    b.onclick = function () {{
      state.f = b.dataset.f; state.shown = PAGE;
      document.querySelectorAll(".seg button").forEach(function (o) {{ o.setAttribute("aria-pressed", String(o === b)); }});
      render();
    }};
  }});
  document.querySelectorAll("th[data-k]").forEach(function (th) {{
    th.onclick = function () {{
      if (state.k === th.dataset.k) state.dir = state.dir === "asc" ? "desc" : "asc";
      else {{ state.k = th.dataset.k; state.dir = th.dataset.k === "t" ? "asc" : "desc"; }}
      render();
    }};
  }});
  render();
}})();
</script>
</body>
</html>
"""
