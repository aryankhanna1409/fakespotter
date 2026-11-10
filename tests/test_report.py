import re

import pandas as pd
from rich.console import Console

from fakespotter.report import print_summary, render_html, write_html
from fakespotter.report.html import _nice_scale


def test_html_report_structure(results_frame):
    html = render_html(results_frame)
    assert html.startswith("<!doctype html>")
    assert html.count("<svg") >= 5
    assert "r/FashionReps, r/Reps" in html
    assert "Key findings" in html and "prefers-color-scheme: dark" in html
    assert "http://" not in html.replace("http://www.w3.org", "")  # no external requests
    assert "<script src" not in html and "<link" not in html


def test_html_escapes_untrusted_titles(results_frame):
    frame = results_frame.copy()
    frame.loc[0, "title"] = '<img src=x onerror=alert(1)> & "quotes"'
    frame.loc[1, "title"] = "</script><script>alert(2)</script>"
    frame.loc[2, "url"] = "javascript:alert(3)"
    html = render_html(frame)
    # The embedded JSON stays inside its (non-executable) script element...
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert "alert(2)" in match.group(1) and "</script>" not in match.group(1)
    # ...and everywhere else in the page the titles are escaped.
    markup = html.replace(match.group(0), "")
    assert "<img src=x" not in markup and "&lt;img src=x" in markup
    assert "<script>alert(2)" not in markup
    # Links are filtered to http(s) in the table renderer.
    assert "safeUrl" in html


def test_html_handles_all_errors_and_single_row(results_frame):
    frame = results_frame.copy()
    frame["sentiment_label"] = "ERROR"
    frame["sentiment_score"] = 0.0
    html = render_html(frame)
    assert "No scored threads." in html
    render_html(results_frame.head(1))


def test_html_handles_undated_rows(results_frame):
    frame = results_frame.copy()
    frame["created_date"] = pd.NaT
    assert "No dated threads." in render_html(frame)


def test_write_html_creates_folders(tmp_path, results_frame):
    path = write_html(results_frame, tmp_path / "nested" / "r.html", title="Custom title")
    assert "Custom title" in path.read_text(encoding="utf-8")


def test_nice_scale():
    assert _nice_scale(47) == (50, [0, 10, 20, 30, 40, 50])
    assert _nice_scale(3)[0] == 3
    assert _nice_scale(0)[0] >= 1


def test_terminal_summary(results_frame):
    console = Console(record=True, width=120)
    print_summary(results_frame, console, top=2)
    text = console.export_text()
    assert "Net sentiment" in text and "-0.20" in text
    assert "By subreddit" in text and "r/FashionReps" in text
    assert "Most negative" in text and "Seller scam, avoid" in text
