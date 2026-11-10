"""Reports: a rich terminal summary and a self-contained HTML page."""

from fakespotter.report.html import render_html, write_html
from fakespotter.report.terminal import print_summary

__all__ = ["print_summary", "render_html", "write_html"]
