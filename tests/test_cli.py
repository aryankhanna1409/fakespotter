import subprocess
import sys
from pathlib import Path

import pytest

from fakespotter import cli
from fakespotter.export import read_results

from .conftest import FakeSource, KeywordAnalyzer, make_post

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_backends(monkeypatch):
    source = FakeSource(
        {
            "Reps": [make_post(1, title="good haul"), make_post(2, title="boom"), make_post(3)],
            "FashionReps": [make_post(4, subreddit="FashionReps")],
        },
        comments={"p1": ["good good"]},
    )
    analyzer = KeywordAnalyzer()
    monkeypatch.setattr(cli, "get_source", lambda name: source)
    monkeypatch.setattr(cli, "_analyzer", lambda args: analyzer)
    return source, analyzer


def test_scrape_writes_csv_and_html(tmp_path, fake_backends, capsys):
    out, html = tmp_path / "r.csv", tmp_path / "r.html"
    code = cli.main(
        [
            "scrape",
            "-s",
            "Reps,r/FashionReps",
            "--backend",
            "lexicon",
            "--delay",
            "0",
            "-o",
            str(out),
            "--html",
            str(html),
        ]
    )
    assert code == 0
    frame = read_results(out)
    assert frame["post_id"].tolist() == ["p1", "p2", "p3", "p4"]
    assert frame["sentiment_label"].tolist() == ["POSITIVE", "ERROR", "NEGATIVE", "NEGATIVE"]
    assert html.exists()
    printed = capsys.readouterr().out
    assert "Scored 3/4 threads" in printed and "[1/100]" in printed


def test_scrape_append_and_skip_existing(tmp_path, fake_backends):
    source, analyzer = fake_backends
    out = tmp_path / "r.csv"
    assert cli.main(["scrape", "-b", "lexicon", "--delay", "0", "-o", str(out), "-q"]) == 0
    analyzer.seen.clear()
    assert (
        cli.main(
            [
                "scrape",
                "-b",
                "lexicon",
                "--delay",
                "0",
                "-o",
                str(out),
                "-q",
                "--append",
                "--skip-existing",
            ]
        )
        == 0
    )
    # p1 and p3 were scored before; p2 errored, so only it is retried.
    assert analyzer.seen == ["boom Body text"]
    assert len(read_results(out)) == 3


def test_scrape_reports_listing_failure(tmp_path, monkeypatch, capsys):
    source = FakeSource({}, fail_listing={"Reps"})
    monkeypatch.setattr(cli, "get_source", lambda name: source)
    monkeypatch.setattr(cli, "_analyzer", lambda args: KeywordAnalyzer())
    assert cli.main(["scrape", "-b", "lexicon", "-o", str(tmp_path / "x.csv")]) == 1
    assert "cannot list r/Reps" in capsys.readouterr().err
    assert not (tmp_path / "x.csv").exists()


def test_scrape_rejects_bad_limit(capsys):
    assert cli.main(["scrape", "--limit", "0"]) == 2


def test_report_command(tmp_path, results_frame, capsys):
    from fakespotter.export import write_results

    path = tmp_path / "r.csv"
    write_results(results_frame, path)
    assert cli.main(["report", str(path), "--html", str(tmp_path / "r.html"), "--top", "1"]) == 0
    assert (tmp_path / "r.html").exists()
    assert "Net sentiment" in capsys.readouterr().out


def test_report_errors(tmp_path, capsys):
    assert cli.main(["report", str(tmp_path / "missing.csv")]) == 1
    assert cli.main(["report", str(cli.sample_path("fake_watches_reddit_posts.csv"))]) == 1
    assert "no sentiment columns" in capsys.readouterr().err


def test_demo(tmp_path, capsys):
    assert cli.main(["demo", "--html", str(tmp_path / "demo.html")]) == 0
    assert "r/Reps" in capsys.readouterr().out
    assert (tmp_path / "demo.html").stat().st_size > 10_000


def test_score_command(capsys):
    assert cli.main(["score", "-b", "lexicon", "this is amazing", "total scam"]) == 0
    out = capsys.readouterr().out
    assert "POSITIVE" in out and "NEGATIVE" in out


def test_doctor(monkeypatch, capsys):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    assert cli.main(["doctor"]) == 0
    assert "Reddit credentials" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "scrape" in capsys.readouterr().out


def test_subreddit_parsing():
    assert cli._subreddits(["Reps, r/FashionReps", "/r/reps/", "Watches"]) == [
        "Reps",
        "FashionReps",
        "Watches",
    ]
    assert cli._subreddits(None) == ["Reps"]


def test_module_and_legacy_script_entry_points():
    def run(*args):
        return subprocess.run(
            [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=120
        )

    result = run("-m", "fakespotter", "--version")
    assert result.returncode == 0 and "fakespotter" in result.stdout
    # The original command from the README still works (and accepts scrape options).
    result = run("fakespotter/fakespotter.py", "--help")
    assert result.returncode == 0 and "--subreddit" in result.stdout
