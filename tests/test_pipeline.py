import pytest

from fakespotter.export import CORE_COLUMNS
from fakespotter.models import ERROR, NEGATIVE, POSITIVE
from fakespotter.pipeline import Scraper
from fakespotter.text import DEFAULT_MAX_CHARS

from .conftest import FakeSource, make_post


def test_run_scores_every_post_and_sleeps_between(analyzer):
    source = FakeSource(
        {"Reps": [make_post(1, title="good stuff"), make_post(2), make_post(3)]},
        comments={"p2": ["meh", "ok"]},
    )
    sleeps = []
    result = Scraper(source, analyzer, delay=2, sleep=sleeps.append).run(["Reps"], limit=3)
    assert [s.sentiment.label for s in result.scored] == [POSITIVE, NEGATIVE, NEGATIVE]
    assert sleeps == [2, 2]  # between posts, not after the last
    assert analyzer.seen[1] == "Post number 2 Body text meh ok"
    assert result.model == "keyword-test" and result.finished_at is not None


def test_failures_become_error_rows(analyzer):
    posts = [make_post(1), make_post(2, title="boom"), make_post(3)]
    source = FakeSource({"Reps": posts}, fail_comments={"p1"})
    result = Scraper(source, analyzer, delay=0).run(["Reps"])
    labels = [(s.sentiment.label, s.sentiment.score) for s in result.scored]
    assert labels == [(ERROR, 0.0), (ERROR, 0.0), (NEGATIVE, 0.8)]
    assert "ConnectionError" in result.scored[0].error
    assert "model exploded" in result.scored[1].error
    assert result.failed == 2


def test_listing_failure_is_reported_and_run_continues(analyzer):
    source = FakeSource({"Good": [make_post(1)]}, fail_listing={"Bad"})
    result = Scraper(source, analyzer, delay=0).run(["Bad", "Good"])
    assert list(result.errors) == ["Bad"]
    assert len(result.scored) == 1


def test_skip_ids_and_callbacks(analyzer):
    source = FakeSource({"Reps": [make_post(1), make_post(2), make_post(3)]})
    seen, subs = [], []
    Scraper(source, analyzer, delay=0).run(
        ["Reps"],
        skip_ids={"p2"},
        on_subreddit=lambda s, n: subs.append((s, n)),
        on_progress=lambda sp, i, n: seen.append((sp.post.id, i, n)),
    )
    assert subs == [("Reps", 100)]
    assert seen == [("p1", 1, 100), ("p3", 3, 100)]


def test_comment_limit_and_options_passed_to_source(analyzer):
    source = FakeSource({"Reps": [make_post(1)]}, comments={"p1": ["a", "b", "c"]})
    Scraper(source, analyzer, delay=0, comment_limit=2).run(
        ["Reps"], sort="top", limit=5, time_filter="year"
    )
    assert ("list", "Reps", "top", 5, "year") in source.calls
    assert ("comments", "p1", 2) in source.calls
    assert analyzer.seen == ["Post number 1 Body text a b"]


def test_truncate_strategy_limits_text(analyzer):
    source = FakeSource({"Reps": [make_post(1)]}, comments={"p1": ["word " * 2000]})
    result = Scraper(source, analyzer, delay=0).run(["Reps"])
    assert len(analyzer.seen[0]) <= DEFAULT_MAX_CHARS
    assert result.scored[0].text_chars == len(analyzer.seen[0])


def test_chunked_strategy_scores_the_whole_thread(analyzer):
    comments = ["bad " * 100, "good " * 300]
    source = FakeSource({"Reps": [make_post(1, title="x", selftext="")]}, comments={"p1": comments})
    result = Scraper(source, analyzer, delay=0, max_chars=500, strategy="chunked").run(["Reps"])
    assert len(analyzer.seen) > 1
    assert result.scored[0].sentiment.label == POSITIVE  # 'good' dominates by length


def test_preloaded_comments_skip_fetch(analyzer):
    post = make_post(1, comments=["preloaded good"])
    source = FakeSource({"Reps": [post]})
    Scraper(source, analyzer, delay=0).run(["Reps"])
    assert not [c for c in source.calls if c[0] == "comments"]


def test_empty_thread_is_error(analyzer):
    source = FakeSource({"Reps": [make_post(1, title="", selftext="")]})
    result = Scraper(source, analyzer, delay=0).run(["Reps"])
    assert result.scored[0].sentiment.is_error


def test_to_frame_has_documented_columns_first(analyzer):
    source = FakeSource({"Reps": [make_post(1)]})
    frame = Scraper(source, analyzer, delay=0).run(["Reps"]).to_frame()
    assert list(frame.columns[: len(CORE_COLUMNS)]) == CORE_COLUMNS
    row = frame.iloc[0]
    assert row["created_date"] == "2025-05-28 03:40:00"
    assert row["post_id"] == "p1" and row["model"] == "keyword-test"


def test_bad_arguments(analyzer):
    with pytest.raises(ValueError):
        Scraper(FakeSource({}), analyzer, strategy="nope")
    with pytest.raises(ValueError):
        Scraper(FakeSource({}), analyzer, max_chars=0)
