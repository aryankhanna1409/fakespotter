import pandas as pd
import pytest

from fakespotter import analysis as A
from fakespotter.models import NEGATIVE, POSITIVE


def test_summary_numbers(results_frame):
    s = A.summarize(results_frame)
    assert (s.total, s.scored) == (6, 5)
    assert s.counts == {"POSITIVE": 2, "NEGATIVE": 3, "ERROR": 1}
    assert s.net_sentiment == pytest.approx(-0.2)
    assert s.share(POSITIVE) == pytest.approx(0.4)
    assert s.error_rate == pytest.approx(1 / 6)
    assert s.uncertain == 1
    assert s.subreddits == ["FashionReps", "Reps"]
    assert s.mood == "mostly negative"
    # Engagement weights: positives 61 + 25 = 86, negatives 36 + 15 + 10 = 61.
    assert s.weighted_net_sentiment == pytest.approx((86 - 61) / (86 + 61))


def test_mood_bands():
    def mood(n):
        return A.Summary(1, 1, {}, n, None, None, {}, 0, None, None, [], None, None).mood

    assert [mood(x) for x in (-0.9, -0.3, 0.0, 0.3, 0.9, None)] == [
        "strongly negative",
        "mostly negative",
        "mixed",
        "mostly positive",
        "strongly positive",
        "no scored threads",
    ]


def test_spearman_matches_rank_correlation():
    a = pd.Series([1, 2, 3, 4, 5])
    assert A._spearman(a, a**3) == pytest.approx(1.0)
    assert A._spearman(a, -a) == pytest.approx(-1.0)
    assert A._spearman(a[:2], a[:2]) is None


def test_timeline_daily_buckets_and_gaps(results_frame):
    tl = A.timeline(results_frame)
    assert A.pick_frequency(results_frame) == "D"
    assert tl["period"].dt.strftime("%m-%d").tolist() == ["05-20", "05-21", "05-22", "05-23"]
    assert tl[POSITIVE].tolist() == [1, 1, 0, 0]
    assert tl["ERROR"].tolist() == [0, 0, 0, 1]
    assert tl["net"].iloc[1] == pytest.approx(0.0)


def test_timeline_weekly_for_longer_spans(results_frame):
    frame = results_frame.copy()
    frame.loc[0, "created_date"] = pd.Timestamp("2025-03-01")
    assert A.pick_frequency(frame) == "W"
    tl = A.timeline(frame)
    assert (tl["period"].dt.dayofweek == 0).all()  # weeks start on Monday
    assert tl[[POSITIVE, NEGATIVE, "ERROR"]].to_numpy().sum() == len(frame)


def test_top_threads_and_most_discussed(results_frame):
    assert A.top_threads(results_frame, NEGATIVE, 2)["post_id"].tolist() == ["a3", "a4"]
    assert A.most_discussed(results_frame, 1)["post_id"].tolist() == ["a6"]


def test_by_subreddit(results_frame):
    table = A.by_subreddit(results_frame).set_index("subreddit")
    assert table.loc["Reps", "threads"] == 4
    assert table.loc["FashionReps", "net"] == pytest.approx(-1.0)


def test_distinctive_terms(results_frame):
    pos, neg = A.distinctive_terms(results_frame, min_count=1)
    assert pos and neg
    assert "scam" in [t.term for t in neg]
    assert all(t.z > 0 for t in pos) and all(t.z < 0 for t in neg)


def test_frequent_terms_ignores_stopwords(results_frame):
    terms = dict(A.frequent_terms(results_frame))
    assert terms["haul"] == 3
    assert "for" not in terms and "my" not in terms


def test_requires_sentiment_columns():
    with pytest.raises(A.NoSentimentError):
        A.summarize(pd.DataFrame({"title": ["x"]}))


def test_sample_data_summary():
    from fakespotter.cli import sample_path
    from fakespotter.export import read_results

    s = A.summarize(read_results(sample_path()))
    assert s.counts == {"POSITIVE": 3, "NEGATIVE": 29, "ERROR": 68}


def test_spearman_constant_input_is_none():
    assert A._spearman(pd.Series([1, 1, 1, 1]), pd.Series([1, 2, 3, 4])) is None
