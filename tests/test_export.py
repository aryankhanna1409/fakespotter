import pandas as pd

from fakespotter.config import load_dotenv
from fakespotter.export import CORE_COLUMNS, existing_ids, merge, read_results, write_results


def _row(pid, label="NEGATIVE", score=0.9, utc=1.0, title="t"):
    return {
        "title": title,
        "score": 1,
        "num_comments": 1,
        "url": f"https://www.reddit.com/r/Reps/comments/{pid}/x/",
        "created_utc": utc,
        "sentiment_label": label,
        "sentiment_score": score,
        "created_date": "2025-01-01 00:00:00",
        "post_id": pid,
    }


def test_csv_roundtrip(tmp_path):
    path = tmp_path / "out.csv"
    write_results(pd.DataFrame([_row("a"), _row("b")]), path)
    back = read_results(path)
    assert list(back.columns[: len(CORE_COLUMNS)]) == CORE_COLUMNS
    assert back["post_id"].tolist() == ["a", "b"]
    assert pd.api.types.is_datetime64_any_dtype(back["created_date"])


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "out.jsonl"
    write_results(pd.DataFrame([_row("a", title="emoji 🔥")]), path)
    assert "🔥" in path.read_text(encoding="utf-8")
    assert read_results(path)["title"].tolist() == ["emoji 🔥"]


def test_append_merges_keeping_newest(tmp_path):
    path = tmp_path / "out.csv"
    write_results(pd.DataFrame([_row("a", score=0.6), _row("b")]), path)
    merged = write_results(pd.DataFrame([_row("a", score=0.99), _row("c")]), path, append=True)
    assert sorted(merged["post_id"]) == ["a", "b", "c"]
    assert merged.set_index("post_id").loc["a", "sentiment_score"] == 0.99
    assert len(read_results(path)) == 3


def test_merge_never_replaces_a_verdict_with_an_error():
    old = pd.DataFrame([_row("a", label="POSITIVE", score=0.8)])
    new = pd.DataFrame([_row("a", label="ERROR", score=0.0)])
    assert merge(old, new)["sentiment_label"].tolist() == ["POSITIVE"]


def test_existing_ids_only_counts_successes(tmp_path):
    path = tmp_path / "out.csv"
    write_results(pd.DataFrame([_row("a"), _row("b", label="ERROR", score=0.0)]), path)
    assert existing_ids(path) == {"a"}
    assert existing_ids(tmp_path / "missing.csv") == set()


def test_reads_bundled_samples_and_infers_columns():
    from fakespotter.cli import sample_path

    deep = read_results(sample_path())
    assert len(deep) == 100
    assert set(deep["subreddit"]) == {"Reps"}  # inferred from URLs; gallery links filled in
    assert deep["post_id"].notna().sum() > 0
    old = read_results(sample_path("fake_watches_reddit_posts.csv"))
    assert "created_date" in old and len(old) == 30


def test_load_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        '# comment\nexport FS_A="quoted value"\nFS_B=plain # trailing\nFS_C=keep\nnot a pair\n'
    )
    monkeypatch.setenv("FS_C", "already set")
    monkeypatch.delenv("FS_A", raising=False)
    monkeypatch.delenv("FS_B", raising=False)
    loaded = load_dotenv(env)
    import os

    assert os.environ["FS_A"] == "quoted value"
    assert os.environ["FS_B"] == "plain"
    assert os.environ["FS_C"] == "already set"  # existing env wins
    assert loaded["FS_C"] == "keep"
    assert load_dotenv(tmp_path / "missing") == {}
