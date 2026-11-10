import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from fakespotter.config import RedditCredentials
from fakespotter.sources import SourceError, get_source
from fakespotter.sources.public import PublicJsonSource, flatten_comments
from fakespotter.sources.reddit import PrawSource, to_post


def test_flatten_comments_walks_tree_and_skips_more(comments_json):
    assert list(flatten_comments(comments_json[1])) == [
        "Looks great, GL!",
        "Agreed, really clean",
        "[deleted]",
        "Which seller?",
    ]


def test_public_source_lists_posts(listing_json):
    urls = []

    def fetch(url):
        urls.append(url)
        return listing_json

    source = PublicJsonSource(fetch=fetch)
    posts = list(source.iter_posts("Reps", sort="top", limit=10, time_filter="month"))
    assert [p.id for p in posts] == ["x1", "x2", "x3"]  # the stray t1 child is ignored
    assert posts[0].permalink == "https://www.reddit.com/r/Reps/comments/x1/slug_1/"
    assert posts[0].subreddit == "Reps" and posts[0].author == "user1"
    assert "/r/Reps/top.json" in urls[0] and "t=month" in urls[0] and "limit=10" in urls[0]


def test_public_source_paginates_and_stops_at_limit():
    pages = {
        None: {
            "data": {
                "after": "t3_b",
                "children": [
                    {"kind": "t3", "data": {"id": f"a{i}", "title": "t", "created_utc": 1}}
                    for i in range(3)
                ],
            }
        },
        "t3_b": {
            "data": {
                "after": "t3_c",
                "children": [
                    {"kind": "t3", "data": {"id": f"b{i}", "title": "t", "created_utc": 1}}
                    for i in range(3)
                ],
            }
        },
    }

    def fetch(url):
        after = url.split("after=")[1].split("&")[0] if "after=" in url else None
        return pages[after]

    posts = list(PublicJsonSource(fetch=fetch).iter_posts("Reps", limit=5))
    assert [p.id for p in posts] == ["a0", "a1", "a2", "b0", "b1"]


def test_public_source_fetches_comments(comments_json):
    seen = []

    def fetch(url):
        seen.append(url)
        return comments_json

    source = PublicJsonSource(fetch=fetch)
    post = to_post(
        SimpleNamespace(id="x1", title="t", created_utc=1, permalink="/r/Reps/comments/x1/slug/")
    )
    assert source.fetch_comments(post, limit=2) == ["Looks great, GL!", "Agreed, really clean"]
    assert seen[0].startswith("https://www.reddit.com/r/Reps/comments/x1/slug.json")


def _http_error(code):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b""))


def test_public_source_retries_on_429_then_succeeds():
    calls = {"n": 0}
    sleeps = []

    def fetch(url):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(429)
        return {"data": {"children": []}}

    source = PublicJsonSource(fetch=fetch, sleep=sleeps.append, backoff=1)
    assert list(source.iter_posts("Reps")) == []
    assert sleeps == [1, 2]


def test_public_source_explains_403():
    source = PublicJsonSource(
        fetch=lambda url: (_ for _ in ()).throw(_http_error(403)), sleep=lambda s: None
    )
    with pytest.raises(SourceError, match="REDDIT_CLIENT_ID"):
        list(source.iter_posts("Reps"))


def test_public_source_rejects_bad_sort():
    with pytest.raises(SourceError):
        list(PublicJsonSource(fetch=lambda u: {}).iter_posts("Reps", sort="best"))


class FakeComments:
    def __init__(self, bodies):
        self.bodies = bodies
        self.replace_more_calls = []

    def replace_more(self, limit):
        self.replace_more_calls.append(limit)

    def list(self):
        return [SimpleNamespace(body=b) for b in self.bodies]


class FakeSubreddit:
    def __init__(self, submissions):
        self.submissions = submissions
        self.calls = []

    def hot(self, **kwargs):
        self.calls.append(("hot", kwargs))
        return iter(self.submissions[: kwargs["limit"]])

    def top(self, **kwargs):
        self.calls.append(("top", kwargs))
        return iter(self.submissions[: kwargs["limit"]])


def _submission(i):
    return SimpleNamespace(
        id=f"s{i}",
        title=f"Title {i}",
        selftext="body",
        score=i,
        num_comments=2,
        url=f"https://i.redd.it/{i}.jpg",
        created_utc=1700000000 + i,
        subreddit=SimpleNamespace(display_name="Reps"),
        permalink=f"/r/Reps/comments/s{i}/t/",
        author="someone",
        comments=FakeComments([f"comment {i}a", f"comment {i}b"]),
    )


def test_praw_source_lists_and_fetches_comments():
    sub = FakeSubreddit([_submission(i) for i in range(5)])
    reddit = SimpleNamespace(subreddit=lambda name: sub)
    source = PrawSource(reddit=reddit)
    posts = list(source.iter_posts("Reps", sort="top", limit=3, time_filter="year"))
    assert [p.id for p in posts] == ["s0", "s1", "s2"]
    assert sub.calls == [("top", {"limit": 3, "time_filter": "year"})]
    assert posts[1].permalink == "https://www.reddit.com/r/Reps/comments/s1/t/"
    assert source.fetch_comments(posts[1]) == ["comment 1a", "comment 1b"]


def test_praw_source_hot_has_no_time_filter():
    sub = FakeSubreddit([_submission(0)])
    source = PrawSource(reddit=SimpleNamespace(subreddit=lambda name: sub))
    list(source.iter_posts("Reps", sort="hot", limit=1))
    assert sub.calls == [("hot", {"limit": 1})]


def test_get_source_auto(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    assert get_source("auto").name == "public"
    with pytest.raises(SourceError, match="REDDIT_CLIENT_ID"):
        get_source("praw")
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    assert get_source("auto").name == "praw"
    with pytest.raises(SourceError):
        get_source("twitter")


def test_credentials_from_env(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", " id ")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "ua/1.0")
    assert RedditCredentials.from_env() == RedditCredentials("id", "secret", "ua/1.0")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "")
    assert RedditCredentials.from_env() is None


def test_listing_json_fixture_is_valid(listing_json):
    json.dumps(listing_json)
