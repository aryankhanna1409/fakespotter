import pytest

from fakespotter.text import build_thread_text, chunk, clean, truncate


def test_clean_collapses_whitespace_and_drops_urls():
    assert clean("  hello\n\n  world https://x.com/a?b=1 end ") == "hello world end"
    assert clean(None) == ""


def test_truncate_cuts_on_word_boundary():
    assert truncate("one two three four", 9) == "one two"
    assert truncate("short", 100) == "short"
    assert truncate("exactly", 7) == "exactly"


def test_truncate_keeps_word_ending_at_limit():
    # The character after the cut is a space, so "two" is a whole word and stays.
    assert truncate("one two three", 7) == "one two"


def test_truncate_single_long_word_is_hard_cut():
    assert truncate("abcdefghij", 4) == "abcd"


def test_build_thread_text_combines_title_body_comments():
    text = build_thread_text("Title", "Body here", ["first", "second"])
    assert text == "Title Body here first second"


def test_build_thread_text_skips_removed_and_deleted():
    text = build_thread_text("Title", "[removed]", ["[deleted]", "kept"])
    assert text == "Title kept"


def test_build_thread_text_respects_max_chars():
    text = build_thread_text("t", "word " * 1000, [], max_chars=2048)
    assert len(text) <= 2048
    assert not text.endswith(" ")
    assert len(build_thread_text("t", "word " * 1000, [], max_chars=None)) > 2048


def test_chunk_covers_all_words_in_order():
    text = " ".join(f"w{i}" for i in range(500))
    pieces = chunk(text, 100)
    assert all(len(p) <= 100 for p in pieces)
    assert " ".join(pieces) == text


def test_chunk_rejects_bad_size():
    with pytest.raises(ValueError):
        chunk("abc", 0)
    assert chunk("", 10) == []
