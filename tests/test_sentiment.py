import pytest

from fakespotter.models import ERROR, NEGATIVE, POSITIVE, SentimentResult
from fakespotter.sentiment import AnalyzerError, aggregate, get_analyzer
from fakespotter.sentiment.lexicon import LexiconAnalyzer
from fakespotter.sentiment.transformer import DEFAULT_MODEL, TransformerAnalyzer


def test_sentiment_result_helpers():
    assert SentimentResult(POSITIVE, 0.9).signed == 0.9
    assert SentimentResult(NEGATIVE, 0.9).signed == -0.9
    assert SentimentResult(NEGATIVE, 0.9).positive_probability == pytest.approx(0.1)
    err = SentimentResult.error()
    assert (err.label, err.score, err.signed, err.is_error) == (ERROR, 0.0, None, True)


@pytest.mark.parametrize(
    "text,label",
    [
        ("This haul is amazing, I love it 🔥", POSITIVE),
        ("AVOID this seller, total scam", NEGATIVE),
        ("not good at all, really disappointed", NEGATIVE),
        ("GL, looks really clean", POSITIVE),
        ("Package got seized by customs, worst experience", NEGATIVE),
    ],
)
def test_lexicon_labels(text, label):
    assert LexiconAnalyzer().analyze_one(text).label == label


def test_lexicon_negation_flips_polarity():
    lex = LexiconAnalyzer()
    assert lex.raw_score("this is good") > 0
    assert lex.raw_score("this is not good") < 0


def test_lexicon_intensifier_strengthens():
    lex = LexiconAnalyzer()
    assert lex.raw_score("really bad") < lex.raw_score("bad") < 0


def test_lexicon_scores_are_confidences():
    for result in LexiconAnalyzer().analyze(["", "meh", "love love love"]):
        assert 0.5 <= result.score <= 1.0


def test_aggregate_weights_by_length():
    results = [SentimentResult(POSITIVE, 0.9), SentimentResult(NEGATIVE, 0.9)]
    assert aggregate(results, [3, 1]).label == POSITIVE
    assert aggregate(results, [1, 3]).label == NEGATIVE
    agg = aggregate(results, [3, 1])
    assert agg.score == pytest.approx((0.9 * 3 + 0.1 * 1) / 4)


def test_aggregate_ignores_errors_and_handles_all_errors():
    assert aggregate([SentimentResult.error(), SentimentResult(POSITIVE, 0.8)]) == SentimentResult(
        POSITIVE, 0.8
    )
    assert aggregate([SentimentResult.error()]).is_error
    with pytest.raises(ValueError):
        aggregate([SentimentResult(POSITIVE, 0.8)], [1, 2])


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [{"label": "LABEL_1" if "yay" in t else "NEGATIVE", "score": 0.77} for t in texts]


def test_transformer_analyzer_truncates_by_tokens_and_maps_labels():
    pipe = FakePipeline()
    analyzer = TransformerAnalyzer(pipeline=pipe)
    out = analyzer.analyze(["yay", "boo"])
    assert out == [SentimentResult(POSITIVE, 0.77), SentimentResult(NEGATIVE, 0.77)]
    _, kwargs = pipe.calls[0]
    assert kwargs["truncation"] is True and kwargs["max_length"] == 512
    assert analyzer.analyze([]) == []
    assert analyzer.name == DEFAULT_MODEL.rsplit("/", 1)[-1]


def test_transformer_rejects_unknown_labels():
    analyzer = TransformerAnalyzer(
        pipeline=lambda texts, **kw: [{"label": "NEUTRAL", "score": 0.5}]
    )
    with pytest.raises(AnalyzerError):
        analyzer.analyze(["x"])


def test_transformer_handles_top_k_style_output():
    analyzer = TransformerAnalyzer(
        pipeline=lambda texts, **kw: [
            [{"label": "NEGATIVE", "score": 0.3}, {"label": "POSITIVE", "score": 0.7}]
        ]
    )
    assert analyzer.analyze_one("x") == SentimentResult(POSITIVE, 0.7)


def test_get_analyzer():
    assert isinstance(get_analyzer("lexicon"), LexiconAnalyzer)
    assert isinstance(get_analyzer("distilbert"), TransformerAnalyzer)
    with pytest.raises(AnalyzerError):
        get_analyzer("nope")


@pytest.mark.model
def test_real_distilbert():  # pragma: no cover - opt in with: pytest -m model
    pytest.importorskip("transformers")
    analyzer = get_analyzer("transformers").load()
    assert analyzer.analyze_one("I absolutely love this, it is wonderful").label == POSITIVE
    assert analyzer.analyze_one("This was a terrible, awful experience").label == NEGATIVE
    # Far beyond 512 tokens: must not error now that truncation is by tokens.
    assert not analyzer.analyze_one("great haul 🔥 " * 800).is_error
