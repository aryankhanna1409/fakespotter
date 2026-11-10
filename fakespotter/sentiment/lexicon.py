"""A small, dependency-free lexicon scorer.

Rule-based in the spirit of VADER: word valences, negation ("not good"), intensifiers
("really bad"), dampeners ("kinda nice") and a handful of terms from replica-buying
communities ("GL" = green light, "RL" = red light, "scam"). It is deliberately simple
and transparent. It exists so fakespotter runs anywhere, fast; DistilBERT is the
reference model.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from fakespotter.models import NEGATIVE, POSITIVE, SentimentResult
from fakespotter.sentiment.base import SentimentAnalyzer

_POSITIVE = """
amazing awesome beautiful best better brilliant clean comfortable comfy cool crisp decent
delighted durable easy excellent fantastic fast favorite favourite fine fire flawless fresh
gl glad good gorgeous great happy helpful impressed incredible insane legit like liked love
loved lovely nice perfect pleased pretty quality quick recommend recommended reliable
satisfied smooth solid steal stunning super sweet thank thanks thrilled top trusted
w win wonderful worth wow accurate affordable arrived banger beast bargain clutch dope
excited exceptional fair gem goated heat helpful hyped neat on-point outstanding premium
safe spot-on superb tight underrated valid
""".split()

_NEGATIVE = """
annoyed annoying awful bad broke broken cheap cheated complain complaint confiscated
crap damaged defective delay delayed disappointed disappointing disaster dodgy fail failed
fake faulty flaw flawed frustrated garbage hate hated horrible issue issues l lost mess
missing nightmare off overpriced poor problem problems ripoff rip-off rl rubbish sad scam
scammed scammer seized shady sketchy slow sucks terrible trash ugly unhappy unreliable
upset useless wasted worse worst wrong avoid beware bummer chargeback cringe dissapointed
dissapointing fraud gross janky lame lousy meh mid misleading refund regret ruined stolen
stuck tariff tariffs taxed unacceptable
""".split()

LEXICON: dict[str, float] = {w: 1.0 for w in _POSITIVE} | {w: -1.0 for w in _NEGATIVE}
# A few strong words get extra weight.
LEXICON.update(
    {
        "love": 1.5,
        "perfect": 1.5,
        "flawless": 1.5,
        "amazing": 1.5,
        "excellent": 1.5,
        "best": 1.3,
        "goated": 1.5,
        "scam": -2.0,
        "scammed": -2.0,
        "scammer": -2.0,
        "fraud": -2.0,
        "worst": -1.8,
        "terrible": -1.6,
        "awful": -1.6,
        "horrible": -1.6,
        "seized": -1.4,
        "confiscated": -1.4,
        "avoid": -1.4,
        "garbage": -1.5,
        "trash": -1.5,
    }
)

NEGATIONS = {
    "not",
    "no",
    "never",
    "nothing",
    "nobody",
    "none",
    "neither",
    "nor",
    "without",
    "cannot",
    "cant",
    "dont",
    "doesnt",
    "didnt",
    "isnt",
    "wasnt",
    "arent",
    "werent",
    "wont",
    "wouldnt",
    "shouldnt",
    "couldnt",
    "aint",
    "hardly",
}
INTENSIFIERS = {
    "very": 1.4,
    "really": 1.4,
    "so": 1.3,
    "super": 1.4,
    "extremely": 1.6,
    "totally": 1.3,
    "absolutely": 1.5,
    "incredibly": 1.5,
    "insanely": 1.5,
    "highly": 1.3,
    "too": 1.2,
    "kinda": 0.6,
    "somewhat": 0.6,
    "slightly": 0.5,
    "barely": 0.4,
    "fairly": 0.8,
}
_TOKEN = re.compile(r"[a-z][a-z'\-]*|[!?]")
_POS_EMOJI = set("❤😍🔥🙌👍🤩😊😁🥰💯✅")
_NEG_EMOJI = set("👎😡😠😢😭💀🤬❌")


def tokenize(text: str) -> list[str]:
    return [t.replace("'", "") for t in _TOKEN.findall(text.lower())]


class LexiconAnalyzer(SentimentAnalyzer):
    name = "lexicon"

    def __init__(self, *, steepness: float = 1.2, window: int = 3) -> None:
        self.steepness = steepness
        self.window = window

    def raw_score(self, text: str) -> float:
        """Average valence per sentiment-bearing word (negative < 0 < positive)."""
        tokens = tokenize(text)
        total = 0.0
        hits = 0
        for i, token in enumerate(tokens):
            valence = LEXICON.get(token)
            if valence is None:
                continue
            context = tokens[max(0, i - self.window) : i]
            if any(t in NEGATIONS for t in context):
                valence *= -0.75
            for t in context[-2:]:
                valence *= INTENSIFIERS.get(t, 1.0)
            # "!!!" after a word amplifies it a little.
            if i + 1 < len(tokens) and tokens[i + 1] == "!":
                valence *= 1.15
            total += valence
            hits += 1
        emoji = sum(ch in _POS_EMOJI for ch in text) - sum(ch in _NEG_EMOJI for ch in text)
        total += 0.8 * emoji
        hits += abs(emoji)
        if hits == 0:
            return 0.0
        # Dampen by sqrt so one strong word in a long thread doesn't dominate.
        return total / math.sqrt(hits)

    def analyze(self, texts: Sequence[str]) -> list[SentimentResult]:
        results = []
        for text in texts:
            raw = self.raw_score(text)
            p_pos = 1.0 / (1.0 + math.exp(-self.steepness * raw))
            if p_pos >= 0.5:
                results.append(SentimentResult(POSITIVE, p_pos))
            else:
                results.append(SentimentResult(NEGATIVE, 1.0 - p_pos))
        return results
