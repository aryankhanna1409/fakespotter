"""fakespotter: track how Reddit talks about counterfeit goods and score every thread.

The package is organised as a small pipeline::

    Source (Reddit API / public JSON)  ->  thread text  ->  SentimentAnalyzer  ->  CSV / report

Quick use from Python::

    from fakespotter import Scraper, get_analyzer, get_source

    scraper = Scraper(get_source("auto"), get_analyzer("transformers"))
    frame = scraper.run(["Reps"], limit=25).to_frame()
"""

from fakespotter.models import Post, ScoredPost, SentimentResult
from fakespotter.pipeline import RunResult, Scraper
from fakespotter.sentiment import get_analyzer
from fakespotter.sources import get_source

__version__ = "1.0.0"

__all__ = [
    "Post",
    "RunResult",
    "Scraper",
    "ScoredPost",
    "SentimentResult",
    "__version__",
    "get_analyzer",
    "get_source",
]
