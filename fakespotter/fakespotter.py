"""Backward-compatible entry point: ``python fakespotter/fakespotter.py``.

Does exactly what the original script did (scores the 100 hot posts of r/Reps with
DistilBERT and writes ``fakewatches_deep_sentiment.csv`` to the current folder), now
through the fakespotter package. Any extra arguments are passed to ``fakespotter scrape``,
so ``python fakespotter/fakespotter.py --limit 10`` works too. For everything else, use
the ``fakespotter`` command (see the README).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    # Running this file directly puts this folder first on sys.path, where this very file
    # would shadow the ``fakespotter`` package. Point the path at the repository root.
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.path.insert(0, str(here.parent))

    from fakespotter.cli import main

    defaults = [
        "scrape",
        "--subreddit",
        "Reps",
        "--limit",
        "100",
        "--output",
        "fakewatches_deep_sentiment.csv",
    ]
    raise SystemExit(main(defaults + sys.argv[1:]))
