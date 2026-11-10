# Architecture

fakespotter is a small pipeline with one job per module. Each stage talks to the next
through plain data classes (`fakespotter/models.py`), so any stage can be swapped or
faked in tests.

```
              ┌──────────────┐     ┌───────────────┐     ┌──────────────────┐     ┌──────────────┐
 subreddits ─▶│  Source      │──▶  │  text.py      │──▶  │ SentimentAnalyzer│──▶  │  export.py   │──▶ CSV / JSONL
              │ praw | public│Post │ build + trim  │ str │ DistilBERT|lexicon│Res. │ write/merge  │
              └──────────────┘     └───────────────┘     └──────────────────┘     └──────┬───────┘
                         ▲                 pipeline.Scraper drives the loop               │
                         └─ rate-limit sleep between posts, ERROR on any failure          ▼
                                                                         analysis.py ─▶ report/ (terminal, HTML)
```

## Stages

### Sources (`fakespotter/sources/`)

`Source` has two methods: `iter_posts(subreddit, sort, limit, time_filter)` yields
`Post` objects, and `fetch_comments(post, limit)` returns comment bodies. Comments are
fetched lazily, one post at a time, so the rate-limit delay applies to the expensive
calls.

* `PrawSource` wraps PRAW in read-only mode. `replace_more=0` (the default, as in the
  original script) keeps only the comments Reddit returns with the post;
  `replace_more=None` expands every "load more comments" link.
* `PublicJsonSource` uses `urllib` only. It paginates listings with Reddit's `after`
  cursor (100 per page), retries 429 and 5xx responses with exponential backoff, and
  walks the comment tree depth-first. A 401/403 produces an error that explains how to
  switch to API keys.

`get_source("auto")` picks PRAW when `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` are
set (from the environment or a `.env` file), otherwise the public source.

### Text (`fakespotter/text.py`)

`build_thread_text` joins title, body and comments, collapses whitespace, drops bare
URLs and `[removed]`/`[deleted]` placeholders, and cuts to `max_chars` on a word
boundary. `chunk` splits a whole thread into word-aligned pieces for the chunked
strategy.

### Sentiment (`fakespotter/sentiment/`)

A `SentimentAnalyzer` takes a list of texts and returns `SentimentResult(label, score)`.

* `TransformerAnalyzer` loads the Hugging Face pipeline on first use and calls it with
  `truncation=True, max_length=512`. Without that, any text over 512 tokens raises an
  exception inside the model. Labels are normalised (`LABEL_1` → `POSITIVE`, …).
* `LexiconAnalyzer` is a transparent rule-based scorer (valences, negation, intensifiers,
  emoji and a few community terms) mapped through a logistic to a confidence in
  [0.5, 1]. It exists for machines without PyTorch, quick dry runs and tests.

**Chunked strategy.** Each chunk's verdict is converted to P(positive), the
probabilities are averaged weighted by chunk length, and the mean is converted back to a
label and confidence (`sentiment.aggregate`). At most 8 chunks per thread are scored.

### Pipeline (`fakespotter/pipeline.py`)

`Scraper.run` iterates subreddits and posts, sleeps `delay` seconds between posts (not
after the last), and turns any exception while fetching comments or scoring into an
`ERROR` / `0.0` row with the reason attached, so one bad thread never ends a run. A
listing failure for one subreddit is recorded in `RunResult.errors` and the run moves to
the next. Callbacks report progress to the CLI.

### Export (`fakespotter/export.py`)

Rows keep the original script's eight columns first and in the same order; extra
columns follow. `--append` merges with the existing file by `post_id` (falling back to
URL), keeping the newest row, except that a fresh `ERROR` never replaces an earlier real
verdict. `read_results` also normalises older files: it parses dates and infers
`post_id` and `subreddit` from URLs.

### Analysis (`fakespotter/analysis.py`)

Pure functions over a DataFrame:

* `summarize` gives counts, net sentiment, engagement-weighted net sentiment, mean
  confidence, uncertain verdicts and Spearman correlations (computed as Pearson on
  ranks, so SciPy isn't needed).
* `timeline` buckets by day, week (Monday start) or month depending on the span, and
  fills empty periods so gaps stay visible.
* `distinctive_terms` ranks title words by the weighted log-odds ratio with an
  informative Dirichlet prior, z-scored (Monroe, Colaresi & Quinn, 2008, "Fightin'
  Words"). Counts are per thread, so a word repeated in one title counts once.

### Reports (`fakespotter/report/`)

* `terminal.py` prints a rich panel (metrics, verdict bar, volume sparkline, leaning
  words) and "most negative / positive / discussed" tables.
* `html.py` renders one self-contained file: inline SVG charts sized to their cards, a
  hover tooltip layer, and a searchable, filterable, sortable thread table fed from an
  embedded JSON block. Colours are CSS custom properties with separate light and dark
  values (blue = positive, red = negative, grey = error), following the viewer's OS
  theme with a manual toggle. Every title is HTML-escaped, the embedded JSON escapes
  `</`, and table links are limited to `http(s)` URLs.

## Testing

`tests/conftest.py` provides a `FakeSource` and a deterministic `KeywordAnalyzer`, so
the pipeline, CLI and reports are tested without network access or model downloads.
Reddit JSON fixtures live in `tests/fixtures/`. Tests marked `model` load the real
DistilBERT and are run with `pytest -m model` (a separate CI job).
