<div align="center">

<picture><source media="(prefers-color-scheme: light)" srcset="assets/banner-light.svg" /><img src="assets/banner.svg" width="100%" alt="fakespotter: Reddit sentiment on counterfeit goods, via DistilBERT" /></picture>

**Tracks how Reddit talks about counterfeit goods and scores the sentiment of every thread.**

![Python](https://img.shields.io/badge/Python-3.10%2B-8a5a22?style=flat-square&logo=python&labelColor=1d120b&logoColor=efdfc4)
![PRAW](https://img.shields.io/badge/PRAW-Reddit%20API-8a5a22?style=flat-square&logo=reddit&labelColor=1d120b&logoColor=efdfc4)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-DistilBERT-8a5a22?style=flat-square&labelColor=1d120b)
![pandas](https://img.shields.io/badge/pandas-CSV%20export-8a5a22?style=flat-square&logo=pandas&labelColor=1d120b&logoColor=efdfc4)
![CI](https://github.com/aryankhanna1409/fakespotter/actions/workflows/ci.yml/badge.svg)

</div>

---

## What it does

1. Pulls the current **hot posts** from one or more subreddits (the example run uses `r/Reps`, a replica community) through the official Reddit API, or through Reddit's public JSON if you have no API keys.
2. For each post, combines the title, body and all comments into one text, trimmed to about 2,000 characters.
3. Scores it with a **DistilBERT sentiment model** (fine-tuned on SST-2), giving a `POSITIVE` or `NEGATIVE` label and a confidence score.
4. Saves everything to a CSV (or JSON Lines) so you can sort, chart and compare threads.
5. Turns any results file into a **terminal summary** and a **self-contained HTML report**: net sentiment, trends over time, how confident the model is, whether engagement tracks sentiment, the title words that tip the balance, and a searchable table of every thread.

Posts are fetched slowly (a 2-second pause between each, configurable) to respect Reddit's rate limits. If a thread can't be scored, its label is `ERROR` with a score of `0.0` and the run carries on.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/report-dark.png">
    <img src="assets/report-light.png" width="90%" alt="The HTML report for the bundled r/Reps sample: net sentiment -0.81, verdict split and threads over time">
  </picture>
  <br><sub>The HTML report for the bundled sample (<code>fakespotter demo</code>). One file, no external requests, light and dark themes.</sub>
</p>

---

## Quick start

```bash
git clone https://github.com/aryankhanna1409/fakespotter.git
cd fakespotter

pip install -r requirements.txt   # or: pip install -e ".[ml]"
```

See it work straight away, no credentials needed:

```bash
fakespotter demo --open           # report on the bundled 100-thread r/Reps sample
```

Then score live threads:

1. Create a "script" app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) to get a client ID and secret.
2. Set them as environment variables (or copy `.env.example` to `.env`, which fakespotter reads automatically):

```bash
export REDDIT_CLIENT_ID="your-client-id"
export REDDIT_CLIENT_SECRET="your-client-secret"
# optional: export REDDIT_USER_AGENT="FakeSpotter/1.0 (by u/your-username)"
```

3. Run it:

```bash
fakespotter scrape --html report.html
```

That reads the 100 hot posts of r/Reps, writes `fakewatches_deep_sentiment.csv` in the folder you run from, prints a summary and saves an HTML report. The first run downloads the sentiment model (about 270 MB).

> The original command still works and does exactly the same: `python fakespotter/fakespotter.py`.

`fakespotter doctor` checks your installation and credentials, and `fakespotter doctor --load-model` also loads DistilBERT.

---

## Commands

| Command | What it does |
|---|---|
| `fakespotter scrape` | Fetch posts, score every thread, save a CSV/JSONL, optionally an HTML report |
| `fakespotter report FILE` | Summarise a results file in the terminal; `--html out.html` for the full report |
| `fakespotter demo` | Build the report from the bundled sample, no credentials needed |
| `fakespotter score "TEXT" …` | Score any text (one text per argument, or one per line on stdin) |
| `fakespotter doctor` | Check dependencies, credentials and (with `--load-model`) the model |

Also available as `python -m fakespotter …`.

### Recipes

```bash
# Several communities at once; top posts of the past month
fakespotter scrape -s Reps -s FashionReps -s RepTime --sort top --time month -n 50

# Track a community over time: run daily, merging into one file and only scoring new posts
fakespotter scrape -s RepTime --sort new --append --skip-existing -o reptime.csv

# Score whole threads, not just the first ~2,000 characters
fakespotter scrape --strategy chunked

# No PyTorch? The lexicon backend is instant (less accurate; good for exploring)
fakespotter scrape --backend lexicon --delay 1

# Any Hugging Face sentiment checkpoint with POSITIVE/NEGATIVE labels
fakespotter scrape --model distilbert/distilbert-base-uncased-finetuned-sst-2-english

# Re-render a report from an existing file and open it
fakespotter report reptime.csv --html reptime.html --open
```

### `scrape` options

| Option | Default | Meaning |
|---|---|---|
| `-s, --subreddit` | `Reps` | Subreddit(s) to read; repeat or comma-separate |
| `-n, --limit` | `100` | Posts per subreddit |
| `--sort` | `hot` | `hot`, `new`, `top`, `rising` or `controversial` |
| `--time` | `week` | Window for `top`/`controversial`: `hour` … `all` |
| `--source` | `auto` | `praw` (API keys), `public` (no keys), `auto` picks `praw` when keys are set |
| `-b, --backend` | `transformers` | `transformers` (DistilBERT) or `lexicon` |
| `--strategy` | `truncate` | `truncate` scores the first `--max-chars`; `chunked` scores the whole thread in pieces and averages them |
| `--max-chars` | `2048` | Characters per scored text |
| `--comment-limit` | all | Use at most this many comments per thread |
| `--delay` | `2` | Seconds between posts |
| `-o, --output` | `fakewatches_deep_sentiment.csv` | `.csv` or `.jsonl` |
| `--append` | off | Merge into the existing file (newest row per post wins; an `ERROR` never overwrites a real verdict) |
| `--skip-existing` | off | With `--append`, don't re-score posts that already have a verdict |
| `--html` | – | Also write the HTML report |

Press <kbd>Ctrl</kbd>+<kbd>C</kbd> during a scrape and everything scored so far is still saved.

---

## The output

| Column | Meaning |
|---|---|
| `title`, `url` | The post |
| `score`, `num_comments` | Reddit upvotes and comment count |
| `created_utc`, `created_date` | When it was posted |
| `sentiment_label`, `sentiment_score` | Model verdict and confidence |
| `post_id`, `subreddit`, `permalink` | Where it came from (added after the columns above) |
| `text_chars`, `model`, `scraped_at` | How much text was scored, by which model, and when |

The first eight columns are exactly those of the original script, in the same order, so existing spreadsheets and notebooks keep working.

### What the report measures

| Metric | How it's calculated |
|---|---|
| **Net sentiment** | (positive − negative) ÷ scored threads, from −1 (all negative) to +1 (all positive) |
| **Engagement-weighted** | The same, with each thread weighted by upvotes + comments, so you can see whether the loud threads lean differently |
| **Mean confidence** | Average model confidence; verdicts under 60% are flagged as uncertain |
| **ρ vs upvotes / comments** | Spearman rank correlation between signed sentiment and engagement |
| **Threads over time** | Daily, weekly or monthly buckets depending on the span, stacked by verdict |
| **Words that tip the balance** | Title words that separate negative from positive threads, by weighted log-odds with an informative Dirichlet prior (Monroe et al., 2008), which stays sensible on small samples |

---

## How it works

```
 Reddit API (PRAW) ─┐
                    ├─▶ Post ─▶ title + body + comments ─▶ DistilBERT ─▶ CSV / JSONL ─▶ terminal + HTML report
 Public JSON ───────┘            (cleaned, ~2,000 chars)    or lexicon
```

* **Sources** (`fakespotter/sources/`): `praw` uses the official API, exactly as the original script did. `public` reads `reddit.com/r/<sub>/<sort>.json`, retries on rate limits with exponential backoff, and walks the full comment tree; Reddit refuses anonymous requests from some networks (cloud hosts especially), so API keys are the reliable route.
* **Text** (`fakespotter/text.py`): whitespace is collapsed, bare URLs and `[removed]`/`[deleted]` bodies are dropped, and the text is cut on a word boundary.
* **Sentiment** (`fakespotter/sentiment/`): DistilBERT SST-2, pinned to `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, with **token-level truncation to 512 tokens**. The original script trimmed to ~2,000 characters, which for emoji- and slang-heavy Reddit threads is often more than 512 tokens; the model then raised an error and the thread was saved as `ERROR`. That is the most likely reason 68 of the 100 threads in the bundled sample have no verdict; with token truncation, a fresh scrape scores them.
* **Report** (`fakespotter/report/`): inline SVG charts and a few lines of vanilla JavaScript for tooltips and the thread table. Nothing loads from the network, titles are HTML-escaped, and links are restricted to `http(s)`.

More detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Use it from Python

```python
from fakespotter import Scraper, get_analyzer, get_source
from fakespotter.analysis import summarize
from fakespotter.report import write_html

scraper = Scraper(get_source("auto"), get_analyzer("transformers"), delay=2)
result = scraper.run(["Reps", "RepTime"], sort="top", time_filter="month", limit=50)

frame = result.to_frame()
print(summarize(frame).net_sentiment)
write_html(frame, "report.html")
```

### Reading the results responsibly

SST-2 was trained on film reviews, not replica-buying slang: "GL", "W2C" or "🔥" mean little to it, and a positive haul post with a few complaints in the comments can come out negative. Treat verdicts as a signal across many threads, not a judgment of any one of them, and look at the confidence column. The lexicon backend knows some community terms but is cruder overall.

---

## Development

```bash
pip install -r requirements-dev.txt     # or: pip install -e ".[ml,dev]"

pytest                 # fast suite: no network, no model download
pytest -m model        # also loads the real DistilBERT model
ruff check . && ruff format --check .
```

The fast suite fakes Reddit and the model, so it runs in seconds anywhere. CI runs it on Python 3.10 to 3.13, plus a separate job that downloads DistilBERT and checks real verdicts.

---

## What's in the repo

| Path | Purpose |
|---|---|
| `fakespotter/cli.py` | The `fakespotter` command |
| `fakespotter/pipeline.py` | The fetch → build text → score loop |
| `fakespotter/sources/` | Reddit API (PRAW) and public JSON sources |
| `fakespotter/sentiment/` | DistilBERT and lexicon backends |
| `fakespotter/analysis.py` | Summary metrics, timelines, distinctive terms |
| `fakespotter/report/` | Terminal summary and HTML report |
| `fakespotter/export.py` | CSV / JSONL reading, writing and merging |
| `fakespotter/fakespotter.py` | The original entry point, kept for compatibility |
| `fakespotter/data/fakewatches_deep_sentiment.csv` | Sample output: 100 r/Reps posts with sentiment (from the original script) |
| `fakespotter/data/fake_watches_reddit_posts.csv` | Earlier sample: 30 posts, titles and metadata only |
| `tests/` | The test suite |

Reddit is the only source implemented so far; the `Source` interface in `fakespotter/sources/base.py` is two methods, so adding another forum is a small job.

---

<div align="center">

<picture><source media="(prefers-color-scheme: light)" srcset="assets/footer-light.svg" /><img src="assets/footer.svg" width="100%" alt="Off the menu: from the archive" /></picture>

Reads public discussion only

<sub>Part of <a href="https://github.com/aryankhanna1409">Aryan Khanna's portfolio</a>: one loop, five pours, from robots to AI systems.</sub>

</div>
