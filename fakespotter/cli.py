"""Command-line interface: ``fakespotter <command> [options]``.

Commands:
    scrape   Fetch posts, score every thread and save a CSV (the README's workflow).
    report   Summarise a results file in the terminal and/or as an HTML report.
    demo     Build a report from the bundled r/Reps sample; no credentials needed.
    score    Score a piece of text, to see what the model makes of it.
    doctor   Check the installation, credentials and model.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import webbrowser
from importlib import resources
from pathlib import Path

from rich.console import Console
from rich.text import Text

from fakespotter import __version__
from fakespotter.config import RedditCredentials, load_dotenv
from fakespotter.models import ERROR, NEGATIVE, POSITIVE, ScoredPost
from fakespotter.pipeline import STRATEGIES, RunResult, Scraper
from fakespotter.sentiment import BACKENDS, AnalyzerError, get_analyzer
from fakespotter.sources import SORTS, SOURCES, TIME_FILTERS, SourceError, get_source
from fakespotter.text import DEFAULT_MAX_CHARS

DEFAULT_OUTPUT = "fakewatches_deep_sentiment.csv"
SAMPLE_FILE = "fakewatches_deep_sentiment.csv"
STYLE = {POSITIVE: "blue", NEGATIVE: "red", ERROR: "yellow"}

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)


def sample_path(name: str = SAMPLE_FILE) -> Path:
    """Path to a CSV bundled with the package (see ``fakespotter/data``)."""
    return Path(str(resources.files("fakespotter") / "data" / name))


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fakespotter",
        description="Track how Reddit talks about counterfeit goods and score the sentiment of every thread.",
        epilog="Run `fakespotter <command> --help` for a command's options. Try `fakespotter demo` first.",
    )
    parser.add_argument("--version", action="version", version=f"fakespotter {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # scrape ---------------------------------------------------------------------------
    p = sub.add_parser(
        "scrape",
        help="fetch posts, score every thread, save a CSV",
        description="Fetch posts from one or more subreddits, score each thread "
        "(title + body + comments) and save the results.",
    )
    p.add_argument(
        "-s",
        "--subreddit",
        action="append",
        metavar="NAME",
        help="subreddit to read, without r/ (repeat or comma-separate for several; default: Reps)",
    )
    p.add_argument(
        "-n", "--limit", type=int, default=100, help="posts per subreddit (default: 100)"
    )
    p.add_argument("--sort", choices=SORTS, default="hot", help="listing to read (default: hot)")
    p.add_argument(
        "--time",
        dest="time_filter",
        choices=TIME_FILTERS,
        default="week",
        help="time window for --sort top/controversial (default: week)",
    )
    p.add_argument(
        "--source",
        choices=SOURCES,
        default="auto",
        help="praw = official API (needs credentials); public = no keys; auto picks (default)",
    )
    p.add_argument(
        "-b",
        "--backend",
        choices=BACKENDS,
        default="transformers",
        help="sentiment model: transformers = DistilBERT (default), lexicon = fast, no PyTorch",
    )
    p.add_argument(
        "--model", default=None, help="Hugging Face model ID for the transformers backend"
    )
    p.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="truncate",
        help="truncate = score the first --max-chars characters (default); "
        "chunked = score the whole thread in pieces and average",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"characters per scored text (default: {DEFAULT_MAX_CHARS})",
    )
    p.add_argument(
        "--comment-limit", type=int, default=None, help="use at most this many comments per thread"
    )
    p.add_argument(
        "--delay", type=float, default=2.0, help="seconds to wait between posts (default: 2)"
    )
    p.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"results file, .csv or .jsonl (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="merge into an existing output file instead of overwriting it (newest row per post wins)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="with --append, don't re-score posts already scored in the output file",
    )
    p.add_argument("--html", metavar="PATH", help="also write an HTML report here")
    p.add_argument(
        "-q", "--quiet", action="store_true", help="only print errors and the final line"
    )
    p.add_argument("--no-summary", action="store_true", help="skip the summary printed at the end")
    p.set_defaults(func=cmd_scrape)

    # report ---------------------------------------------------------------------------
    p = sub.add_parser(
        "report",
        help="summarise a results file (terminal and/or HTML)",
        description="Summarise a results file written by `fakespotter scrape`.",
    )
    p.add_argument("file", help="results file (.csv or .jsonl)")
    p.add_argument("--html", metavar="PATH", help="write a self-contained HTML report here")
    p.add_argument("--title", help="headline for the HTML report")
    p.add_argument("--open", action="store_true", help="open the HTML report in a browser")
    p.add_argument(
        "--top", type=int, default=5, help="threads per 'most …' table (default: 5, 0 hides them)"
    )
    p.set_defaults(func=cmd_report)

    # demo -----------------------------------------------------------------------------
    p = sub.add_parser("demo", help="report on the bundled r/Reps sample (no credentials needed)")
    p.add_argument(
        "--html",
        metavar="PATH",
        default="fakespotter-demo.html",
        help="where to write the HTML report (default: fakespotter-demo.html)",
    )
    p.add_argument("--open", action="store_true", help="open the HTML report in a browser")
    p.set_defaults(func=cmd_demo)

    # score ----------------------------------------------------------------------------
    p = sub.add_parser(
        "score",
        help="score text given as arguments or on stdin",
        description="Score text with the sentiment model. Each argument is one text (quote sentences); with no TEXT, reads one text per line from stdin.",
    )
    p.add_argument("text", nargs="*", help="text to score")
    p.add_argument("-b", "--backend", choices=BACKENDS, default="transformers")
    p.add_argument(
        "--model", default=None, help="Hugging Face model ID for the transformers backend"
    )
    p.set_defaults(func=cmd_score)

    # doctor ---------------------------------------------------------------------------
    p = sub.add_parser("doctor", help="check dependencies, credentials and the model")
    p.add_argument(
        "--load-model", action="store_true", help="also load (and if needed download) DistilBERT"
    )
    p.set_defaults(func=cmd_doctor)
    return parser


# ----------------------------------------------------------------------------- helpers
def _subreddits(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or ["Reps"]:
        for name in value.split(","):
            name = name.strip().removeprefix("/").removeprefix("r/").strip("/")
            if name and name.lower() not in (n.lower() for n in names):
                names.append(name)
    return names


def _analyzer(args: argparse.Namespace):
    kwargs = {}
    if args.backend == "transformers" and args.model:
        kwargs["model"] = args.model
    return get_analyzer(args.backend, **kwargs)


def _verdict(scored: ScoredPost) -> Text:
    r = scored.sentiment
    if r.is_error:
        return Text(f"ERROR ({scored.error})" if scored.error else "ERROR", style=STYLE[ERROR])
    return Text(f"{r.label} {r.score:.0%}", style=STYLE[r.label])


def _short(title: str, n: int = 60) -> str:
    title = " ".join(title.split())
    return title if len(title) <= n else title[: n - 1] + "…"


def _open(path: Path) -> None:
    webbrowser.open(path.resolve().as_uri())


# ----------------------------------------------------------------------------- commands
def cmd_scrape(args: argparse.Namespace) -> int:
    from fakespotter.export import existing_ids, write_results

    if args.limit <= 0:
        err_console.print("[red]--limit must be positive[/]")
        return 2
    subreddits = _subreddits(args.subreddit)
    try:
        source = get_source(args.source)
        analyzer = _analyzer(args)
        scraper = Scraper(
            source,
            analyzer,
            delay=args.delay,
            max_chars=args.max_chars,
            strategy=args.strategy,
            comment_limit=args.comment_limit,
        )
    except (SourceError, AnalyzerError, ValueError) as exc:
        err_console.print(f"[red]Error:[/] {exc}")
        return 2

    if args.backend == "transformers":
        with console.status("Loading the sentiment model (the first run downloads it)…"):
            try:
                analyzer.load()
            except AnalyzerError as exc:
                err_console.print(f"[red]Error:[/] {exc}")
                return 2

    skip: set[str] = set()
    if args.skip_existing:
        if not args.append:
            err_console.print("[yellow]--skip-existing only applies with --append; ignoring it.[/]")
        else:
            skip = existing_ids(args.output)

    if not args.quiet:
        console.print(
            f"[bold]fakespotter[/] {__version__} · source [cyan]{source.name}[/] · model [cyan]{analyzer.name}[/] · "
            f"{args.sort} · {args.limit} posts from {', '.join('r/' + s for s in subreddits)}"
        )
        if skip:
            console.print(f"[dim]Skipping {len(skip)} posts already scored in {args.output}.[/]")

    collected: list[ScoredPost] = []

    def on_subreddit(name: str, limit: int) -> None:
        if not args.quiet and len(subreddits) > 1:
            console.rule(f"r/{name}")

    def on_progress(scored: ScoredPost, index: int, limit: int) -> None:
        collected.append(scored)
        if not args.quiet:
            console.print(
                Text.assemble(
                    (f"[{index}/{limit}] ", "dim"),
                    f"{_short(scored.post.title)}  ",
                    _verdict(scored),
                )
            )

    started = time.monotonic()
    interrupted = False
    try:
        result = scraper.run(
            subreddits,
            sort=args.sort,
            limit=args.limit,
            time_filter=args.time_filter,
            skip_ids=skip,
            on_subreddit=on_subreddit,
            on_progress=on_progress,
        )
    except KeyboardInterrupt:
        interrupted = True
        result = RunResult(scored=collected, model=analyzer.name)
        err_console.print("\n[yellow]Interrupted, saving what was scored so far…[/]")

    for name, error in result.errors.items():
        err_console.print(f"[red]r/{name}:[/] {error}")
    if not result.scored:
        if args.append and Path(args.output).exists() and not result.errors:
            console.print("Nothing new to score.")
            return 0
        err_console.print("[red]No posts were scored, so nothing was written.[/]")
        return 1

    frame = write_results(result.to_frame(), args.output, append=args.append)
    elapsed = time.monotonic() - started
    ok = len(result.scored) - result.failed
    console.print(
        f"\n[green]✓[/] Scored {ok}/{len(result.scored)} threads in {elapsed:.0f}s → [bold]{args.output}[/]"
        + (f" ({len(frame)} rows in total)" if args.append else "")
    )
    if args.html:
        from fakespotter.report import write_html

        path = write_html(frame, args.html, source_name=Path(args.output).name)
        console.print(f"[green]✓[/] HTML report → [bold]{path}[/]")
    if not args.quiet and not args.no_summary:
        from fakespotter.report import print_summary

        console.print()
        print_summary(frame, console, top=3)
    if interrupted:
        return 130
    return 1 if result.errors and len(result.errors) == len(subreddits) else 0


def cmd_report(args: argparse.Namespace) -> int:
    from fakespotter.analysis import NoSentimentError
    from fakespotter.export import read_results
    from fakespotter.report import print_summary, write_html

    try:
        frame = read_results(args.file)
        print_summary(frame, console, top=args.top, source_name=Path(args.file).name)
        if args.html:
            path = write_html(frame, args.html, title=args.title, source_name=Path(args.file).name)
            console.print(f"[green]✓[/] HTML report → [bold]{path}[/]")
            if args.open:
                _open(path)
    except (FileNotFoundError, NoSentimentError) as exc:
        err_console.print(f"[red]Error:[/] {exc}")
        return 1
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from fakespotter.export import read_results
    from fakespotter.report import print_summary, write_html

    path = sample_path()
    console.print(
        f"[dim]Bundled sample: 100 hot r/Reps threads scored by the original script ({path.name})[/]\n"
    )
    frame = read_results(path)
    print_summary(frame, console, top=3, source_name=path.name)
    out = write_html(frame, args.html, source_name=path.name)
    console.print(f"[green]✓[/] HTML report → [bold]{out}[/]")
    if args.open:
        _open(out)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    texts = list(args.text)
    if not texts:
        if sys.stdin.isatty():
            err_console.print("Give text as arguments, or pipe lines in on stdin.")
            return 2
        texts = [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        return 0
    try:
        analyzer = _analyzer(args)
        results = analyzer.analyze(texts)
    except AnalyzerError as exc:
        err_console.print(f"[red]Error:[/] {exc}")
        return 2
    for text, r in zip(texts, results):
        console.print(
            Text.assemble(
                (f"{r.label:<8} {r.score:6.1%}  ", STYLE.get(r.label, "")), _short(text, 90)
            )
        )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    def row(ok: bool | None, name: str, detail: str) -> None:
        mark = {True: "[green]✓[/]", False: "[red]✗[/]", None: "[yellow]![/]"}[ok]
        console.print(f"{mark} {name:<22} [dim]{detail}[/]")

    console.print(f"[bold]fakespotter {__version__}[/] · Python {sys.version.split()[0]}\n")
    for module, why in (
        ("pandas", "required"),
        ("praw", "required: official Reddit API"),
        ("rich", "required"),
        ("transformers", "for the DistilBERT backend"),
        ("torch", "for the DistilBERT backend"),
    ):
        found = importlib.util.find_spec(module) is not None
        required = why == "required" or why.startswith("required")
        row(
            found if found or required else None, module, why if found else f"not installed ({why})"
        )
    creds = RedditCredentials.from_env()
    row(
        True if creds else None,
        "Reddit credentials",
        "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET set"
        if creds
        else "not set: `scrape` will use the public JSON source (see README)",
    )
    ua = os.environ.get("REDDIT_USER_AGENT")
    row(
        True if ua else None,
        "User agent",
        ua or "using the default; set REDDIT_USER_AGENT to identify yourself",
    )
    status = 0
    if args.load_model:
        try:
            with console.status("Loading DistilBERT…"):
                analyzer = get_analyzer("transformers").load()
                result = analyzer.analyze_one("This works great, thank you!")
            row(True, "DistilBERT", f"loaded; sample verdict {result.label} {result.score:.0%}")
        except Exception as exc:  # noqa: BLE001
            row(False, "DistilBERT", str(exc))
            status = 1
    return status


# ----------------------------------------------------------------------------- entry
def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
