"""CLI for weighted-h-index."""

from __future__ import annotations

from pathlib import Path

import typer

from .data import OpenAlexClient, build_researcher_corpus, load_corpus, save_corpus
from .index import h_index, weighted_h_index
from .llm.cache import LLMCache
from .llm.providers import get_client
from .llm.tracker import CostTracker
from .scoring import CorpusScorer

app = typer.Typer(help="Weighted H-Index CLI")

DATA_DIR = Path("data")


@app.command()
def fetch(
    author_id: str = typer.Argument(..., help="OpenAlex author ID (e.g. A510...)"),
    per_page: int = typer.Option(200, help="Results per page"),
    mailto: str = typer.Option("", help="Email for OpenAlex polite pool"),
    out: str = typer.Option("data/corpus.jsonl", help="Output JSONL path"),
    max_citing: int = typer.Option(0, help="Max citing works per author work (0 = unlimited)"),
) -> None:
    """Fetch a researcher's subgraph from OpenAlex and save to JSONL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mailto_value = mailto or None
    client = OpenAlexClient(mailto=mailto_value)
    try:
        corpus = build_researcher_corpus(
            client,
            author_id,
            max_citing_per_work=max_citing if max_citing > 0 else None,
        )
        save_corpus(corpus, out)
        cited = len(corpus.cited_papers())
        citing = len(corpus.citing_papers())
        typer.echo(
            f"Saved corpus to {out}: {cited} author works, {citing} citing works, "
            f"{len(corpus.edges)} edges"
        )
    finally:
        client.close()


@app.command()
def compute(
    author_id: str = typer.Argument(..., help="OpenAlex author ID"),
    corpus_path: str = typer.Option("data/corpus.jsonl", help="Corpus JSONL path"),
    method: str = typer.Option("weighted", help="weighted | conventional | hip"),
    use_llm: bool = typer.Option(True, help="Use LLM relevance scoring for weighted method"),
    provider: str = typer.Option("qwen", help="LLM provider: qwen | deepseek | kimi | glm"),
    concurrency: int = typer.Option(5, help="Concurrent LLM requests"),
    rate_delay: float = typer.Option(0.0, help="Delay (s) between LLM calls"),
    threshold: float = typer.Option(0.0, help="Only count edges with weight >= threshold"),
    cache_db: str = typer.Option("data/llm_cache.sqlite", help="LLM response cache path"),
    cost_report: str = typer.Option("data/cost_report.json", help="Cost report output path"),
) -> None:
    """Compute an h-index variant for an author from a saved corpus."""
    corpus = load_corpus(author_id, corpus_path)
    cited = corpus.cited_papers()

    if method == "conventional":
        counts = [p.cited_by_count for p in cited]
        score = h_index(counts)
        typer.echo(f"{method} h-index for {author_id}: {score} ({len(cited)} author works)")
        return

    if method == "hip":
        counts = [p.cited_by_count for p in cited]
        score = h_index(counts)
        typer.warn(
            "hip-index requires full-text mention counts; falling back to "
            "conventional h-index until LLM weights are applied."
        )
        typer.echo(f"{method} h-index for {author_id}: {score} ({len(cited)} author works)")
        return

    if method == "weighted":
        if not use_llm or not corpus.edges:
            # Unit-weight fallback
            weighted_counts: dict[str, float] = {p.openalex_id: 0.0 for p in cited}
            for edge in corpus.edges:
                if edge.cited_paper_id in weighted_counts:
                    weighted_counts[edge.cited_paper_id] += 1.0
            if threshold > 0:
                weighted_counts = {k: v for k, v in weighted_counts.items() if v >= threshold}
            score = weighted_h_index(weighted_counts.values())
            typer.echo(
                f"{method} h-index for {author_id}: {score} "
                f"({len(cited)} author works, unit weights, no LLM)"
            )
            return

        # LLM-based scoring
        llm_client = get_client(provider)
        cache = LLMCache(db_path=cache_db)
        tracker = CostTracker()
        llm_client.cache = cache
        llm_client.tracker = tracker

        try:
            scorer = CorpusScorer(
                client=llm_client,
                concurrency=concurrency,
                rate_delay=rate_delay,
                use_cache=True,
            )
            result = scorer.score_corpus_sync(corpus)

            if threshold > 0:
                result = scorer.apply_threshold(result, threshold)

            score = weighted_h_index(result.weighted_counts.values())
            stats = cache.stats()
            cost = tracker.totals()

            typer.echo(
                f"{method} h-index for {author_id}: {score} "
                f"({len(cited)} author works, {len(result.edges)} scored edges)"
            )
            typer.echo(
                f"  LLM calls: {result.total_calls}, cache entries: {stats['entries']}, "
                f"skipped (empty abstract): {result.skipped_empty_abstract}"
            )
            typer.echo(
                f"  Estimated cost: ${cost['estimated_cost_usd']:.6f} "
                f"({cost['input_tokens']} in / {cost['output_tokens']} out tokens)"
            )
            tracker.save(cost_report)
            typer.echo(f"  Cost report saved to {cost_report}")
        finally:
            llm_client.close()
        return

    typer.echo(f"Unknown method: {method}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
