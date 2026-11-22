"""CLI for weighted-h-index."""

from __future__ import annotations

from pathlib import Path

import typer

from .data import OpenAlexClient, build_researcher_corpus, load_corpus, save_corpus
from .index import h_index, weighted_h_index

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
) -> None:
    """Compute an h-index variant for an author from a saved corpus."""
    corpus = load_corpus(author_id, corpus_path)
    cited = corpus.cited_papers()

    if method == "conventional":
        counts = [p.cited_by_count for p in cited]
        score = h_index(counts)
    elif method == "hip":
        # hip-index needs mention counts; with abstracts-only we fall back to
        # cited_by_count as a proxy weight placeholder until full text is available.
        counts = [p.cited_by_count for p in cited]
        score = h_index(counts)
        typer.warn(
            "hip-index requires full-text mention counts; falling back to "
            "conventional h-index until LLM weights are applied."
        )
    elif method == "weighted":
        # Weighted counts placeholder: each edge weight defaults to 1.0 until
        # Phase 2 LLM scoring is wired in.
        weighted_counts: dict[str, float] = {p.openalex_id: 0.0 for p in cited}
        for edge in corpus.edges:
            if edge.cited_paper_id in weighted_counts:
                weighted_counts[edge.cited_paper_id] += 1.0
        score = weighted_h_index(weighted_counts.values())
    else:
        typer.echo(f"Unknown method: {method}")
        raise typer.Exit(code=1)

    typer.echo(f"{method} h-index for {author_id}: {score} ({len(cited)} author works)")


if __name__ == "__main__":
    app()
