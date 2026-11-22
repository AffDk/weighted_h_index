"""Batch download researcher corpora from OpenAlex for the weighted h-index paper.

Reads OpenAlex author IDs from a text file (one per line), fetches each author's
oeuvre + all citing works, and saves a JSONL corpus per author.

Usage:
    uv run python scripts/batch_fetch.py authors.txt --out-dir data/corpora --max-citing 50

Env:
    OPENALEX_API_KEY  (required — set in .env)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.data import OpenAlexClient, build_researcher_corpus, save_corpus


def read_author_ids(path: str | Path) -> list[str]:
    """Read author IDs from a file, one per line, skipping blanks and comments."""
    ids: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip URL prefix if present
            if line.startswith("https://openalex.org/authors/"):
                line = line.rsplit("/", 1)[-1]
            ids.append(line)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch fetch OpenAlex researcher corpora")
    parser.add_argument("author_file", help="Text file with one OpenAlex author ID per line")
    parser.add_argument(
        "--out-dir",
        default="data/corpora",
        help="Directory to save per-author JSONL corpora",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=200,
        help="OpenAlex results per page",
    )
    parser.add_argument(
        "--max-citing",
        type=int,
        default=50,
        help="Max citing works per author work (0 = unlimited)",
    )
    parser.add_argument(
        "--mailto",
        default="",
        help="Email for OpenAlex polite pool",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay (s) between authors to be polite to OpenAlex",
    )
    args = parser.parse_args()

    author_ids = read_author_ids(args.author_file)
    if not author_ids:
        print("No author IDs found in file.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mailto_value = args.mailto or None
    client = OpenAlexClient(mailto=mailto_value)

    successes = 0
    failures = 0
    total_edges = 0
    total_papers = 0

    try:
        for idx, author_id in enumerate(author_ids, start=1):
            out_path = out_dir / f"{author_id}.jsonl"
            if out_path.exists():
                print(f"[{idx}/{len(author_ids)}] {author_id}: corpus exists, skipping")
                successes += 1
                continue

            print(f"[{idx}/{len(author_ids)}] {author_id}: fetching...", end=" ")
            try:
                corpus = build_researcher_corpus(
                    client,
                    author_id,
                    max_citing_per_work=args.max_citing if args.max_citing > 0 else None,
                )
                save_corpus(corpus, out_path)
                cited = len(corpus.cited_papers())
                citing = len(corpus.citing_papers())
                edges = len(corpus.edges)
                total_edges += edges
                total_papers += cited
                successes += 1
                print(f"OK ({cited} works, {citing} citing, {edges} edges)")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAILED: {exc}")

            if idx < len(author_ids) and args.delay > 0:
                time.sleep(args.delay)
    finally:
        client.close()

    print()
    print("=" * 50)
    print(f"Done: {successes} succeeded, {failures} failed out of {len(author_ids)}")
    print(f"Total author papers fetched: {total_papers}")
    print(f"Total citation edges: {total_edges}")
    print(f"Corpora saved to: {out_dir}")


if __name__ == "__main__":
    main()
