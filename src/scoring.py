"""CorpusScorer: applies LLM-derived relevance weights to citation edges.

For each edge (citing_paper -> cited_paper), scores relevance from the pair
of abstracts using the LLM client, with caching to avoid redundant API calls.
Computes per-paper weighted citation counts (wcp) usable by the weighted
h-index. Supports concurrency control and an optional weight threshold
for ablation studies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .data import Paper, ResearcherCorpus
from .llm.client import LLMClient


@dataclass
class ScoredEdge:
    citing_paper_id: str
    cited_paper_id: str
    weight: float
    reasoning: str | None
    cached: bool


@dataclass
class ScoringResult:
    edges: list[ScoredEdge] = field(default_factory=list)
    weighted_counts: dict[str, float] = field(default_factory=dict)
    total_calls: int = 0
    cache_hits: int = 0
    skipped_empty_abstract: int = 0


class CorpusScorer:
    """Scores citation edges in a researcher corpus using an LLM."""

    def __init__(
        self,
        client: LLMClient,
        concurrency: int = 5,
        rate_delay: float = 0.0,
        use_cache: bool = True,
    ) -> None:
        self.client = client
        self.concurrency = concurrency
        self.rate_delay = rate_delay
        self.use_cache = use_cache
        self._semaphore = asyncio.Semaphore(concurrency)

    async def _score_one(
        self,
        citing: Paper,
        cited: Paper,
    ) -> ScoredEdge | None:
        if not citing.abstract or not cited.abstract:
            return None

        async with self._semaphore:
            if self.rate_delay > 0:
                await asyncio.sleep(self.rate_delay)
            response = await self.client.score_relevance(
                citing_abstract=citing.abstract,
                cited_abstract=cited.abstract,
                use_cache=self.use_cache,
            )
        return ScoredEdge(
            citing_paper_id=citing.openalex_id,
            cited_paper_id=cited.openalex_id,
            weight=response.weight,
            reasoning=response.reasoning,
            cached=False,  # cache hit/miss tracked at client level
        )

    async def score_corpus(self, corpus: ResearcherCorpus) -> ScoringResult:
        """Score all edges in the corpus. Returns per-edge weights and wcp."""
        cited_by_id = {p.openalex_id: p for p in corpus.cited_papers()}
        citing_by_id = {p.openalex_id: p for p in corpus.citing_papers()}

        tasks = []
        for edge in corpus.edges:
            citing = citing_by_id.get(edge.citing_paper_id)
            cited = cited_by_id.get(edge.cited_paper_id)
            if citing is None or cited is None:
                continue
            tasks.append(self._score_one(citing, cited))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        scored_edges: list[ScoredEdge] = []
        weighted_counts: dict[str, float] = {
            p.openalex_id: 0.0 for p in cited_by_id.values()
        }
        skipped = 0

        for r in results:
            if isinstance(r, Exception):
                skipped += 1
                continue
            if r is None:
                skipped += 1
                continue
            scored_edges.append(r)
            if r.cited_paper_id in weighted_counts:
                weighted_counts[r.cited_paper_id] += r.weight

        return ScoringResult(
            edges=scored_edges,
            weighted_counts=weighted_counts,
            total_calls=len(tasks),
            cache_hits=0,  # placeholder; cache stats available via client.cache.stats()
            skipped_empty_abstract=skipped,
        )

    def score_corpus_sync(self, corpus: ResearcherCorpus) -> ScoringResult:
        """Synchronous wrapper around score_corpus."""
        return asyncio.run(self.score_corpus(corpus))

    def apply_threshold(
        self, result: ScoringResult, threshold: float
    ) -> ScoringResult:
        """Return a copy with only edges having weight >= threshold, and
        recomputed weighted counts."""
        filtered_edges = [e for e in result.edges if e.weight >= threshold]
        weighted_counts: dict[str, float] = {
            p: 0.0 for p in result.weighted_counts
        }
        for e in filtered_edges:
            if e.cited_paper_id in weighted_counts:
                weighted_counts[e.cited_paper_id] += e.weight
        return ScoringResult(
            edges=filtered_edges,
            weighted_counts=weighted_counts,
            total_calls=result.total_calls,
            cache_hits=result.cache_hits,
            skipped_empty_abstract=result.skipped_empty_abstract,
        )
