"""Tests for CorpusScorer."""

from src.data import Paper, ResearcherCorpus
from src.llm.client import LLMClient
from src.llm.types import RelevanceResponse
from src.scoring import CorpusScorer


class FakeLLMClient(LLMClient):
    """Fake client that returns deterministic weights without calling the API."""

    def __init__(self, weights: dict[tuple[str, str], float]) -> None:
        # Skip parent init (no openai needed)
        self.weights = weights
        self.calls: list[tuple[str, str]] = []
        from src.llm.cache import LLMCache
        from src.llm.tracker import CostTracker
        self.cache = LLMCache(":memory:")
        self.tracker = CostTracker()
        self.provider = "test"
        self.model = "fake-model"

    async def score_relevance(
        self,
        citing_abstract: str,
        cited_abstract: str,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> RelevanceResponse:
        key = (citing_abstract, cited_abstract)
        self.calls.append(key)
        weight = self.weights.get(key, 0.0)
        return RelevanceResponse(weight=weight, reasoning="fake")


def _make_corpus() -> ResearcherCorpus:
    corpus = ResearcherCorpus(researcher_id="A1")
    # Author works (cited papers)
    p1 = Paper("W1", "Paper 1", "Abstract about neural networks.", 2020, 5, is_author_work=True)
    p2 = Paper("W2", "Paper 2", "Abstract about transformers.", 2021, 3, is_author_work=True)
    # Citing works
    c1 = Paper("C1", "Citing 1", "Citing abstract about neural networks.", 2022, 0, is_author_work=False)
    c2 = Paper("C2", "Citing 2", "Citing abstract about transformers.", 2022, 0, is_author_work=False)
    corpus.add_paper(p1)
    corpus.add_paper(p2)
    corpus.add_paper(c1)
    corpus.add_paper(c2)
    corpus.add_edge("C1", "W1")  # high relevance
    corpus.add_edge("C2", "W2")  # high relevance
    corpus.add_edge("C1", "W2")  # low relevance
    return corpus


def test_scorer_basic():
    corpus = _make_corpus()
    weights = {
        ("Citing abstract about neural networks.", "Abstract about neural networks."): 0.9,
        ("Citing abstract about transformers.", "Abstract about transformers."): 0.8,
        ("Citing abstract about neural networks.", "Abstract about transformers."): 0.1,
    }
    client = FakeLLMClient(weights)
    scorer = CorpusScorer(client=client, concurrency=2)
    result = scorer.score_corpus_sync(corpus)

    assert len(result.edges) == 3
    # W1: 0.9 (from C1) = 0.9
    # W2: 0.8 (from C2) + 0.1 (from C1) = 0.9
    assert result.weighted_counts["W1"] == 0.9
    assert result.weighted_counts["W2"] == 0.9


def test_scorer_threshold():
    corpus = _make_corpus()
    weights = {
        ("Citing abstract about neural networks.", "Abstract about neural networks."): 0.9,
        ("Citing abstract about transformers.", "Abstract about transformers."): 0.8,
        ("Citing abstract about neural networks.", "Abstract about transformers."): 0.1,
    }
    client = FakeLLMClient(weights)
    scorer = CorpusScorer(client=client, concurrency=2)
    result = scorer.score_corpus_sync(corpus)

    filtered = scorer.apply_threshold(result, threshold=0.5)
    assert len(filtered.edges) == 2  # only 0.9 and 0.8 edges remain
    assert filtered.weighted_counts["W1"] == 0.9
    assert filtered.weighted_counts["W2"] == 0.8


def test_scorer_skips_empty_abstract():
    corpus = ResearcherCorpus(researcher_id="A1")
    p1 = Paper("W1", "Paper 1", "", 2020, 5, is_author_work=True)  # empty abstract
    c1 = Paper("C1", "Citing 1", "Citing abstract.", 2022, 0, is_author_work=False)
    corpus.add_paper(p1)
    corpus.add_paper(c1)
    corpus.add_edge("C1", "W1")

    weights = {}
    client = FakeLLMClient(weights)
    scorer = CorpusScorer(client=client, concurrency=2)
    result = scorer.score_corpus_sync(corpus)

    assert len(result.edges) == 0
    assert result.skipped_empty_abstract == 1


def test_scorer_dedups_by_cache_key():
    """Same abstract pair scored twice should use cache (second call returns cached)."""
    corpus = _make_corpus()
    weights = {
        ("Citing abstract about neural networks.", "Abstract about neural networks."): 0.9,
    }
    client = FakeLLMClient(weights)
    scorer = CorpusScorer(client=client, concurrency=1)
    result = scorer.score_corpus_sync(corpus)

    # Only one edge matches the weight key; the other two get default 0.0
    # but they still call the fake client. Cache is keyed by abstract pair.
    assert len(result.edges) == 3
    # First edge (C1->W1) has weight 0.9
    w1_edge = next(e for e in result.edges if e.cited_paper_id == "W1")
    assert w1_edge.weight == 0.9
