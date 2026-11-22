"""Tests for data models and corpus serialization."""

import tempfile
from pathlib import Path

from src.data import Paper, ResearcherCorpus, abstract_from_inverted_index


def test_abstract_reconstruction():
    inverted = {"hello": [0], "world": [1], "foo": [2]}
    assert abstract_from_inverted_index(inverted) == "hello world foo"
    assert abstract_from_inverted_index({}) == ""
    assert abstract_from_inverted_index(None) == ""


def test_corpus_roundtrip():
    corpus = ResearcherCorpus(researcher_id="A1")
    p1 = Paper("W1", "Title 1", "Abstract one.", 2020, 5, is_author_work=True)
    p2 = Paper("W2", "Title 2", "Abstract two.", 2021, 3, is_author_work=False)
    corpus.add_paper(p1)
    corpus.add_paper(p2)
    corpus.add_edge("W2", "W1")

    assert len(corpus.cited_papers()) == 1
    assert len(corpus.citing_papers()) == 1
    assert len(corpus.edges) == 1

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "corpus.jsonl"
        corpus.to_jsonl(path)
        loaded = ResearcherCorpus.from_jsonl("A1", path)
        assert len(loaded.papers) == 2
        assert len(loaded.edges) == 1
        assert loaded.papers["W1"].is_author_work is True
        assert loaded.papers["W2"].is_author_work is False
