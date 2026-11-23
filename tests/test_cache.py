"""Tests for LLM response cache."""

import tempfile
from pathlib import Path

from src.llm.cache import LLMCache
from src.llm.types import RelevanceResponse


def test_cache_put_get_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMCache(db_path=Path(d) / "test.sqlite")
        try:
            resp = RelevanceResponse(weight=0.75, reasoning="high relevance", input_tokens=10, output_tokens=5)
            cache.put("citing abstract A", "cited abstract B", "gemini-model", resp, input_tokens=10, output_tokens=5)
            fetched = cache.get("citing abstract A", "cited abstract B", "gemini-model")
            assert fetched is not None
            assert fetched.weight == 0.75
            assert fetched.reasoning == "high relevance"
        finally:
            cache.close()


def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMCache(db_path=Path(d) / "test.sqlite")
        try:
            fetched = cache.get("unknown citing", "unknown cited", "gemini-model")
            assert fetched is None
        finally:
            cache.close()


def test_cache_stats():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMCache(db_path=Path(d) / "test.sqlite")
        try:
            resp = RelevanceResponse(weight=0.5, reasoning="x", input_tokens=100, output_tokens=50)
            cache.put("a", "b", "m", resp, input_tokens=100, output_tokens=50)
            stats = cache.stats()
            assert stats["entries"] == 1
            assert stats["input_tokens"] == 100
            assert stats["output_tokens"] == 50
        finally:
            cache.close()
