"""Tests for provider configuration."""

import pytest

from src.llm.providers import get_client


def test_get_client_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="Missing API key env var: GEMINI_API_KEY"):
        get_client()


def test_get_client_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider: bogus"):
        get_client("bogus")
