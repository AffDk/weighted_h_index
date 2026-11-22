"""Shared types for the LLM module."""

from __future__ import annotations

from pydantic import BaseModel


class RelevanceResponse(BaseModel):
    weight: float
    reasoning: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
