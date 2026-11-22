"""Base OpenAI-compatible LLM client for relevance scoring."""

from __future__ import annotations

import os

from pydantic import BaseModel

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - openai optional at import time
    AsyncOpenAI = None  # type: ignore[assignment]


class RelevanceResponse(BaseModel):
    weight: float
    reasoning: str | None = None


class LLMClient:
    """Minimal async client wrapper around OpenAI-compatible endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen3.7-flash",
        timeout: float = 60.0,
    ) -> None:
        if AsyncOpenAI is None:  # pragma: no cover
            raise ImportError("openai package is required for LLMClient")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL", "qwen3.7-flash")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)

    async def score_relevance(
        self,
        citing_abstract: str,
        cited_abstract: str,
        temperature: float = 0.0,
    ) -> RelevanceResponse:
        """Score how much the citing article relies on the cited article (0-1)."""
        prompt = (
            "You are evaluating citation relevance between two scientific articles.\n"
            "Given the abstract of a citing article and the abstract of a cited article, "
            "estimate how heavily the citing article relies on the cited article.\n"
            "Return a weight in [0,1]: 1.0 = heavily reliant, 0.0 = marginally relevant.\n"
            "Output ONLY a JSON object: {\"weight\": <float>, \"reasoning\": <short string>}.\n\n"
            f"CITING ABSTRACT: {citing_abstract}\n\n"
            f"CITED ABSTRACT: {cited_abstract}\n"
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        content = resp.choices[0].message.content or "{}"
        import json

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"weight": 0.0, "reasoning": "parse_failed"}
        return RelevanceResponse(weight=float(data.get("weight", 0.0)), reasoning=str(data.get("reasoning", "")))
