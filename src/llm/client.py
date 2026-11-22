"""Base OpenAI-compatible LLM client for relevance scoring."""

from __future__ import annotations

import os

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - openai optional at import time
    AsyncOpenAI = None  # type: ignore[assignment]

from .cache import LLMCache
from .tracker import CostTracker
from .types import RelevanceResponse


class LLMClient:
    """Minimal async client wrapper around OpenAI-compatible endpoints.

    Supports response caching (to avoid redundant API calls) and
    cost/token tracking (for budgeting and reporting).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen3.7-flash",
        timeout: float = 60.0,
        cache: LLMCache | None = None,
        tracker: CostTracker | None = None,
        provider: str = "qwen",
    ) -> None:
        if AsyncOpenAI is None:  # pragma: no cover
            raise ImportError("openai package is required for LLMClient")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL", "qwen3.7-flash")
        self.provider = provider
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)
        self.cache = cache or LLMCache()
        self.tracker = tracker or CostTracker()

    async def score_relevance(
        self,
        citing_abstract: str,
        cited_abstract: str,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> RelevanceResponse:
        """Score how much the citing article relies on the cited article (0-1).

        Checks cache first; on a miss, calls the LLM API, caches the result,
        and records token usage/cost.
        """
        if use_cache:
            cached = self.cache.get(citing_abstract, cited_abstract, self.model)
            if cached is not None:
                return cached

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

        input_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(resp.usage, "completion_tokens", 0) or 0

        import json

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"weight": 0.0, "reasoning": "parse_failed"}

        weight = max(0.0, min(1.0, float(data.get("weight", 0.0))))
        reasoning = str(data.get("reasoning", ""))
        response = RelevanceResponse(
            weight=weight,
            reasoning=reasoning,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if use_cache:
            self.cache.put(
                citing_abstract,
                cited_abstract,
                self.model,
                response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        self.tracker.record(self.provider, self.model, input_tokens, output_tokens)
        return response

    def close(self) -> None:
        self.cache.close()
