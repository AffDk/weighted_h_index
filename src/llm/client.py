"""Gemini LLM client for relevance scoring."""

from __future__ import annotations

import json
import os

import httpx

from .cache import LLMCache
from .tracker import CostTracker
from .types import RelevanceResponse


class LLMClient:
    """Minimal async client for the Gemini generateContent API.

    Supports response caching (to avoid redundant API calls) and
    cost/token tracking (for budgeting and reporting).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gemini-2.5-flash",
        timeout: float = 60.0,
        cache: LLMCache | None = None,
        tracker: CostTracker | None = None,
        provider: str = "gemini",
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.provider = provider
        self.timeout = timeout
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

        Checks cache first; on a miss, calls the Gemini API, caches the result,
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
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
        url = self.base_url.rstrip("/") + f"/v1beta/models/{self.model}:generateContent"

        async with httpx.AsyncClient(
            headers={"x-goog-api-key": self.api_key}, timeout=self.timeout
        ) as http_client:
            resp = await http_client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        content = "".join(part.get("text", "") for part in parts).strip()
        if content.startswith("```json"):
            content = content[len("```json") :].strip()
        if content.startswith("```"):
            content = content[len("```") :].strip()
        if content.endswith("```"):
            content = content[: -len("```")].strip()

        usage = data.get("usageMetadata", {})
        input_tokens = int(usage.get("promptTokenCount", 0) or 0)
        output_tokens = int(usage.get("candidatesTokenCount", 0) or 0)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"weight": 0.0, "reasoning": "parse_failed"}

        weight = max(0.0, min(1.0, float(parsed.get("weight", 0.0))))
        reasoning = str(parsed.get("reasoning", ""))
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
