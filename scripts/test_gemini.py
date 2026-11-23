"""Key-safe connectivity test for Gemini via Google AI Studio API."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.llm.cache import LLMCache
from src.llm.client import LLMClient
from src.llm.tracker import CostTracker

load_dotenv()

REPORT = Path("data/gemini_test_report.json")


def redact(text: str, key: str | None) -> str:
    if key:
        text = text.replace(key, "***REDACTED***")
    return text[:400]


async def probe(key: str) -> dict | str:
    tracker = CostTracker()
    client = LLMClient(
        api_key=key,
        provider="gemini",
        cache=LLMCache(db_path="data/gemini_test_cache.sqlite"),
        tracker=tracker,
    )
    try:
        response = await client.score_relevance(
            "A short citing abstract about dependency parsing.",
            "A short cited abstract about dependency parsing.",
            use_cache=False,
        )
        totals = tracker.totals()
        return {
            "model": client.model,
            "ok": True,
            "weight": response.weight,
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "estimated_cost_usd": totals["estimated_cost_usd"],
            "reasoning": response.reasoning,
        }
    except Exception as exc:  # noqa: BLE001
        message = redact(str(exc), key)
        print(f"FAIL gemini | {type(exc).__name__}: {message}")
        return message
    finally:
        client.close()


async def main() -> None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY missing from environment")
        return

    print("Probing Gemini (key never printed)...")
    result = await probe(key)
    if isinstance(result, str):
        print("No working Gemini model/endpoint found")
        return

    print(
        f"OK   gemini | {result['model']} | "
        f"in={result['input_tokens']} out={result['output_tokens']} "
        f"cost=${result['estimated_cost_usd']:.6f} weight={result['weight']:.2f}"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Report saved to {REPORT}")


if __name__ == "__main__":
    asyncio.run(main())
