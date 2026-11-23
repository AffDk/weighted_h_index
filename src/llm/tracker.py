"""Cost/token tracking for LLM API calls.

Tracks input/output tokens per provider+model and estimates cost using
published per-million-token rates. Rates are best-effort and should be
verified against each provider's pricing page before budgeting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Approximate per-1M-token rates (input, output) in USD.
# Sources: provider pricing pages, retrieved Nov 2026.
PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "gemini": {
        "gemini-2.5-flash": (0.15, 0.60),
    },
}


@dataclass
class CallRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class CostTracker:
    """Accumulates token usage and estimated cost across LLM calls."""

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CallRecord:
        rate = PRICING.get(provider, {}).get(model, (0.0, 0.0))
        cost = (input_tokens / 1_000_000) * rate[0] + (output_tokens / 1_000_000) * rate[1]
        record = CallRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )
        self.calls.append(record)
        return record

    def totals(self) -> dict[str, float | int]:
        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)
        total_cost = sum(c.estimated_cost_usd for c in self.calls)
        return {
            "calls": len(self.calls),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "estimated_cost_usd": round(total_cost, 6),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "totals": self.totals(),
            "calls": [
                {
                    "provider": c.provider,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "estimated_cost_usd": c.estimated_cost_usd,
                }
                for c in self.calls
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
