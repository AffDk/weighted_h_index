"""Cost estimation for running the weighted h-index evaluation.

OpenAlex: free with API key (~$1/day shared budget, essentially free for research scale).
LLM scoring: Gemini via Google AI Studio API.
"""

from __future__ import annotations

import argparse


def estimate_llm_cost(
    n_researchers: int,
    avg_author_works: int,
    avg_citing_per_work: int,
    avg_tokens_per_call: int = 900,
    input_price_per_1m: float = 0.15,
    output_price_per_1m: float = 0.60,
    avg_output_tokens: int = 60,
) -> dict[str, float]:
    total_edges = n_researchers * avg_author_works * avg_citing_per_work
    total_input_tokens = total_edges * avg_tokens_per_call
    total_output_tokens = total_edges * avg_output_tokens

    input_cost = (total_input_tokens / 1_000_000) * input_price_per_1m
    output_cost = (total_output_tokens / 1_000_000) * output_price_per_1m
    total_cost = input_cost + output_cost

    return {
        "total_edges": float(total_edges),
        "total_input_tokens": float(total_input_tokens),
        "total_output_tokens": float(total_output_tokens),
        "input_cost_usd": round(input_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(total_cost, 4),
    }


def print_estimate(label: str, result: dict[str, float]) -> None:
    print(f"\n=== {label} ===")
    print(f"  Total citation edges to score:  {result['total_edges']:,.0f}")
    print(f"  Est. input tokens:              {result['total_input_tokens']:,.0f}")
    print(f"  Est. output tokens:             {result['total_output_tokens']:,.0f}")
    print(f"  Est. input cost:                ${result['input_cost_usd']:.4f}")
    print(f"  Est. output cost:               ${result['output_cost_usd']:.4f}")
    print(f"  EST. TOTAL LLM COST:            ${result['total_cost_usd']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate LLM API cost for weighted h-index eval")
    parser.add_argument("--researchers", type=int, default=100, help="Number of researchers")
    parser.add_argument("--avg-works", type=int, default=50, help="Avg author works per researcher")
    parser.add_argument(
        "--avg-citing",
        type=int,
        default=10,
        help="Avg citing works per author work (edges per paper)",
    )
    parser.add_argument(
        "--avg-tokens",
        type=int,
        default=900,
        help="Avg input tokens per LLM call (abstracts + prompt)",
    )
    args = parser.parse_args()

    print("Weighted h-index evaluation — LLM cost estimate (Gemini 2.5 Flash)")
    print("Pricing: $0.15 in / $0.60 out per 1M tokens (verify against Google AI Studio pricing before budgeting)")
    print("OpenAlex: free with API key (shared $1/day budget, effectively free at this scale)")

    result = estimate_llm_cost(
        n_researchers=args.researchers,
        avg_author_works=args.avg_works,
        avg_citing_per_work=args.avg_citing,
        avg_tokens_per_call=args.avg_tokens,
    )
    print_estimate("Small scale (pilot)", estimate_llm_cost(20, 50, 10, args.avg_tokens))
    print_estimate("Medium scale", estimate_llm_cost(100, 50, 10, args.avg_tokens))
    print_estimate("Large scale", estimate_llm_cost(500, 100, 15, args.avg_tokens))
    print_estimate("Your custom params", result)


if __name__ == "__main__":
    main()
