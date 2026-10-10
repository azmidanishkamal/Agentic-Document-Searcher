"""Cost estimation from stored token counts, plus re-costing a saved results file.

Unknown prices are `None`, never guessed; any unpriced usage makes the cost
`None` and the run's `cost_usd.complete` false. Because token counts are stored
per question, a results file can be re-costed after prices are filled in:

    python -m src.eval.cost eval_results/<run>.json [--out recosted.json]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenPrice:
    input_per_1m: float
    output_per_1m: float


# USD. Fill in from official pricing pages; None = unknown.
LLM_PRICES: dict[str, TokenPrice | None] = {
    "gpt-5.4": None,
}
# USD per Cohere rerank search (one query over <=100 docs).
RERANK_PRICE_PER_SEARCH: float | None = None


def llm_cost(
    usage: Iterable[Mapping[str, Any]],
    prices: Mapping[str, TokenPrice | None] = LLM_PRICES,
) -> float | None:
    """Cost of a list of `{model, prompt_tokens, completion_tokens}` records."""
    total = 0.0
    for record in usage:
        price = prices.get(record["model"])
        if price is None:
            return None
        total += (
            record["prompt_tokens"] * price.input_per_1m
            + record["completion_tokens"] * price.output_per_1m
        ) / 1_000_000
    return total


def rerank_cost(
    searches: int, price: float | None = RERANK_PRICE_PER_SEARCH
) -> float | None:
    if searches == 0:
        return 0.0
    return None if price is None else searches * price


def _sum_or_none(values: Iterable[float | None]) -> float | None:
    total = 0.0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def question_cost(
    usage: Mapping[str, Any],
    prices: Mapping[str, TokenPrice | None] = LLM_PRICES,
    rerank_price: float | None = RERANK_PRICE_PER_SEARCH,
) -> dict[str, float | None]:
    agent = llm_cost(usage["agent"], prices)
    rerank = rerank_cost(usage["rerank_searches"], rerank_price)
    judge = llm_cost(usage["judge"], prices)
    return {
        "agent": agent,
        "rerank": rerank,
        "judge": judge,
        "total": _sum_or_none([agent, rerank, judge]),
    }


def summarize_costs(
    per_question: Iterable[Mapping[str, float | None]],
) -> dict[str, Any]:
    rows = list(per_question)
    summary: dict[str, Any] = {
        key: _sum_or_none(row[key] for row in rows)
        for key in ("agent", "rerank", "judge", "total")
    }
    summary["complete"] = summary["total"] is not None
    return summary


def recost_results(
    results: dict[str, Any],
    prices: Mapping[str, TokenPrice | None] = LLM_PRICES,
    rerank_price: float | None = RERANK_PRICE_PER_SEARCH,
) -> dict[str, Any]:
    """Recompute every cost field in a results dict from its stored usage."""
    for question in results["questions"]:
        question["cost_usd"] = question_cost(question["usage"], prices, rerank_price)
    results["summary"]["cost_usd"] = summarize_costs(
        q["cost_usd"] for q in results["questions"]
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-cost a saved eval results file.")
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--out", type=Path, help="Write here instead of overwriting in place."
    )
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    recost_results(results)
    out = args.out or args.results
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"]["cost_usd"], indent=2))


if __name__ == "__main__":
    main()
