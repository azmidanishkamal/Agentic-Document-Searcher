"""CLI entry point:
`python -m src.eval.cli [--backend pinecone] [--retrieval-mode hybrid_reranked]
[--judge-model gpt-5.4] [--questions q01,q02] [--runs N] [--out PATH]`

With --runs N > 1 the same config runs N times; each run's results file is
written as usual, plus an aggregate file combining them."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.eval.aggregate import aggregate_runs
from src.eval.runner import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_OUTPUT_DIR,
    EvalConfig,
    run_eval,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the golden-set evaluation.")
    parser.add_argument(
        "--backend", choices=["pinecone", "weaviate"], default="pinecone"
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "hybrid", "hybrid_reranked"],
        default="hybrid_reranked",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--golden", type=Path, default=EvalConfig.golden_path)
    parser.add_argument(
        "--questions", default="", help="Comma-separated ids; default all."
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Repeat the run N times (default 1)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=f"Default: {DEFAULT_OUTPUT_DIR}/<run_id>.json. With --runs > 1, "
        "run k is written to <out stem>_run<k>.json and the aggregate to "
        "<out stem>_aggregate.json.",
    )
    return parser.parse_args()


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    for noisy in ("httpx", "httpcore", "openai", "pinecone", "cohere"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()
    args = _parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    config = EvalConfig(
        backend=args.backend,
        retrieval_mode=args.retrieval_mode,
        judge_model=args.judge_model,
        golden_path=args.golden,
        question_ids=tuple(q.strip() for q in args.questions.split(",") if q.strip()),
    )

    if args.runs == 1:
        results = run_eval(config)
        out = args.out or config.output_dir / f"{results['run']['run_id']}.json"
        _write(out, results)
        print(json.dumps(results["summary"], indent=2))
        print(f"\nWrote {out}")
        return

    runs = []
    for k in range(1, args.runs + 1):
        print(f"=== Run {k}/{args.runs} ===", flush=True)
        results = run_eval(config)
        out = (
            args.out.with_name(f"{args.out.stem}_run{k}.json")
            if args.out
            else config.output_dir / f"{results['run']['run_id']}.json"
        )
        _write(out, results)
        print(f"Wrote run {k}: {out}")
        runs.append(results)

    aggregate = aggregate_runs(runs)
    agg_out = (
        args.out.with_name(f"{args.out.stem}_aggregate.json")
        if args.out
        else config.output_dir
        / f"{runs[0]['run']['run_id']}_x{args.runs}_aggregate.json"
    )
    _write(agg_out, aggregate)
    summary = {k: v for k, v in aggregate.items() if k != "per_question"}
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {agg_out}")


if __name__ == "__main__":
    main()
