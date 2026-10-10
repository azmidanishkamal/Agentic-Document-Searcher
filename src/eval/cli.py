"""CLI entry point:
`python -m src.eval.cli [--backend pinecone] [--retrieval-mode hybrid_reranked]
[--judge-model gpt-5.4] [--questions q01,q02] [--out PATH]`"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

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
        "--out", type=Path, help=f"Default: {DEFAULT_OUTPUT_DIR}/<run_id>.json"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    for noisy in ("httpx", "httpcore", "openai", "pinecone", "cohere"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()
    args = _parse_args()

    config = EvalConfig(
        backend=args.backend,
        retrieval_mode=args.retrieval_mode,
        judge_model=args.judge_model,
        golden_path=args.golden,
        question_ids=tuple(q.strip() for q in args.questions.split(",") if q.strip()),
    )
    results = run_eval(config)

    out = args.out or config.output_dir / f"{results['run']['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
