"""Live regression check against a committed baseline.

Runs a subset of golden questions on one backend/retrieval mode and fails if
any question whose answer passed in that combination's baseline now fails (or
errors). Regressed or errored questions are rerun once before failing, since
a single LLM run can flip on a borderline question.

    python -m src.eval.regression --backend weaviate \\
        --retrieval-mode hybrid_reranked --questions q01,q02

Baselines live at `eval_data/baselines/ci/<backend>_<retrieval_mode>.json`:
the results file of a full golden-set run for that combination."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.eval.runner import EvalConfig, run_eval

BASELINE_DIR = Path("eval_data/baselines/ci")
DEFAULT_OUTPUT_DIR = Path("eval_results/regression")

# Exit codes: 1 = regression/error found, 2 = setup problem (no API calls made).
EXIT_REGRESSION = 1
EXIT_SETUP = 2

RunEval = Callable[[EvalConfig], dict[str, Any]]


class BaselineError(Exception):
    """The baseline for the requested combination is missing or unusable."""


def baseline_path(backend: str, retrieval_mode: str, root: Path = BASELINE_DIR) -> Path:
    return root / f"{backend}_{retrieval_mode}.json"


def load_baseline(
    backend: str,
    retrieval_mode: str,
    question_ids: tuple[str, ...],
    root: Path = BASELINE_DIR,
) -> dict[str, dict[str, Any]]:
    """Baseline records by question id. Raises BaselineError if the file is
    missing, was recorded for another combination, or lacks a requested
    question, so a misconfigured run fails before spending anything."""
    path = baseline_path(backend, retrieval_mode, root)
    if not path.exists():
        available = sorted(p.stem for p in root.glob("*.json")) if root.exists() else []
        raise BaselineError(
            f"No baseline for backend={backend!r}, retrieval_mode={retrieval_mode!r}: "
            f"expected {path.as_posix()}. Run the full golden set for this "
            "combination (python -m src.eval.cli --backend ... --retrieval-mode "
            "...) and commit its results file there. Available baselines: "
            f"{', '.join(available) or 'none'}."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data["run"].get("config", {})
    if (config.get("backend"), config.get("retrieval_mode")) != (
        backend,
        retrieval_mode,
    ):
        raise BaselineError(
            f"{path.as_posix()} was recorded with backend="
            f"{config.get('backend')!r}, retrieval_mode="
            f"{config.get('retrieval_mode')!r}, not {backend!r}/{retrieval_mode!r}."
        )
    records = {q["id"]: q for q in data["questions"]}
    missing = [qid for qid in question_ids if qid not in records]
    if missing:
        raise BaselineError(
            f"{path.as_posix()} has no record for: {', '.join(missing)}."
        )
    return records


def _passed(record: dict[str, Any]) -> bool:
    return bool(record["answer_score"]["passed"])


@dataclass
class Comparison:
    regressions: list[str] = field(default_factory=list)  # baseline pass -> fail
    errors: list[str] = field(default_factory=list)  # agent/judge raised
    not_gated: list[str] = field(default_factory=list)  # failed in baseline too

    @property
    def failed(self) -> list[str]:
        return self.errors + [q for q in self.regressions if q not in self.errors]


def compare(results: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> Comparison:
    comparison = Comparison()
    for record in results["questions"]:
        qid = record["id"]
        if record["error"] is not None:
            comparison.errors.append(qid)
        elif not _passed(baseline[qid]):
            comparison.not_gated.append(qid)
        elif not _passed(record):
            comparison.regressions.append(qid)
    return comparison


def _merge(first: dict[str, Any], recheck: dict[str, Any]) -> dict[str, Any]:
    """`first` with each rechecked question's record replaced by its rerun."""
    rerun = {q["id"]: q for q in recheck["questions"]}
    return {
        **first,
        "questions": [rerun.get(q["id"], q) for q in first["questions"]],
    }


@dataclass
class CheckOutcome:
    first: dict[str, Any]
    recheck: dict[str, Any] | None
    comparison: Comparison  # final, after any recheck


def run_check(
    config: EvalConfig,
    baseline: dict[str, dict[str, Any]],
    run: RunEval = run_eval,
) -> CheckOutcome:
    first = run(config)
    comparison = compare(first, baseline)
    if not comparison.failed:
        return CheckOutcome(first, None, comparison)
    recheck = run(replace(config, question_ids=tuple(comparison.failed)))
    return CheckOutcome(first, recheck, compare(_merge(first, recheck), baseline))


def _verdict(record: dict[str, Any] | None) -> str:
    if record is None:
        return ""
    if record["error"] is not None:
        return "ERROR"
    return "pass" if _passed(record) else "FAIL"


def summary_markdown(
    outcome: CheckOutcome, baseline: dict[str, dict[str, Any]], title: str
) -> str:
    rerun = {q["id"]: q for q in (outcome.recheck or {}).get("questions", [])}
    comparison = outcome.comparison
    lines = [
        f"### {title}",
        "",
        "| Question | Baseline | Run | Recheck | Result |",
        "|---|---|---|---|---|",
    ]
    for record in outcome.first["questions"]:
        qid = record["id"]
        if qid in comparison.errors:
            result = "error"
        elif qid in comparison.regressions:
            result = "**regression**"
        elif qid in comparison.not_gated:
            result = "not gated (fails in baseline)"
        else:
            result = "ok"
        lines.append(
            f"| {qid} | {_verdict(baseline[qid])} | {_verdict(record)} | "
            f"{_verdict(rerun.get(qid))} | {result} |"
        )
    cost = outcome.first["summary"]["cost_usd"]["total"]
    if outcome.recheck is not None:
        cost += outcome.recheck["summary"]["cost_usd"]["total"]
    lines += ["", f"Cost: ${cost:.4f}"]
    for qid in comparison.failed:
        final = rerun.get(qid) or next(
            q for q in outcome.first["questions"] if q["id"] == qid
        )
        reason = final["error"] or final["answer_score"]["reason"]
        lines.append(f"- **{qid}**: {' '.join(str(reason).split())[:500]}")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["pinecone", "weaviate"], required=True)
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "hybrid", "hybrid_reranked"],
        required=True,
    )
    parser.add_argument(
        "--questions", required=True, help="Comma-separated golden question ids."
    )
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None, run: RunEval = run_eval) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    for noisy in ("httpx", "httpcore", "openai", "pinecone", "cohere"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = _parse_args(argv)
    question_ids = tuple(q.strip() for q in args.questions.split(",") if q.strip())
    if not question_ids:
        print("error: --questions is empty", file=sys.stderr)
        return EXIT_SETUP

    try:
        baseline = load_baseline(
            args.backend, args.retrieval_mode, question_ids, args.baseline_dir
        )
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_SETUP

    config = EvalConfig(
        backend=args.backend,
        retrieval_mode=args.retrieval_mode,
        question_ids=question_ids,
    )
    outcome = run_check(config, baseline, run)

    _write(args.out_dir / "results.json", outcome.first)
    if outcome.recheck is not None:
        _write(args.out_dir / "recheck.json", outcome.recheck)
    summary = summary_markdown(
        outcome, baseline, f"Live eval: {args.backend} / {args.retrieval_mode}"
    )
    print(summary)
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(summary)

    if outcome.comparison.failed:
        print(
            f"FAILED: {', '.join(outcome.comparison.failed)} failed after a recheck.",
            file=sys.stderr,
        )
        return EXIT_REGRESSION
    return 0


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
