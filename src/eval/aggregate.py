"""Combine several eval runs of the same config into one aggregate: pass-rate
spread across runs, per-question pass counts, and which questions flip."""

from __future__ import annotations

import statistics
from typing import Any

TOKEN_FIELDS = ("agent_prompt", "agent_completion", "judge_prompt", "judge_completion")


def _spread(values: list[float | None]) -> dict[str, float | None]:
    present = [v for v in values if v is not None]
    if not present:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(min(present), 4),
        "mean": round(statistics.mean(present), 4),
        "max": round(max(present), 4),
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def _check_same_config(runs: list[dict[str, Any]]) -> dict[str, Any]:
    config = runs[0]["run"]["config"]
    for run in runs[1:]:
        if run["run"]["config"] != config:
            raise ValueError(
                f"Run {run['run']['run_id']} has a different config than "
                f"{runs[0]['run']['run_id']}; refusing to aggregate."
            )
    return config


def _per_run(run: dict[str, Any]) -> dict[str, Any]:
    summary = run["summary"]
    tokens = {name: summary["tokens"][name] for name in TOKEN_FIELDS}
    tokens["total"] = sum(tokens.values())
    return {
        "run_id": run["run"]["run_id"],
        "n": summary["n"],
        "n_errors": summary["n_errors"],
        "answer_passed": summary["answer_passed"],
        "answer_pass_rate": summary["answer_pass_rate"],
        "retrieval_passed": summary["retrieval_passed"],
        "retrieval_n": summary["retrieval_n"],
        "retrieval_pass_rate": summary["retrieval_pass_rate"],
        "latency_total_s": summary["latency_s"]["total"],
        "latency_mean_s": summary["latency_s"]["mean"],
        "tokens": tokens,
    }


def _per_question(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pass counts per question id, in golden-set order. A question missing
    from a run (e.g. filtered out) just has fewer runs counted."""
    stats: dict[str, dict[str, Any]] = {}
    for run in runs:
        for record in run["questions"]:
            entry = stats.setdefault(
                record["id"],
                {
                    "runs": 0,
                    "answer_passed": 0,
                    "retrieval_runs": 0,
                    "retrieval_passed": 0,
                    "errors": 0,
                },
            )
            entry["runs"] += 1
            entry["answer_passed"] += bool(record["answer_score"]["passed"])
            if record["retrieval_score"] is not None:
                entry["retrieval_runs"] += 1
                entry["retrieval_passed"] += bool(record["retrieval_score"]["passed"])
            entry["errors"] += record["error"] is not None
    for entry in stats.values():
        entry["answer"] = f"{entry['answer_passed']}/{entry['runs']}"
        entry["retrieval"] = (
            f"{entry['retrieval_passed']}/{entry['retrieval_runs']}"
            if entry["retrieval_runs"]
            else None
        )
    return stats


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate results dicts (as returned by `run_eval`) from repeated runs
    of one config."""
    if not runs:
        raise ValueError("Need at least one run to aggregate.")
    config = _check_same_config(runs)
    per_run = [_per_run(run) for run in runs]
    per_question = _per_question(runs)

    unstable = [
        qid
        for qid, entry in per_question.items()
        if 0 < entry["answer_passed"] < entry["runs"]
    ]
    unstable_retrieval = [
        qid
        for qid, entry in per_question.items()
        if 0 < entry["retrieval_passed"] < entry["retrieval_runs"]
    ]
    always_failed = [qid for qid, e in per_question.items() if e["answer_passed"] == 0]

    return {
        "n_runs": len(runs),
        "config": config,
        "run_ids": [r["run_id"] for r in per_run],
        "answer_pass_rate": _spread([r["answer_pass_rate"] for r in per_run]),
        "retrieval_pass_rate": _spread([r["retrieval_pass_rate"] for r in per_run]),
        "unstable_questions": unstable,
        "unstable_retrieval_questions": unstable_retrieval,
        "always_failed_questions": always_failed,
        "latency_total_s": {
            "per_run": [r["latency_total_s"] for r in per_run],
            "mean": _mean([r["latency_total_s"] for r in per_run]),
        },
        "tokens": {
            name: {
                "per_run": [r["tokens"][name] for r in per_run],
                "mean": _mean([r["tokens"][name] for r in per_run]),
            }
            for name in (*TOKEN_FIELDS, "total")
        },
        "per_run": per_run,
        "per_question": per_question,
    }
