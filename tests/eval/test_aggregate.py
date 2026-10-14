from typing import Any

import pytest

from src.eval.aggregate import aggregate_runs

_CONFIG = {"backend": "pinecone", "retrieval_mode": "hybrid_reranked"}


def _q(
    qid: str,
    passed: bool,
    retrieval: bool | None = True,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "id": qid,
        "answer_score": {"passed": passed, "reason": "r"},
        "retrieval_score": None if retrieval is None else {"passed": retrieval},
        "error": error,
    }


def _run(
    run_id: str,
    questions: list[dict[str, Any]],
    latency_total: float,
    agent_prompt: int,
    config: dict[str, Any] = _CONFIG,
) -> dict[str, Any]:
    """Minimal results dict; summary fields are computed the way `summarize` does."""
    answer_passed = sum(q["answer_score"]["passed"] for q in questions)
    retrieval = [q["retrieval_score"] for q in questions if q["retrieval_score"]]
    retrieval_passed = sum(s["passed"] for s in retrieval)
    return {
        "run": {"run_id": run_id, "config": config},
        "summary": {
            "n": len(questions),
            "n_errors": sum(q["error"] is not None for q in questions),
            "answer_passed": answer_passed,
            "answer_pass_rate": round(answer_passed / len(questions), 4),
            "retrieval_n": len(retrieval),
            "retrieval_passed": retrieval_passed,
            "retrieval_pass_rate": (
                round(retrieval_passed / len(retrieval), 4) if retrieval else None
            ),
            "latency_s": {"total": latency_total, "mean": latency_total / 4},
            "tokens": {
                "agent_prompt": agent_prompt,
                "agent_completion": 10,
                "judge_prompt": 100,
                "judge_completion": 1,
            },
        },
        "questions": questions,
    }


@pytest.fixture
def three_runs() -> list[dict[str, Any]]:
    # q1 always passes, q2 never, q3 flips (2/3), q4 is negative (no retrieval
    # score) and flips 1/3. q2's retrieval flips too.
    return [
        _run(
            "r1",
            [_q("q1", True), _q("q2", False), _q("q3", True), _q("q4", True, None)],
            latency_total=40.0,
            agent_prompt=1000,
        ),
        _run(
            "r2",
            [
                _q("q1", True),
                _q("q2", False, retrieval=False),
                _q("q3", False),
                _q("q4", False, None, error="agent: boom"),
            ],
            latency_total=50.0,
            agent_prompt=2000,
        ),
        _run(
            "r3",
            [_q("q1", True), _q("q2", False), _q("q3", True), _q("q4", False, None)],
            latency_total=60.0,
            agent_prompt=3000,
        ),
    ]


def test_pass_rate_spread_across_runs(three_runs: list[dict[str, Any]]) -> None:
    agg = aggregate_runs(three_runs)

    assert agg["n_runs"] == 3
    assert [r["answer_pass_rate"] for r in agg["per_run"]] == [0.75, 0.25, 0.5]
    assert agg["answer_pass_rate"] == {"min": 0.25, "mean": 0.5, "max": 0.75}
    # Retrieval is scored on 3 questions per run (q4 is negative): 3/3, 2/3, 3/3.
    assert [r["retrieval_pass_rate"] for r in agg["per_run"]] == [1.0, 0.6667, 1.0]
    assert agg["retrieval_pass_rate"] == {"min": 0.6667, "mean": 0.8889, "max": 1.0}


def test_per_question_counts_and_unstable(three_runs: list[dict[str, Any]]) -> None:
    pq = aggregate_runs(three_runs)["per_question"]

    assert [pq[q]["answer"] for q in ("q1", "q2", "q3", "q4")] == [
        "3/3",
        "0/3",
        "2/3",
        "1/3",
    ]
    assert pq["q2"]["retrieval"] == "2/3"
    assert pq["q4"]["retrieval"] is None  # never retrieval-scored
    assert pq["q4"]["errors"] == 1


def test_unstable_and_always_failed_lists(three_runs: list[dict[str, Any]]) -> None:
    agg = aggregate_runs(three_runs)

    assert agg["unstable_questions"] == ["q3", "q4"]  # golden-set order
    assert agg["unstable_retrieval_questions"] == ["q2"]
    assert agg["always_failed_questions"] == ["q2"]


def test_latency_and_token_totals(three_runs: list[dict[str, Any]]) -> None:
    agg = aggregate_runs(three_runs)

    assert agg["latency_total_s"] == {"per_run": [40.0, 50.0, 60.0], "mean": 50.0}
    assert agg["tokens"]["agent_prompt"] == {
        "per_run": [1000, 2000, 3000],
        "mean": 2000.0,
    }
    assert agg["tokens"]["total"]["per_run"] == [1111, 2111, 3111]
    assert agg["tokens"]["total"]["mean"] == 2111.0


def test_single_run_has_no_unstable_questions(
    three_runs: list[dict[str, Any]],
) -> None:
    agg = aggregate_runs(three_runs[:1])

    assert agg["unstable_questions"] == []
    assert agg["answer_pass_rate"] == {"min": 0.75, "mean": 0.75, "max": 0.75}
    assert agg["per_question"]["q2"]["answer"] == "0/1"


def test_refuses_mixed_configs(three_runs: list[dict[str, Any]]) -> None:
    other = _run(
        "r4",
        [_q("q1", True)],
        latency_total=1.0,
        agent_prompt=1,
        config={**_CONFIG, "retrieval_mode": "dense"},
    )
    with pytest.raises(ValueError, match="different config"):
        aggregate_runs([three_runs[0], other])


def test_refuses_empty() -> None:
    with pytest.raises(ValueError):
        aggregate_runs([])
