import json
from pathlib import Path
from typing import Any

import pytest

from src.agent.llm import LLMUsage
from src.eval import cost as cost_mod
from src.eval.cost import TokenPrice, recost_results
from src.eval.judge import LLMJudge
from src.eval.runner import EvalConfig, run_eval
from tests.eval.conftest import ScriptedLLM, make_chunk, verdict


class FakeStore:
    def __init__(self) -> None:
        self.closed = False

    def query_hybrid_reranked(
        self, query_text: str, top_k: int = 5, **_: Any
    ) -> list[Any]:
        return [make_chunk("c1")]

    def close(self) -> None:
        self.closed = True


class UsageLLM(ScriptedLLM):
    """Scripted LLM that also records fixed token usage like OpenAILLMClient."""

    def __init__(self, replies: list[dict[str, Any]], model: str = "m") -> None:
        super().__init__(replies)
        self.model = model
        self.usage: list[LLMUsage] = []

    def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        self.usage.append(LLMUsage(self.model, prompt_tokens=100, completion_tokens=10))
        return super().complete_json(**kwargs)


def _agent_llms(answerable: bool) -> Any:
    def factory() -> tuple[UsageLLM, UsageLLM]:
        grade = {
            "sub_parts": [
                {"part": "p", "supported": True, "evidence_chunk_ids": ["c1"]}
            ],
            "decision": "sufficient",
            "reason": "ok",
        }
        answer = {
            "answer": "C$361 million [c1].",
            "answerable": answerable,
            "cited_chunk_ids": ["c1"],
        }
        return UsageLLM([answer]), UsageLLM([grade])

    return factory


@pytest.fixture
def golden_file(tmp_path: Path) -> Path:
    rows = [
        {
            "id": "q02",
            "question": "Net earnings?",
            "question_type": "single_fact_lookup_table",
            "eval_type": "reference",
            "draft_answer": "$361 million CAD",
            "relevant_chunk_ids": ["c1"],
        },
        {
            "id": "q18",
            "question": "Solar revenue?",
            "question_type": "negative_unanswerable",
            "eval_type": "reference",
            "draft_answer": "Not present.",
        },
    ]
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_run_eval_end_to_end_with_fakes(golden_file: Path) -> None:
    # q02: answer verdict + retrieval verdict. q18: no judge calls (agent answers -> fabricated).
    judge_llm = UsageLLM(
        [verdict(True, "matches"), verdict(True, "present")], model="judge"
    )
    store = FakeStore()
    results = run_eval(
        EvalConfig(golden_path=golden_file),
        judge=LLMJudge(judge_llm),
        store=store,  # type: ignore[arg-type]
        agent_llms=_agent_llms(answerable=True),
    )

    assert not store.closed  # injected store is the caller's to close
    q02, q18 = results["questions"]
    assert q02["answer_score"] == {"passed": True, "reason": "matches"}
    assert q02["retrieval_score"]["passed"] is True
    assert q02["retrieval_score"]["golden_chunk_id_recall"] == 1.0
    assert q02["agent"]["retry_count"] == 0
    assert q02["usage"]["rerank_searches"] == 1
    assert len(q02["usage"]["agent"]) == 2  # grade + generate
    assert len(q02["usage"]["judge"]) == 2  # answer + retrieval
    assert q18["scoring_mode"] == "negative"
    assert q18["answer_score"]["passed"] is False
    assert q18["retrieval_score"] is None
    assert q18["usage"]["judge"] == []  # judge usage attributed per question

    summary = results["summary"]
    assert summary["answer_pass_rate"] == 0.5
    assert summary["retrieval_n"] == 1 and summary["retrieval_pass_rate"] == 1.0
    assert summary["by_scoring_mode"]["negative"] == {
        "n": 1,
        "passed": 0,
        "pass_rate": 0.0,
    }
    assert summary["by_question_type"]["single_fact_lookup_table"]["pass_rate"] == 1.0
    assert summary["tokens"]["agent_prompt"] == 400
    assert summary["cost_usd"]["complete"] is False  # model "m" is unpriced
    assert results["run"]["config"]["backend"] == "pinecone"
    assert results["run"]["config"]["retrieval_mode"] == "hybrid_reranked"


def test_agent_error_counts_as_failure(golden_file: Path) -> None:
    def broken() -> tuple[ScriptedLLM, ScriptedLLM]:
        return ScriptedLLM([]), ScriptedLLM([])  # pop from empty -> IndexError

    results = run_eval(
        EvalConfig(golden_path=golden_file, question_ids=("q02",)),
        judge=LLMJudge(ScriptedLLM([])),
        store=FakeStore(),  # type: ignore[arg-type]
        agent_llms=broken,
    )
    (q02,) = results["questions"]
    assert q02["error"].startswith("agent: IndexError")
    assert q02["answer_score"]["passed"] is False
    assert results["summary"]["n_errors"] == 1


def test_recost_from_saved_results(golden_file: Path) -> None:
    results = run_eval(
        EvalConfig(golden_path=golden_file, question_ids=("q02",)),
        judge=LLMJudge(UsageLLM([verdict(True), verdict(True)], model="judge")),
        store=FakeStore(),  # type: ignore[arg-type]
        agent_llms=_agent_llms(answerable=True),
    )
    assert results["questions"][0]["cost_usd"]["total"] is None

    # Round-trip through JSON, as a saved file would be.
    saved = json.loads(json.dumps(results))
    prices = {"m": TokenPrice(1.0, 10.0), "judge": TokenPrice(2.0, 20.0)}
    recost_results(saved, prices, rerank_price=0.002)

    q = saved["questions"][0]["cost_usd"]
    # agent: 2 calls * (100*1 + 10*10)/1e6; judge: 2 * (100*2 + 10*20)/1e6
    assert q["agent"] == pytest.approx(400e-6)
    assert q["judge"] == pytest.approx(800e-6)
    assert q["rerank"] == pytest.approx(0.002)
    assert saved["summary"]["cost_usd"]["complete"] is True
    assert saved["summary"]["cost_usd"]["total"] == pytest.approx(0.0032)


def test_partial_prices_leave_cost_incomplete() -> None:
    usage = [
        {"model": "known", "prompt_tokens": 1, "completion_tokens": 1},
        {"model": "unknown", "prompt_tokens": 1, "completion_tokens": 1},
    ]
    assert cost_mod.llm_cost(usage, {"known": TokenPrice(1, 1)}) is None
    assert cost_mod.rerank_cost(0, None) == 0.0
