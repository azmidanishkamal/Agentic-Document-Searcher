import json

import pytest

from src.eval import judge as judge_mod
from src.eval.golden import GoldenItem, load_golden_set
from src.eval.judge import LLMJudge
from src.eval.scoring import golden_chunk_id_recall, score_answer, score_retrieval
from tests.eval.conftest import RUBRIC, ScriptedLLM, make_chunk, verdict

# --- scoring mode selection ---


def test_negative_question_type_overrides_reference_eval_type(
    negative_item: GoldenItem,
) -> None:
    assert negative_item.eval_type == "reference"
    assert negative_item.scoring_mode == "negative"


def test_rubric_without_rubric_is_rejected() -> None:
    item = GoldenItem("q", "?", "multi_chunk_synthesis", "rubric", "a")
    with pytest.raises(ValueError, match="no answer_rubric"):
        _ = item.scoring_mode


def test_real_golden_set_modes() -> None:
    modes = {item.id: item.scoring_mode for item in load_golden_set()}
    assert len(modes) == 20
    assert {q for q, m in modes.items() if m == "rubric"} == {"q09", "q12"}
    assert {q for q, m in modes.items() if m == "negative"} == {"q18", "q19", "q20"}


# --- reference ---


def test_reference_pass_uses_judge_with_golden_and_agent_answer(
    reference_item: GoldenItem,
) -> None:
    llm = ScriptedLLM([verdict(True, "States $361M CAD.")])
    score = score_answer(reference_item, "C$361 million.", True, LLMJudge(llm))

    assert score.passed is True
    assert score.reason == "States $361M CAD."
    system, user = llm.calls[0]
    assert system == judge_mod.REFERENCE_SYSTEM
    assert "$361 million CAD" in user  # golden answer
    assert "C$361 million." in user  # agent answer
    assert reference_item.question in user


def test_reference_fail_propagates_reason(reference_item: GoldenItem) -> None:
    llm = ScriptedLLM([verdict(False, "Gives $89M, the FY2022 figure.")])
    score = score_answer(reference_item, "$89 million.", True, LLMJudge(llm))
    assert score.passed is False
    assert "FY2022" in score.reason


def test_malformed_judge_reply_raises(reference_item: GoldenItem) -> None:
    llm = ScriptedLLM([{"reason": "x", "passed": "yes"}])
    with pytest.raises(TypeError, match="Malformed"):
        score_answer(reference_item, "a", True, LLMJudge(llm))


# --- rubric ---


@pytest.mark.parametrize("passed", [True, False])
def test_rubric_judged_against_answer_rubric(
    rubric_item: GoldenItem, passed: bool
) -> None:
    llm = ScriptedLLM([verdict(passed, "reason")])
    score = score_answer(rubric_item, "Crane and Calpine.", True, LLMJudge(llm))

    assert score.passed is passed
    system, user = llm.calls[0]
    assert system == judge_mod.RUBRIC_SYSTEM
    assert json.dumps(RUBRIC, indent=2) in user
    assert RUBRIC["pass_condition"] in user and RUBRIC["fail_condition"] in user


# --- negative ---


def test_negative_declined_passes_without_judge(negative_item: GoldenItem) -> None:
    llm = ScriptedLLM([])
    score = score_answer(
        negative_item, "Not disclosed in the filings [c1].", False, LLMJudge(llm)
    )
    assert score.passed is True
    assert llm.calls == []


def test_negative_answered_fails(negative_item: GoldenItem) -> None:
    llm = ScriptedLLM([])
    score = score_answer(
        negative_item, "About 12% came from solar.", True, LLMJudge(llm)
    )
    assert score.passed is False
    assert "answerable=True" in score.reason
    assert llm.calls == []


# --- retrieval ---


def test_retrieval_judged_on_chunk_content(reference_item: GoldenItem) -> None:
    llm = ScriptedLLM([verdict(True, "Chunk states 361.")])
    chunks = [make_chunk("CCJ_2023_40-F_0100", "Net earnings attributable ... 361")]
    score = score_retrieval(reference_item, chunks, LLMJudge(llm))

    assert score is not None
    assert score.passed is True  # different chunk id than golden, still passes
    assert score.golden_chunk_id_recall == 0.0
    _, user = llm.calls[0]
    assert "Net earnings attributable ... 361" in user


def test_retrieval_none_for_negative(negative_item: GoldenItem) -> None:
    assert (
        score_retrieval(negative_item, [make_chunk("c1")], LLMJudge(ScriptedLLM([])))
        is None
    )


def test_retrieval_empty_chunks_fails_without_judge(reference_item: GoldenItem) -> None:
    llm = ScriptedLLM([])
    score = score_retrieval(reference_item, [], LLMJudge(llm))
    assert score is not None and score.passed is False
    assert llm.calls == []


def test_golden_chunk_id_recall() -> None:
    item = GoldenItem("q", "?", "t", "reference", "a", relevant_chunk_ids=["a", "b"])
    assert golden_chunk_id_recall(item, ["b", "c"]) == 0.5
    assert (
        golden_chunk_id_recall(GoldenItem("q", "?", "t", "reference", "a"), ["b"])
        is None
    )


def test_rubric_retrieval_checks_required_points_only(rubric_item: GoldenItem) -> None:
    llm = ScriptedLLM([verdict(True)])
    score_retrieval(rubric_item, [make_chunk("c1")], LLMJudge(llm))
    _, user = llm.calls[0]
    assert "- Crane restart\n- Calpine acquisition" in user
