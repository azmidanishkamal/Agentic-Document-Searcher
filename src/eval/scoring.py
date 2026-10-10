"""Per-question scoring, branched by the golden item's scoring mode."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.eval.golden import GoldenItem
from src.eval.judge import LLMJudge
from src.retrieval.results import RetrievalResult


@dataclass(frozen=True)
class AnswerScore:
    passed: bool
    reason: str


@dataclass(frozen=True)
class RetrievalScore:
    passed: bool
    reason: str
    # Informational only: the agent can legitimately ground in different
    # chunks that carry the same facts, so this is never the pass criterion.
    golden_chunk_id_recall: float | None


def score_answer(
    item: GoldenItem, answer: str, answerable: bool, judge: LLMJudge
) -> AnswerScore:
    mode = item.scoring_mode
    if mode == "negative":
        # Declining is correct regardless of citations; answering is fabrication.
        if answerable:
            return AnswerScore(
                False, "Agent answered (answerable=True) an unanswerable question."
            )
        return AnswerScore(True, "Agent declined (answerable=False) as expected.")
    if mode == "rubric":
        assert item.answer_rubric is not None  # guaranteed by scoring_mode
        verdict = judge.judge_rubric(item.question, item.answer_rubric, answer)
    else:
        verdict = judge.judge_reference(item.question, item.draft_answer, answer)
    return AnswerScore(verdict.passed, verdict.reason)


def golden_chunk_id_recall(
    item: GoldenItem, retrieved_ids: Sequence[str]
) -> float | None:
    if not item.relevant_chunk_ids:
        return None
    retrieved = set(retrieved_ids)
    hits = sum(1 for chunk_id in item.relevant_chunk_ids if chunk_id in retrieved)
    return hits / len(item.relevant_chunk_ids)


def score_retrieval(
    item: GoldenItem, chunks: Sequence[RetrievalResult], judge: LLMJudge
) -> RetrievalScore | None:
    """Whether the retrieved chunks contain the facts needed, judged by content.
    None for negatives: there are no facts to retrieve."""
    if item.scoring_mode == "negative":
        return None
    recall = golden_chunk_id_recall(item, [chunk.id for chunk in chunks])
    if not chunks:
        return RetrievalScore(False, "No chunks were retrieved.", recall)
    verdict = judge.judge_retrieval(item.question, item.reference_facts, chunks)
    return RetrievalScore(verdict.passed, verdict.reason, recall)
