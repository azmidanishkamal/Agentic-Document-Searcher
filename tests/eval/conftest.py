from typing import Any

import pytest

from src.eval.golden import GoldenItem
from src.ingestion.models import FilingMetadata
from src.retrieval.results import RetrievalResult

_FILING = FilingMetadata(
    company="Cameco Corporation",
    ticker="CCJ",
    cik="0000002012",
    form_type="40-F",
    fiscal_year=2023,
    filing_date="2024-03-15",
    accession_number="0000002012-24-000001",
    source_url="https://www.sec.gov/Archives/edgar/data/2012/000000201224000001",
)

RUBRIC = {
    "required_points": ["Crane restart", "Calpine acquisition"],
    "credit_points": ["Crane: ~$1.6B capex"],
    "pass_condition": "Identifies both Crane and Calpine.",
    "fail_condition": "Omits one or both initiatives.",
}


def make_chunk(
    id_: str, text: str = "Cameco holds a 49% interest in Westinghouse."
) -> RetrievalResult:
    return RetrievalResult(
        id=id_, text=text, score=0.9, filing=_FILING, chunk_index=0, total_chunks=1
    )


class ScriptedLLM:
    """Pops queued replies in order and records every (system, user) prompt."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete_json(
        self, *, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append((system, user))
        return self.replies.pop(0)


def verdict(passed: bool, reason: str = "r") -> dict[str, Any]:
    return {"reason": reason, "passed": passed}


@pytest.fixture
def reference_item() -> GoldenItem:
    return GoldenItem(
        id="q02",
        question="What was Cameco's net earnings attributable to equity holders for FY2023?",
        question_type="single_fact_lookup_table",
        eval_type="reference",
        draft_answer="$361 million CAD",
        relevant_chunk_ids=["CCJ_2024_40-F_0388"],
    )


@pytest.fixture
def rubric_item() -> GoldenItem:
    return GoldenItem(
        id="q09",
        question="What are Constellation's largest growth initiatives?",
        question_type="multi_chunk_synthesis",
        eval_type="rubric",
        draft_answer="Crane and Calpine.",
        answer_rubric=RUBRIC,
    )


@pytest.fixture
def negative_item() -> GoldenItem:
    # Mirrors the golden file: negatives carry eval_type "reference".
    return GoldenItem(
        id="q18",
        question="What percentage of Cameco's FY2025 revenue came from solar panels?",
        question_type="negative_unanswerable",
        eval_type="reference",
        draft_answer="Not present in corpus.",
    )
