"""Typed state and result schemas for the retrieve-grade-generate agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from src.retrieval.results import RetrievalResult

RetrievalMode = Literal["dense", "hybrid", "hybrid_reranked"]
GradeDecision = Literal["sufficient", "insufficient"]


class Citation(TypedDict):
    """A filing a claim in the answer came from. Built from chunk metadata, never
    from model output, so company/year/URL can't be hallucinated."""

    chunk_id: str
    company: str
    ticker: str
    fiscal_year: int
    form_type: str
    source_url: str


class TraceEvent(TypedDict):
    node: str
    retry_count: int
    detail: dict[str, Any]


class AgentState(TypedDict):
    question: str  # the user's original question; never rewritten
    search_query: str  # what `retrieve` searches with; starts as `question`
    retrieved_chunks: list[RetrievalResult]
    grade: GradeDecision | None
    grade_reason: str
    retry_count: int
    answer: str
    answerable: bool
    citations: list[Citation]
    # Append-only: each node returns its own events and LangGraph concatenates.
    trace: Annotated[list[TraceEvent], operator.add]


class AgentResult(TypedDict):
    answer: str
    answerable: bool
    citations: list[Citation]
    retrieved_chunks: list[RetrievalResult]
    retry_count: int
    trace: list[TraceEvent]


def initial_state(question: str) -> AgentState:
    return AgentState(
        question=question,
        search_query=question,
        retrieved_chunks=[],
        grade=None,
        grade_reason="",
        retry_count=0,
        answer="",
        answerable=False,
        citations=[],
        trace=[],
    )
