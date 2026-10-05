"""Graph nodes for the retrieve-grade-generate agent.

Each node takes the current `AgentState` and returns a partial update; the
runtime dependencies (store, LLMs, retrieval mode) are captured on `AgentNodes`
rather than carried in state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent import prompts
from src.agent.llm import LLMClient
from src.agent.state import AgentState, Citation, RetrievalMode, TraceEvent
from src.ingestion.vector_stores.base import VectorStore
from src.retrieval.results import RetrievalResult


@dataclass(frozen=True)
class AgentDeps:
    store: VectorStore
    generator_llm: LLMClient  # generate step
    grader_llm: LLMClient  # grade + reformulate steps; may be a cheaper model
    retrieval_mode: RetrievalMode = "hybrid_reranked"
    top_k: int = 5
    max_retries: int = 2


def run_retrieval(
    store: VectorStore, mode: RetrievalMode, query: str, top_k: int
) -> list[RetrievalResult]:
    if mode == "dense":
        return store.query(query, top_k=top_k)
    if mode == "hybrid":
        return store.query_hybrid(query, top_k=top_k)
    if mode == "hybrid_reranked":
        return store.query_hybrid_reranked(query, top_k=top_k)
    raise ValueError(f"Unknown retrieval mode: {mode!r}")


def citations_for(
    cited_ids: list[str], chunks: list[RetrievalResult]
) -> tuple[list[Citation], list[str]]:
    """Map cited chunk ids to citations from chunk metadata, in citation order.
    Returns (citations, ids the model cited that weren't in the context)."""
    by_id = {chunk.id: chunk for chunk in chunks}
    citations: list[Citation] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for chunk_id in cited_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = by_id.get(chunk_id)
        if chunk is None:
            unknown.append(chunk_id)
            continue
        filing = chunk.filing
        citations.append(
            Citation(
                chunk_id=chunk.id,
                company=filing.company,
                ticker=filing.ticker,
                fiscal_year=filing.fiscal_year,
                form_type=filing.form_type,
                source_url=filing.source_url,
            )
        )
    return citations, unknown


def _event(node: str, state: AgentState, **detail: Any) -> list[TraceEvent]:
    return [TraceEvent(node=node, retry_count=state["retry_count"], detail=detail)]


class AgentNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        chunks = run_retrieval(
            self.deps.store,
            self.deps.retrieval_mode,
            state["search_query"],
            self.deps.top_k,
        )
        return {
            "retrieved_chunks": chunks,
            "trace": _event(
                "retrieve",
                state,
                query=state["search_query"],
                mode=self.deps.retrieval_mode,
                chunk_ids=[chunk.id for chunk in chunks],
            ),
        }

    def grade(self, state: AgentState) -> dict[str, Any]:
        chunks = state["retrieved_chunks"]
        if not chunks:
            decision, reason = "insufficient", "No chunks were retrieved."
        else:
            result = self.deps.grader_llm.complete_json(
                system=prompts.GRADE_SYSTEM,
                user=prompts.grade_user_prompt(state["question"], chunks),
                schema_name=prompts.GRADE_SCHEMA_NAME,
                schema=prompts.GRADE_SCHEMA,
            )
            decision, reason = result["decision"], result["reason"]
        return {
            "grade": decision,
            "grade_reason": reason,
            "trace": _event("grade", state, decision=decision, reason=reason),
        }

    def reformulate(self, state: AgentState) -> dict[str, Any]:
        result = self.deps.grader_llm.complete_json(
            system=prompts.REFORMULATE_SYSTEM,
            user=prompts.reformulate_user_prompt(
                state["question"], state["search_query"], state["grade_reason"]
            ),
            schema_name=prompts.REFORMULATE_SCHEMA_NAME,
            schema=prompts.REFORMULATE_SCHEMA,
        )
        new_query = result["query"].strip() or state["question"]
        return {
            "search_query": new_query,
            "retry_count": state["retry_count"] + 1,
            "trace": _event(
                "reformulate",
                state,
                previous_query=state["search_query"],
                new_query=new_query,
            ),
        }

    def generate(self, state: AgentState) -> dict[str, Any]:
        chunks = state["retrieved_chunks"]
        if not chunks:
            return {
                "answer": prompts.UNANSWERABLE_NO_CONTEXT,
                "answerable": False,
                "citations": [],
                "trace": _event("generate", state, answerable=False, skipped_llm=True),
            }

        grade_reason = state["grade_reason"] if state["grade"] != "sufficient" else None
        result = self.deps.generator_llm.complete_json(
            system=prompts.GENERATE_SYSTEM,
            user=prompts.generate_user_prompt(state["question"], chunks, grade_reason),
            schema_name=prompts.GENERATE_SCHEMA_NAME,
            schema=prompts.GENERATE_SCHEMA,
        )
        citations, unknown_ids = citations_for(result["cited_chunk_ids"], chunks)
        return {
            "answer": result["answer"],
            "answerable": result["answerable"],
            "citations": citations,
            "trace": _event(
                "generate",
                state,
                answerable=result["answerable"],
                forced_after_retries=grade_reason is not None,
                cited_chunk_ids=[c["chunk_id"] for c in citations],
                dropped_unknown_citation_ids=unknown_ids,
            ),
        }
