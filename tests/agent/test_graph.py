from typing import Any

import pytest

from src.agent import prompts
from src.agent.graph import answer_question, route_after_grade
from src.agent.state import AgentState, initial_state
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


def _chunk(
    id_: str, text: str = "Cameco holds a 49% interest in Westinghouse."
) -> RetrievalResult:
    return RetrievalResult(
        id=id_, text=text, score=0.9, filing=_FILING, chunk_index=0, total_chunks=1
    )


class FakeStore:
    """Records which retrieval method was called with which query."""

    def __init__(self, chunks: list[RetrievalResult]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def query(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append(("dense", query_text))
        return self.chunks[:top_k]

    def query_hybrid(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append(("hybrid", query_text))
        return self.chunks[:top_k]

    def query_hybrid_reranked(
        self, query_text: str, top_k: int = 5, rerank_candidates: int = 20
    ) -> list[RetrievalResult]:
        self.calls.append(("hybrid_reranked", query_text))
        return self.chunks[:top_k]

    def close(self) -> None:
        self.closed = True


class ScriptedLLM:
    """Returns queued responses per schema, so each LLM step can be scripted
    independently, and records every call."""

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self.responses = {name: list(queue) for name, queue in responses.items()}
        self.calls: list[tuple[str, str]] = []

    def complete_json(
        self, *, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append((schema_name, user))
        return self.responses[schema_name].pop(0)


def _part(
    part: str, supported: bool, evidence: list[str] | None = None
) -> dict[str, Any]:
    return {"part": part, "supported": supported, "evidence_chunk_ids": evidence or []}


def _grade(
    decision: str, reason: str = "r", sub_parts: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if sub_parts is None:
        sub_parts = [_part("the fact asked for", decision == "sufficient", ["c1"])]
    return {"sub_parts": sub_parts, "decision": decision, "reason": reason}


def _answer(chunk_ids: list[str], answerable: bool = True) -> dict[str, Any]:
    return {
        "answer": "Cameco holds 49% [c1].",
        "answerable": answerable,
        "cited_chunk_ids": chunk_ids,
    }


def _nodes(result: dict[str, Any]) -> list[str]:
    return [event["node"] for event in result["trace"]]


def _state(grade: str, retry_count: int) -> AgentState:
    state = initial_state("q")
    state["grade"] = grade
    state["retry_count"] = retry_count
    return state


# --- route_after_grade (pure routing) ---


def test_route_sufficient_goes_to_generate() -> None:
    assert route_after_grade(_state("sufficient", 0), max_retries=2) == "generate"


def test_route_insufficient_with_retries_left_reformulates() -> None:
    assert route_after_grade(_state("insufficient", 0), max_retries=2) == "reformulate"
    assert route_after_grade(_state("insufficient", 1), max_retries=2) == "reformulate"


def test_route_insufficient_with_retries_exhausted_goes_to_generate() -> None:
    assert route_after_grade(_state("insufficient", 2), max_retries=2) == "generate"
    assert route_after_grade(_state("insufficient", 0), max_retries=0) == "generate"


# --- full graph runs with fakes ---


def test_sufficient_context_goes_straight_to_generate() -> None:
    store = FakeStore([_chunk("c1")])
    llm = ScriptedLLM(
        {
            prompts.GRADE_SCHEMA_NAME: [_grade("sufficient")],
            prompts.GENERATE_SCHEMA_NAME: [_answer(["c1"])],
        }
    )

    result = answer_question("Cameco Westinghouse stake?", store=store, llm=llm)

    assert _nodes(result) == ["retrieve", "grade", "generate"]
    assert result["retry_count"] == 0
    assert store.calls == [("hybrid_reranked", "Cameco Westinghouse stake?")]
    assert result["answerable"] is True
    assert result["retrieved_chunks"] == [_chunk("c1")]
    assert not store.closed  # injected stores are the caller's to close


def test_insufficient_then_sufficient_loops_once_with_rewritten_query() -> None:
    store = FakeStore([_chunk("c1")])
    llm = ScriptedLLM(
        {
            prompts.GRADE_SCHEMA_NAME: [
                _grade("insufficient", "missing stake %"),
                _grade("sufficient"),
            ],
            prompts.REFORMULATE_SCHEMA_NAME: [
                {"query": "Cameco Westinghouse ownership percentage"}
            ],
            prompts.GENERATE_SCHEMA_NAME: [_answer(["c1"])],
        }
    )

    result = answer_question("Cameco Westinghouse stake?", store=store, llm=llm)

    assert _nodes(result) == [
        "retrieve",
        "grade",
        "reformulate",
        "retrieve",
        "grade",
        "generate",
    ]
    assert result["retry_count"] == 1
    assert [query for _, query in store.calls] == [
        "Cameco Westinghouse stake?",
        "Cameco Westinghouse ownership percentage",
    ]
    reformulate_prompt = next(
        user for name, user in llm.calls if name == prompts.REFORMULATE_SCHEMA_NAME
    )
    assert "missing stake %" in reformulate_prompt


def test_retries_exhausted_still_generates_and_terminates() -> None:
    store = FakeStore([_chunk("c1")])
    llm = ScriptedLLM(
        {
            prompts.GRADE_SCHEMA_NAME: [_grade("insufficient", "nothing relevant")] * 3,
            prompts.REFORMULATE_SCHEMA_NAME: [{"query": "q2"}, {"query": "q3"}],
            prompts.GENERATE_SCHEMA_NAME: [_answer([], answerable=False)],
        }
    )

    result = answer_question("Unanswerable?", store=store, llm=llm, max_retries=2)

    assert _nodes(result) == [
        "retrieve", "grade", "reformulate",
        "retrieve", "grade", "reformulate",
        "retrieve", "grade", "generate",
    ]  # fmt: skip
    assert result["retry_count"] == 2
    assert len(store.calls) == 3
    assert result["answerable"] is False
    assert result["trace"][-1]["detail"]["forced_after_retries"] is True
    generate_prompt = next(
        user for name, user in llm.calls if name == prompts.GENERATE_SCHEMA_NAME
    )
    assert "nothing relevant" in generate_prompt


def test_large_max_retries_does_not_hit_recursion_limit() -> None:
    store = FakeStore([_chunk("c1")])
    llm = ScriptedLLM(
        {
            prompts.GRADE_SCHEMA_NAME: [_grade("insufficient")] * 11,
            prompts.REFORMULATE_SCHEMA_NAME: [{"query": f"q{i}"} for i in range(10)],
            prompts.GENERATE_SCHEMA_NAME: [_answer([], answerable=False)],
        }
    )

    result = answer_question("q", store=store, llm=llm, max_retries=10)

    assert result["retry_count"] == 10


@pytest.mark.parametrize("mode", ["dense", "hybrid", "hybrid_reranked"])
def test_retrieval_mode_selects_store_method(mode: str) -> None:
    store = FakeStore([_chunk("c1")])
    llm = ScriptedLLM(
        {
            prompts.GRADE_SCHEMA_NAME: [_grade("sufficient")],
            prompts.GENERATE_SCHEMA_NAME: [_answer(["c1"])],
        }
    )

    answer_question("q", retrieval_mode=mode, store=store, llm=llm)

    assert store.calls == [(mode, "q")]


def test_separate_grader_llm_handles_grade_and_reformulate() -> None:
    store = FakeStore([_chunk("c1")])
    grader = ScriptedLLM(
        {
            prompts.GRADE_SCHEMA_NAME: [_grade("insufficient"), _grade("sufficient")],
            prompts.REFORMULATE_SCHEMA_NAME: [{"query": "q2"}],
        }
    )
    generator = ScriptedLLM({prompts.GENERATE_SCHEMA_NAME: [_answer(["c1"])]})

    answer_question("q", store=store, llm=generator, grader_llm=grader)

    assert {name for name, _ in grader.calls} == {
        prompts.GRADE_SCHEMA_NAME,
        prompts.REFORMULATE_SCHEMA_NAME,
    }
    assert [name for name, _ in generator.calls] == [prompts.GENERATE_SCHEMA_NAME]


# --- citations / grounding ---


def test_citations_come_from_chunk_metadata_and_unknown_ids_are_dropped() -> None:
    store = FakeStore([_chunk("c1"), _chunk("c2")])
    llm = ScriptedLLM(
        {
            prompts.GRADE_SCHEMA_NAME: [_grade("sufficient")],
            prompts.GENERATE_SCHEMA_NAME: [_answer(["c2", "made_up", "c2"])],
        }
    )

    result = answer_question("q", store=store, llm=llm)

    assert result["citations"] == [
        {
            "chunk_id": "c2",
            "company": "Cameco Corporation",
            "ticker": "CCJ",
            "fiscal_year": 2023,
            "form_type": "40-F",
            "source_url": _FILING.source_url,
        }
    ]
    assert result["trace"][-1]["detail"]["dropped_unknown_citation_ids"] == ["made_up"]


def test_no_chunks_declines_without_calling_llm() -> None:
    store = FakeStore([])
    llm = ScriptedLLM({prompts.REFORMULATE_SCHEMA_NAME: [{"query": "q2"}]})

    result = answer_question("q", store=store, llm=llm, max_retries=1)

    assert _nodes(result) == [
        "retrieve",
        "grade",
        "reformulate",
        "retrieve",
        "grade",
        "generate",
    ]
    assert result["answerable"] is False
    assert result["answer"] == prompts.UNANSWERABLE_NO_CONTEXT
    assert result["citations"] == []
    assert [name for name, _ in llm.calls] == [prompts.REFORMULATE_SCHEMA_NAME]


def test_generate_prompt_includes_filing_metadata_for_each_chunk() -> None:
    text = prompts.format_chunks([_chunk("CCJ_2023_40-F_0436")])

    assert "[CCJ_2023_40-F_0436]" in text
    assert "Cameco Corporation" in text
    assert "fiscal year 2023" in text
    assert _FILING.source_url in text


# --- sub-part coverage grading ---


class SequencedStore(FakeStore):
    """Returns a different result list per retrieval call."""

    def __init__(self, per_call: list[list[RetrievalResult]]) -> None:
        super().__init__([])
        self.per_call = per_call

    def query_hybrid_reranked(
        self, query_text: str, top_k: int = 5, rerank_candidates: int = 20
    ) -> list[RetrievalResult]:
        self.calls.append(("hybrid_reranked", query_text))
        return self.per_call[len(self.calls) - 1]


def test_unsupported_sub_part_overrides_a_sufficient_verdict() -> None:
    store = FakeStore([_chunk("c1")])
    inconsistent = _grade(
        "sufficient",
        sub_parts=[
            _part("Cameco side", True, ["c1"]),
            _part("Constellation side", False),
        ],
    )
    llm = ScriptedLLM(
        {
            prompts.GRADE_SCHEMA_NAME: [inconsistent, _grade("sufficient")],
            prompts.REFORMULATE_SCHEMA_NAME: [{"query": "q2"}],
            prompts.GENERATE_SCHEMA_NAME: [_answer(["c1"])],
        }
    )

    result = answer_question("compare", store=store, llm=llm)

    grades = [e["detail"]["decision"] for e in result["trace"] if e["node"] == "grade"]
    assert grades == ["insufficient", "sufficient"]
    assert result["retry_count"] == 1


def test_reformulate_prompt_lists_the_missing_sub_parts() -> None:
    store = FakeStore([_chunk("c1")])
    llm = ScriptedLLM(
        {
            prompts.GRADE_SCHEMA_NAME: [
                _grade(
                    "insufficient",
                    reason="no purchase price",
                    sub_parts=[
                        _part("which acquisition", True, ["c1"]),
                        _part("acquisition purchase price", False),
                    ],
                ),
                _grade("sufficient"),
            ],
            prompts.REFORMULATE_SCHEMA_NAME: [{"query": "acquisition purchase price"}],
            prompts.GENERATE_SCHEMA_NAME: [_answer(["c1"])],
        }
    )

    result = answer_question("q", store=store, llm=llm)

    reformulate_prompt = next(
        user for name, user in llm.calls if name == prompts.REFORMULATE_SCHEMA_NAME
    )
    assert "- acquisition purchase price" in reformulate_prompt
    assert "- which acquisition" not in reformulate_prompt
    assert "no purchase price" in reformulate_prompt
    reformulate_event = next(e for e in result["trace"] if e["node"] == "reformulate")
    assert reformulate_event["detail"]["missing_parts"] == [
        "acquisition purchase price"
    ]


def test_evidence_for_supported_parts_survives_a_targeted_retry() -> None:
    cameco = _chunk("cameco", "Cameco uranium supply risk")
    noise = _chunk("noise", "unrelated")
    constellation = _chunk("ceg", "Constellation fuel procurement")
    store = SequencedStore([[cameco, noise], [constellation]])
    llm = ScriptedLLM(
        {
            prompts.GRADE_SCHEMA_NAME: [
                _grade(
                    "insufficient",
                    sub_parts=[
                        _part("Cameco view", True, ["cameco"]),
                        _part("Constellation view", False),
                    ],
                ),
                _grade(
                    "sufficient",
                    sub_parts=[
                        _part("Cameco view", True, ["cameco"]),
                        _part("Constellation view", True, ["ceg"]),
                    ],
                ),
            ],
            prompts.REFORMULATE_SCHEMA_NAME: [{"query": "Constellation fuel risk"}],
            prompts.GENERATE_SCHEMA_NAME: [_answer(["cameco", "ceg"])],
        }
    )

    result = answer_question("compare", store=store, llm=llm)

    # Supported evidence is kept, unsupported noise is dropped, new hits added.
    assert [c.id for c in result["retrieved_chunks"]] == ["cameco", "ceg"]
    second_grade_prompt = [u for n, u in llm.calls if n == prompts.GRADE_SCHEMA_NAME][1]
    assert "[cameco]" in second_grade_prompt
    assert "[noise]" not in second_grade_prompt
    assert [c["chunk_id"] for c in result["citations"]] == ["cameco", "ceg"]
