"""LangGraph wiring for the retrieve-grade-generate agent, plus the
`answer_question` entry point.

    START -> retrieve -> grade --sufficient--------------------------> generate -> END
                ^          |--insufficient, retries left--> reformulate --+
                |          |--insufficient, retries exhausted--> generate |
                +-----------------------------------------------------------+
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.llm import LLMClient, OpenAILLMClient, agent_model, grade_model
from src.agent.nodes import AgentDeps, AgentNodes
from src.agent.state import AgentResult, AgentState, RetrievalMode, initial_state
from src.ingestion.vector_stores.base import VectorStore

Backend = Literal["pinecone", "weaviate"]
DEFAULT_MAX_RETRIES = 2


def route_after_grade(
    state: AgentState, max_retries: int
) -> Literal["generate", "reformulate"]:
    """Sufficient -> generate. Insufficient -> reformulate while retries remain,
    else generate anyway so the graph always terminates."""
    if state["grade"] == "sufficient":
        return "generate"
    if state["retry_count"] < max_retries:
        return "reformulate"
    return "generate"


def build_graph(deps: AgentDeps) -> CompiledStateGraph:
    nodes = AgentNodes(deps)
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("reformulate", nodes.reformulate)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        lambda state: route_after_grade(state, deps.max_retries),
        {"generate": "generate", "reformulate": "reformulate"},
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()


def make_store(backend: Backend) -> VectorStore:
    # Imported lazily so the agent package doesn't need either SDK configured
    # unless a real store is actually built.
    from src.ingestion.config import IndexConfig, PineconeConfig, WeaviateConfig

    if backend == "pinecone":
        from src.ingestion.vector_stores.pinecone_store import PineconeStore

        return PineconeStore(IndexConfig(), PineconeConfig())
    if backend == "weaviate":
        from src.ingestion.vector_stores.weaviate_store import WeaviateStore

        return WeaviateStore(IndexConfig(), WeaviateConfig())
    raise ValueError(f"Unknown backend: {backend!r}")


def answer_question(
    question: str,
    retrieval_mode: RetrievalMode = "hybrid_reranked",
    *,
    store: VectorStore | None = None,
    backend: Backend = "pinecone",
    llm: LLMClient | None = None,
    grader_llm: LLMClient | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    top_k: int = 5,
) -> AgentResult:
    """Run the agent on `question` and return the answer with its citations,
    the chunks it was grounded in, and a trace of every node that ran.

    `store` overrides `backend`; a store built here is closed before returning.
    `llm` is the generate model (default: `AGENT_LLM_MODEL`); `grader_llm` is
    used for grade/reformulate (default: `llm` if injected, else
    `AGENT_GRADE_MODEL`, falling back to `AGENT_LLM_MODEL`)."""
    owns_store = store is None
    if store is None:
        store = make_store(backend)
    if grader_llm is None:
        grader_llm = llm if llm is not None else OpenAILLMClient(grade_model())
    if llm is None:
        llm = OpenAILLMClient(agent_model())

    deps = AgentDeps(
        store=store,
        generator_llm=llm,
        grader_llm=grader_llm,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        max_retries=max_retries,
    )
    # Each retry loop is 3 supersteps (reformulate, retrieve, grade) on top of
    # the 3-step straight path; the default limit would cap max_retries.
    recursion_limit = 3 * max_retries + 10
    try:
        final = build_graph(deps).invoke(
            initial_state(question), config={"recursion_limit": recursion_limit}
        )
    finally:
        if owns_store:
            store.close()

    return AgentResult(
        answer=final["answer"],
        answerable=final["answerable"],
        citations=final["citations"],
        retrieved_chunks=final["retrieved_chunks"],
        retry_count=final["retry_count"],
        trace=final["trace"],
    )
