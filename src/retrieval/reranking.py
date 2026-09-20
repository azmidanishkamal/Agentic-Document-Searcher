"""Cohere-based reranker for hybrid search candidates.

Kept as its own module, decoupled from both vector store backends, so the
*rerank* step applies identically regardless of which backend produced the
candidate list -- mirrors how `fusion.py` keeps RRF backend-agnostic."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Protocol

from src.retrieval.results import RetrievalResult

DEFAULT_COHERE_RERANK_MODEL = "rerank-v3.5"
COHERE_API_KEY_ENV_VAR = "COHERE_API_KEY"


class RerankResultLike(Protocol):
    index: int
    relevance_score: float


class RerankResponseLike(Protocol):
    results: list[RerankResultLike]


class RerankClient(Protocol):
    """Shape of the Cohere client method we depend on, so tests can inject a
    stand-in without a network call or a real `cohere` client."""

    def rerank(
        self, *, model: str, query: str, documents: list[str], top_n: int
    ) -> RerankResponseLike: ...


def _default_client() -> RerankClient:
    # Imported lazily so constructing a `CohereReranker` with an injected
    # client (as tests do) never requires the `cohere` package's own client
    # setup or an API key.
    import cohere

    return cohere.ClientV2(api_key=os.environ[COHERE_API_KEY_ENV_VAR])


class CohereReranker:
    """Reranks candidates by sending the query + candidate texts to Cohere's
    rerank endpoint and reordering to `top_k` by relevance score."""

    def __init__(self, client: RerankClient | None = None, model: str = DEFAULT_COHERE_RERANK_MODEL) -> None:
        self.client = client or _default_client()
        self.model = model

    def rerank(
        self, query_text: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        response = self.client.rerank(
            model=self.model,
            query=query_text,
            documents=[candidate.text for candidate in candidates],
            top_n=top_k,
        )
        return [
            replace(candidates[result.index], score=result.relevance_score)
            for result in response.results
        ]
