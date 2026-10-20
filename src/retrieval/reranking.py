"""Cohere-based reranker for hybrid search candidates.

Kept as its own module, decoupled from both vector store backends, so the
*rerank* step applies identically regardless of which backend produced the
candidate list -- mirrors how `fusion.py` keeps RRF backend-agnostic."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from src.retrieval.results import RetrievalResult

DEFAULT_COHERE_RERANK_MODEL = "rerank-v3.5"
COHERE_API_KEY_ENV_VAR = "COHERE_API_KEY"

# Cohere trial keys allow 10 rerank calls/minute. Waits double from the base
# and are capped at a full minute, so the last attempts always land in a fresh
# rate-limit window: 5 + 10 + 20 + 40 + 60 = 135s of waiting at most.
DEFAULT_MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 5.0
BACKOFF_CAP_S = 60.0

logger = logging.getLogger(__name__)


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


def _is_rate_limited(exc: Exception) -> bool:
    # Cohere's TooManyRequestsError (an ApiError) carries status_code=429;
    # matching on the attribute keeps tests free of the real `cohere` types.
    return getattr(exc, "status_code", None) == 429


def _retry_after_s(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None) or {}
    value = {k.lower(): v for k, v in headers.items()}.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _default_client() -> RerankClient:
    # Imported lazily so constructing a `CohereReranker` with an injected
    # client (as tests do) never requires the `cohere` package's own client
    # setup or an API key.
    import cohere

    return cohere.ClientV2(api_key=os.environ[COHERE_API_KEY_ENV_VAR])


class CohereReranker:
    """Reranks candidates by sending the query + candidate texts to Cohere's
    rerank endpoint and reordering to `top_k` by relevance score. Rate-limit
    (429) errors are retried with backoff; any other error is raised as is."""

    def __init__(
        self,
        client: RerankClient | None = None,
        model: str = DEFAULT_COHERE_RERANK_MODEL,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client or _default_client()
        self.model = model
        self.max_attempts = max_attempts
        self._sleep = sleep

    def rerank(
        self, query_text: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        response = self._rerank_with_backoff(
            query_text, [candidate.text for candidate in candidates], top_k
        )
        return [
            replace(candidates[result.index], score=result.relevance_score)
            for result in response.results
        ]

    def _rerank_with_backoff(
        self, query_text: str, documents: list[str], top_k: int
    ) -> RerankResponseLike:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.rerank(
                    model=self.model,
                    query=query_text,
                    documents=documents,
                    top_n=top_k,
                )
            except Exception as exc:
                if not _is_rate_limited(exc) or attempt == self.max_attempts:
                    raise
                backoff = min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_CAP_S)
                delay = min(_retry_after_s(exc) or backoff, BACKOFF_CAP_S)
                logger.warning(
                    "Cohere rate limit hit (attempt %d/%d); retrying in %.0fs",
                    attempt,
                    self.max_attempts,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable")
