"""Reciprocal Rank Fusion (RRF), used identically by every backend's
`query_hybrid` so the *fusion* step of hybrid search stays comparable across
Pinecone and Weaviate, regardless of how each backend's sparse leg works
under the hood."""

from __future__ import annotations

from dataclasses import replace

from src.retrieval.results import RetrievalResult

DEFAULT_RRF_K = 60


def default_candidate_pool_size(top_k: int) -> int:
    """How many hits to pull from each leg before fusing. Wider than `top_k`
    so RRF has enough overlap between the dense and sparse rankings to work
    with, not just the final page of results."""
    return max(top_k * 4, 20)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    top_k: int,
    k: int = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    """Fuse multiple best-first ranked lists into one via RRF:
    `score(item) = sum(1 / (k + rank))` over every list the item appears in
    (1-indexed rank). The fused score replaces each result's original
    similarity/BM25 score, since those are on different scales per leg and
    are no longer meaningful once fused."""
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievalResult] = {}
    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(result.id, result)

    fused = [replace(first_seen[id_], score=score) for id_, score in scores.items()]
    fused.sort(key=lambda result: result.score, reverse=True)
    return fused[:top_k]
