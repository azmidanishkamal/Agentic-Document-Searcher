"""Local BM25 keyword index -- the sparse leg for backends with no native
full-text search (Pinecone). Built once from a backend's stored chunk
metadata and cached for reuse across `query_hybrid` calls."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.results import RetrievalResult, result_from_metadata


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    """A local, in-memory BM25 index over the documents supplied at construction."""

    def __init__(self, ids: list[str], metadatas: list[Mapping[str, Any]]) -> None:
        self._ids = ids
        self._metadatas = metadatas
        # BM25Okapi divides by corpus size internally and blows up on an
        # empty corpus, so skip building it when there's nothing to index.
        self._bm25 = (
            BM25Okapi([_tokenize(str(metadata["text"])) for metadata in metadatas])
            if metadatas
            else None
        )

    @classmethod
    def from_metadata(cls, documents: Iterable[tuple[str, Mapping[str, Any]]]) -> BM25Index:
        ids: list[str] = []
        metadatas: list[Mapping[str, Any]] = []
        for id_, metadata in documents:
            ids.append(id_)
            metadatas.append(metadata)
        return cls(ids, metadatas)

    def search(self, query_text: str, top_k: int) -> list[RetrievalResult]:
        if not self._ids:
            return []

        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            result_from_metadata(id_=self._ids[i], score=float(scores[i]), metadata=self._metadatas[i])
            for i in ranked_indices
            if scores[i] > 0
        ]
