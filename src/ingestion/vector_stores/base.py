"""Common interface both vector store backends implement, so retrieval code
can swap between them and comparisons stay apples-to-apples."""

from __future__ import annotations

from typing import Protocol

from src.ingestion.models import Chunk
from src.retrieval.results import RetrievalResult


class VectorStore(Protocol):
    def ensure_index(self) -> None:
        """Create the index/collection if it doesn't already exist."""
        ...

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Write embedded chunks (with metadata) into the store."""
        ...

    def query(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        """Embed `query_text` and return the `top_k` nearest chunks, most similar first."""
        ...

    def query_hybrid(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        """Dense + sparse retrieval, fused via Reciprocal Rank Fusion. The sparse
        leg is backend-native (BM25 for Weaviate, a local keyword pass for
        Pinecone); the fusion step is identical across backends."""
        ...

    def query_hybrid_reranked(
        self, query_text: str, top_k: int = 5, rerank_candidates: int = 20
    ) -> list[RetrievalResult]:
        """`query_hybrid` widened to `rerank_candidates`, then reranked down to
        `top_k` by Cohere Rerank. Kept alongside `query`/`query_hybrid` (both
        left intact) so all three retrieval modes can be compared in M4."""
        ...

    def close(self) -> None:
        """Release the underlying client connection."""
        ...
