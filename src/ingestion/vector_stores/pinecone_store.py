"""Pinecone backend: serverless index, cosine metric, matching `IndexConfig`."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from src.ingestion.config import EmbeddingConfig, IndexConfig, PineconeConfig
from src.ingestion.embeddings import OpenAIEmbedder
from src.ingestion.models import Chunk
from src.retrieval.embedding import embed_query
from src.retrieval.fusion import default_candidate_pool_size, reciprocal_rank_fusion
from src.retrieval.keyword_search import BM25Index
from src.retrieval.reranking import CohereReranker
from src.retrieval.results import RetrievalResult, result_from_metadata

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 100


class PineconeStore:
    def __init__(
        self,
        index_config: IndexConfig,
        pinecone_config: PineconeConfig,
        client: Pinecone | None = None,
        embedder: OpenAIEmbedder | None = None,
        reranker: CohereReranker | None = None,
    ) -> None:
        self.index_config = index_config
        self.pinecone_config = pinecone_config
        self.client = client or Pinecone()
        self._index = None
        # Lazy defaults so constructing a store never requires an OpenAI or
        # Cohere API key unless `query`/`query_hybrid_reranked` is called.
        self.embedder = embedder
        self.reranker = reranker
        # Pinecone has no native full-text search; built lazily by scanning
        # the index once, then cached for reuse across `query_hybrid` calls.
        self._bm25_index: BM25Index | None = None

    def ensure_index(self) -> None:
        existing = {index.name for index in self.client.list_indexes()}
        if self.index_config.name not in existing:
            logger.info("Creating Pinecone index %s", self.index_config.name)
            self.client.create_index(
                name=self.index_config.name,
                dimension=self.index_config.dimension,
                metric=self.index_config.metric,
                spec=ServerlessSpec(
                    cloud=self.pinecone_config.cloud, region=self.pinecone_config.region
                ),
            )
            while not self.client.describe_index(self.index_config.name).status["ready"]:
                time.sleep(1)
        self._index = self.client.Index(self.index_config.name)

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if self._index is None:
            self.ensure_index()

        vectors = [
            {"id": chunk.chunk_id, "values": chunk.embedding, "metadata": chunk.to_metadata_dict()}
            for chunk in chunks
        ]
        for start in range(0, len(vectors), _UPSERT_BATCH_SIZE):
            batch = vectors[start : start + _UPSERT_BATCH_SIZE]
            self._index.upsert(vectors=batch)

    def query(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._index is None:
            self.ensure_index()
        if self.embedder is None:
            self.embedder = OpenAIEmbedder(EmbeddingConfig())

        vector = embed_query(self.embedder, query_text)
        response = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            result_from_metadata(id_=match.id, score=match.score, metadata=match.metadata)
            for match in response.matches
        ]

    def _keyword_index(self) -> BM25Index:
        if self._bm25_index is None:
            if self._index is None:
                self.ensure_index()
            cache_dir = self.pinecone_config.bm25_cache_dir
            if cache_dir is None:
                documents = self._fetch_bm25_documents()
            else:
                documents = self._cached_bm25_documents(cache_dir)
            self._bm25_index = BM25Index.from_metadata(documents)
        return self._bm25_index

    def _fetch_bm25_documents(self) -> list[tuple[str, dict]]:
        """Scan the whole index (list + fetch). Fetch also returns the
        embeddings, which is why this is cached locally."""
        documents: list[tuple[str, dict]] = []
        for page in self._index.list():
            ids = [item.id for item in page.vectors]
            fetched = self._index.fetch(ids=ids)
            documents.extend((id_, vector.metadata) for id_, vector in fetched.vectors.items())
        return documents

    def _cached_bm25_documents(self, cache_dir: Path) -> list[tuple[str, dict]]:
        """BM25 corpus from the local cache, keyed by index name + vector count
        so re-ingestion (which changes the count) never serves a stale cache."""
        name = self.index_config.name
        vector_count = int(self._index.describe_index_stats().total_vector_count)
        path = cache_dir / f"{name}_{vector_count}.json"

        cached = _read_bm25_cache(path, name, vector_count)
        if cached is not None:
            logger.info("Loaded BM25 corpus from %s (%d docs)", path, len(cached))
            return cached

        documents = self._fetch_bm25_documents()
        if len(documents) != vector_count:
            # Stats can lag right after an upsert; don't pin an incomplete corpus.
            logger.warning(
                "Fetched %d docs but index reports %d vectors; not caching BM25 corpus",
                len(documents),
                vector_count,
            )
            return documents

        cache_dir.mkdir(parents=True, exist_ok=True)
        for stale in cache_dir.glob(f"{name}_*.json"):
            stale.unlink()
        payload = {
            "index": name,
            "vector_count": vector_count,
            "documents": [{"id": id_, "metadata": dict(meta)} for id_, meta in documents],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("Wrote BM25 corpus cache %s (%d docs)", path, len(documents))
        return documents

    def query_hybrid(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        pool = default_candidate_pool_size(top_k)
        dense_results = self.query(query_text, top_k=pool)
        sparse_results = self._keyword_index().search(query_text, top_k=pool)
        return reciprocal_rank_fusion([dense_results, sparse_results], top_k=top_k)

    def query_hybrid_reranked(
        self, query_text: str, top_k: int = 5, rerank_candidates: int = 20
    ) -> list[RetrievalResult]:
        if self.reranker is None:
            self.reranker = CohereReranker()

        candidates = self.query_hybrid(query_text, top_k=rerank_candidates)
        return self.reranker.rerank(query_text, candidates, top_k=top_k)

    def close(self) -> None:
        # Index clients hold their own connection pool separate from the
        # control-plane client, so both need closing to avoid unclosed
        # SSL sockets.
        if self._index is not None:
            self._index.close()
        self.client.close()


def _read_bm25_cache(
    path: Path, index_name: str, vector_count: int
) -> list[tuple[str, dict[str, Any]]] | None:
    """Cached documents, or None if the file is missing, unreadable, or was
    written for a different index/count."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["index"] != index_name or payload["vector_count"] != vector_count:
            return None
        return [(doc["id"], doc["metadata"]) for doc in payload["documents"]]
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Ignoring unreadable BM25 cache %s", path)
        return None
