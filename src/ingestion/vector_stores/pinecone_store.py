"""Pinecone backend: serverless index, cosine metric, matching `IndexConfig`."""

from __future__ import annotations

import logging
import time

from pinecone import Pinecone, ServerlessSpec

from src.ingestion.config import EmbeddingConfig, IndexConfig, PineconeConfig
from src.ingestion.embeddings import OpenAIEmbedder
from src.ingestion.models import Chunk
from src.retrieval.embedding import embed_query
from src.retrieval.fusion import default_candidate_pool_size, reciprocal_rank_fusion
from src.retrieval.keyword_search import BM25Index
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
    ) -> None:
        self.index_config = index_config
        self.pinecone_config = pinecone_config
        self.client = client or Pinecone()
        self._index = None
        # Lazy default so constructing a store never requires an OpenAI API
        # key unless `query` is actually called.
        self.embedder = embedder
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
            documents: list[tuple[str, dict]] = []
            for page in self._index.list():
                ids = [item.id for item in page.vectors]
                fetched = self._index.fetch(ids=ids)
                documents.extend((id_, vector.metadata) for id_, vector in fetched.vectors.items())
            self._bm25_index = BM25Index.from_metadata(documents)
        return self._bm25_index

    def query_hybrid(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        pool = default_candidate_pool_size(top_k)
        dense_results = self.query(query_text, top_k=pool)
        sparse_results = self._keyword_index().search(query_text, top_k=pool)
        return reciprocal_rank_fusion([dense_results, sparse_results], top_k=top_k)

    def close(self) -> None:
        # Index clients hold their own connection pool separate from the
        # control-plane client, so both need closing to avoid unclosed
        # SSL sockets.
        if self._index is not None:
            self._index.close()
        self.client.close()
