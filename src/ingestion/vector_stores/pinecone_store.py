"""Pinecone backend: serverless index, cosine metric, matching `IndexConfig`."""

from __future__ import annotations

import logging
import time

from pinecone import Pinecone, ServerlessSpec

from src.ingestion.config import IndexConfig, PineconeConfig
from src.ingestion.models import Chunk

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 100


class PineconeStore:
    def __init__(
        self,
        index_config: IndexConfig,
        pinecone_config: PineconeConfig,
        client: Pinecone | None = None,
    ) -> None:
        self.index_config = index_config
        self.pinecone_config = pinecone_config
        self.client = client or Pinecone()
        self._index = None

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
