"""Weaviate backend: HFresh index, cosine distance, matching `IndexConfig`.

We bring our own OpenAI embeddings (vectorizer=none) so both backends index
the exact same vectors and are comparable at query time. HFresh is the only
vector index type this cluster allows (HNSW is rejected with
CONFIG_NOT_ALLOWED); it only supports cosine/l2-squared distance, which lines
up with `IndexConfig.metric = "cosine"`.
"""

from __future__ import annotations

import logging
import os

import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.init import Auth
from weaviate.util import generate_uuid5

from src.ingestion.config import IndexConfig, WeaviateConfig
from src.ingestion.models import Chunk

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 100


class WeaviateStore:
    def __init__(
        self,
        index_config: IndexConfig,
        weaviate_config: WeaviateConfig,
        client: weaviate.WeaviateClient | None = None,
    ) -> None:
        self.index_config = index_config
        self.weaviate_config = weaviate_config
        self.client = client or weaviate.connect_to_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
        )

    def ensure_index(self) -> None:
        name = self.weaviate_config.collection_name
        if self.client.collections.exists(name):
            return

        logger.info("Creating Weaviate collection %s", name)
        self.client.collections.create(
            name=name,
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            vector_index_config=wvc.Configure.VectorIndex.hfresh(
                distance_metric=wvc.VectorDistances.COSINE,
            ),
            properties=[
                wvc.Property(name="company", data_type=wvc.DataType.TEXT),
                wvc.Property(name="ticker", data_type=wvc.DataType.TEXT),
                wvc.Property(name="cik", data_type=wvc.DataType.TEXT),
                wvc.Property(name="form_type", data_type=wvc.DataType.TEXT),
                wvc.Property(name="fiscal_year", data_type=wvc.DataType.INT),
                wvc.Property(name="filing_date", data_type=wvc.DataType.TEXT),
                wvc.Property(name="accession_number", data_type=wvc.DataType.TEXT),
                wvc.Property(name="source_url", data_type=wvc.DataType.TEXT),
                wvc.Property(name="chunk_index", data_type=wvc.DataType.INT),
                wvc.Property(name="total_chunks", data_type=wvc.DataType.INT),
                wvc.Property(name="text", data_type=wvc.DataType.TEXT),
            ],
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        collection = self.client.collections.get(self.weaviate_config.collection_name)
        with collection.batch.fixed_size(batch_size=_UPSERT_BATCH_SIZE) as batch:
            for chunk in chunks:
                batch.add_object(
                    properties=chunk.to_metadata_dict(),
                    vector=chunk.embedding,
                    uuid=generate_uuid5(chunk.chunk_id),
                )

    def close(self) -> None:
        self.client.close()
