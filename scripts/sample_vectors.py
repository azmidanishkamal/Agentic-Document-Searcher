"""Quick sanity check: fetch 3 sample vectors from each backend and print
their metadata + text, to eyeball that both indexes hold the same data.

Usage: python -m scripts.sample_vectors
"""

from __future__ import annotations

import random

from dotenv import load_dotenv
from pinecone import Pinecone

from src.ingestion.config import IndexConfig, WeaviateConfig
from src.ingestion.vector_stores.weaviate_store import WeaviateStore

_SAMPLE_SIZE = 3


def _print_chunk(source: str, chunk_id: str, metadata: dict) -> None:
    text = metadata.pop("text", "")
    print(f"\n--- {source}: {chunk_id} ---")
    print("metadata:", metadata)
    print("text:", text[:300] + ("..." if len(text) > 300 else ""))


def sample_pinecone(index_config: IndexConfig) -> None:
    client = Pinecone()
    try:
        index = client.Index(index_config.name)
        # A zero vector has undefined cosine similarity, so Pinecone silently
        # returns zero matches for it -- use a random vector to sample instead.
        probe_vector = [random.random() for _ in range(index_config.dimension)]
        result = index.query(
            vector=probe_vector,
            top_k=_SAMPLE_SIZE,
            include_metadata=True,
            include_values=False,
        )
        for match in result["matches"]:
            _print_chunk("Pinecone", match["id"], dict(match["metadata"]))
    finally:
        index.close()
        client.close()


def sample_weaviate(index_config: IndexConfig, weaviate_config: WeaviateConfig) -> None:
    store = WeaviateStore(index_config, weaviate_config)
    try:
        collection = store.client.collections.get(weaviate_config.collection_name)
        for obj in collection.query.fetch_objects(limit=_SAMPLE_SIZE).objects:
            _print_chunk("Weaviate", str(obj.uuid), dict(obj.properties))
    finally:
        store.close()


def main() -> None:
    load_dotenv()
    index_config = IndexConfig()
    sample_pinecone(index_config)
    sample_weaviate(index_config, WeaviateConfig())


if __name__ == "__main__":
    main()
