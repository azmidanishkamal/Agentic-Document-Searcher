"""Shared query-embedding helper, so both backends embed query text with the
exact same model/config used at ingestion time."""

from __future__ import annotations

from src.ingestion.embeddings import OpenAIEmbedder


def embed_query(embedder: OpenAIEmbedder, query_text: str) -> list[float]:
    return embedder.embed_batch([query_text])[0]
