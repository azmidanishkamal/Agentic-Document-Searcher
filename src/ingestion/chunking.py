"""Recursive character chunking, sized in tokens to match the embedding model."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.config import ChunkingConfig


def chunk_text(text: str, config: ChunkingConfig) -> list[str]:
    """Split filing text into ~chunk_size_tokens pieces with token-based overlap."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.encoding_name,
        chunk_size=config.chunk_size_tokens,
        chunk_overlap=config.chunk_overlap_tokens,
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]
