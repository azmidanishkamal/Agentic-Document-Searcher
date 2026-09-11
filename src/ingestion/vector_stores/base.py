"""Common interface both vector store backends implement, so retrieval code
can swap between them and comparisons stay apples-to-apples."""

from __future__ import annotations

from typing import Protocol

from src.ingestion.models import Chunk


class VectorStore(Protocol):
    def ensure_index(self) -> None:
        """Create the index/collection if it doesn't already exist."""
        ...

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Write embedded chunks (with metadata) into the store."""
        ...
