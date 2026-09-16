"""Common result type returned by every `VectorStore.query`, so calling code
doesn't care whether the hit came from Pinecone or Weaviate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.ingestion.models import FilingMetadata


@dataclass(frozen=True)
class RetrievalResult:
    """One nearest-neighbor hit: chunk text + metadata + a similarity score
    (higher is more similar) comparable across backends."""

    id: str
    text: str
    score: float
    filing: FilingMetadata
    chunk_index: int
    total_chunks: int


def result_from_metadata(id_: str, score: float, metadata: Mapping[str, Any]) -> RetrievalResult:
    """Rebuild a `RetrievalResult` from the flat metadata dict produced by
    `Chunk.to_metadata_dict`, as stored by either backend."""
    filing = FilingMetadata(
        company=str(metadata["company"]),
        ticker=str(metadata["ticker"]),
        cik=str(metadata["cik"]),
        form_type=str(metadata["form_type"]),
        fiscal_year=int(metadata["fiscal_year"]),
        filing_date=str(metadata["filing_date"]),
        accession_number=str(metadata["accession_number"]),
        source_url=str(metadata["source_url"]),
    )
    return RetrievalResult(
        id=id_,
        text=str(metadata["text"]),
        score=score,
        filing=filing,
        chunk_index=int(metadata["chunk_index"]),
        total_chunks=int(metadata["total_chunks"]),
    )
