"""Shared data models for filings and chunks moving through the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FilingMetadata:
    """Metadata about one downloaded filing, kept alongside every chunk cut from it."""

    company: str
    ticker: str
    cik: str
    form_type: str
    fiscal_year: int
    filing_date: str  # ISO date, e.g. "2024-02-16"
    accession_number: str
    source_url: str


@dataclass(frozen=True)
class Chunk:
    """One chunk of filing text, ready to embed and upsert."""

    chunk_id: str
    text: str
    chunk_index: int
    total_chunks: int
    filing: FilingMetadata
    embedding: list[float] | None = None

    def to_metadata_dict(self) -> dict[str, str | int | float]:
        """Flatten filing metadata + chunk position into a dict vector stores can index on."""
        flat: dict[str, str | int | float] = asdict(self.filing)
        flat["chunk_index"] = self.chunk_index
        flat["total_chunks"] = self.total_chunks
        flat["text"] = self.text
        return flat
