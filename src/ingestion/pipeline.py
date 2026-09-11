"""Orchestrates the full ingestion pipeline: EDGAR -> parse -> chunk -> embed -> upsert."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.ingestion import edgar_client
from src.ingestion.chunking import chunk_text
from src.ingestion.config import (
    COMPANIES,
    FISCAL_YEARS_BACK,
    ChunkingConfig,
    CompanyConfig,
    EmbeddingConfig,
    IndexConfig,
    PineconeConfig,
    WeaviateConfig,
)
from src.ingestion.embeddings import OpenAIEmbedder
from src.ingestion.models import Chunk, FilingMetadata
from src.ingestion.parsing import html_to_text
from src.ingestion.vector_stores.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionSummary:
    filings_processed: int = 0
    chunks_created: int = 0
    chunks_upserted: int = 0


def _build_chunks(
    filing: edgar_client.FilingRef,
    company: CompanyConfig,
    chunking_config: ChunkingConfig,
) -> list[Chunk]:
    document_names = edgar_client.get_filing_documents(filing)
    text = "\n\n".join(
        html_to_text(edgar_client.download_document(filing, name)) for name in document_names
    )
    pieces = chunk_text(text, chunking_config)

    metadata = FilingMetadata(
        company=company.name,
        ticker=company.ticker,
        cik=filing.cik,
        form_type=filing.form_type,
        fiscal_year=filing.fiscal_year,
        filing_date=filing.filing_date,
        accession_number=filing.accession_number,
        source_url=filing.index_url,
    )

    return [
        Chunk(
            chunk_id=f"{company.ticker}_{filing.fiscal_year}_{filing.form_type}_{i:04d}",
            text=piece,
            chunk_index=i,
            total_chunks=len(pieces),
            filing=metadata,
        )
        for i, piece in enumerate(pieces)
    ]


def run_ingestion(
    companies: list[CompanyConfig] = COMPANIES,
    years_back: int = FISCAL_YEARS_BACK,
    chunking_config: ChunkingConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    stores: list[VectorStore] | None = None,
    dry_run: bool = False,
) -> IngestionSummary:
    """Fetch each company's recent annual filings, chunk + embed them, and upsert
    into every store in `stores`. In dry-run mode, skips embedding and upserting
    so the EDGAR fetch/parse/chunk steps can be verified at no cost."""
    chunking_config = chunking_config or ChunkingConfig()
    embedding_config = embedding_config or EmbeddingConfig()
    summary = IngestionSummary()
    embedder = None if dry_run else OpenAIEmbedder(embedding_config)

    if not dry_run:
        for store in stores or []:
            store.ensure_index()

    for company in companies:
        filings = edgar_client.find_company_filings(company, years_back)
        logger.info("%s: found %d filing(s)", company.ticker, len(filings))

        for filing in filings:
            chunks = _build_chunks(filing, company, chunking_config)
            summary.filings_processed += 1
            summary.chunks_created += len(chunks)
            logger.info(
                "%s FY%d %s: %d chunks", company.ticker, filing.fiscal_year, filing.form_type, len(chunks)
            )

            if dry_run:
                continue

            texts = [chunk.text for chunk in chunks]
            embeddings = embedder.embed_batch(texts)
            embedded_chunks = [
                Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    filing=chunk.filing,
                    embedding=embedding,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]

            for store in stores or []:
                store.upsert_chunks(embedded_chunks)
                summary.chunks_upserted += len(embedded_chunks)

    return summary


def default_stores() -> list[VectorStore]:
    from src.ingestion.vector_stores.pinecone_store import PineconeStore
    from src.ingestion.vector_stores.weaviate_store import WeaviateStore

    index_config = IndexConfig()
    return [
        PineconeStore(index_config, PineconeConfig()),
        WeaviateStore(index_config, WeaviateConfig()),
    ]
