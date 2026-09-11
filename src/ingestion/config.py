"""Configuration for the SEC 10-K ingestion pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompanyConfig:
    """A corpus target: which company, and which EDGAR form type counts as its
    annual report (foreign private issuers like Cameco file 40-F, not 10-K)."""

    name: str
    ticker: str
    sector: str
    form_types: tuple[str, ...] = ("10-K",)


COMPANIES: list[CompanyConfig] = [
    CompanyConfig(name="Constellation Energy Corporation", ticker="CEG", sector="utilities"),
    CompanyConfig(
        name="Cameco Corporation",
        ticker="CCJ",
        sector="nuclear_fuel",
        form_types=("40-F",),  # Canadian foreign private issuer, no 10-K
    ),
    CompanyConfig(name="NuScale Power Corporation", ticker="SMR", sector="nuclear_tech"),
    CompanyConfig(name="Centrus Energy Corp.", ticker="LEU", sector="nuclear_fuel"),
]

FISCAL_YEARS_BACK = 3

EDGAR_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

# SEC's fair-access policy requires a descriptive User-Agent with a contact
# method (see https://www.sec.gov/os/webmaster-faq#developers). Set this via
# env var rather than hardcoding a personal contact into source.
EDGAR_USER_AGENT_ENV_VAR = "EDGAR_USER_AGENT"
EDGAR_MIN_REQUEST_INTERVAL_SECONDS = 0.15  # keeps us well under SEC's 10 req/s cap


@dataclass(frozen=True)
class ChunkingConfig:
    encoding_name: str = "cl100k_base"  # matches text-embedding-3-small's tokenizer
    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 100


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100


@dataclass(frozen=True)
class IndexConfig:
    """Shared index shape so Pinecone and Weaviate stay comparable at query time."""

    name: str = "sec-10k-filings"
    dimension: int = 1536
    metric: str = "cosine"


@dataclass(frozen=True)
class PineconeConfig:
    cloud: str = field(default_factory=lambda: os.environ.get("PINECONE_CLOUD", "aws"))
    region: str = field(default_factory=lambda: os.environ.get("PINECONE_REGION", "us-east-1"))


@dataclass(frozen=True)
class WeaviateConfig:
    collection_name: str = "SEC10KFilings"
