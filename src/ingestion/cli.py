"""CLI entry point: `python -m src.ingestion.cli [--dry-run] [--years-back N] [--stores pinecone,weaviate]`"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from src.ingestion.config import COMPANIES, FISCAL_YEARS_BACK
from src.ingestion.pipeline import run_ingestion

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest SEC 10-K/40-F filings into vector stores.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch, parse, and chunk filings without embedding or upserting (no API cost).",
    )
    parser.add_argument("--years-back", type=int, default=FISCAL_YEARS_BACK)
    parser.add_argument(
        "--stores",
        default="pinecone,weaviate",
        help="Comma-separated subset of: pinecone, weaviate",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    args = _parse_args()

    stores = None
    if not args.dry_run:
        from src.ingestion.config import IndexConfig, PineconeConfig, WeaviateConfig
        from src.ingestion.vector_stores.pinecone_store import PineconeStore
        from src.ingestion.vector_stores.weaviate_store import WeaviateStore

        index_config = IndexConfig()
        backend_map = {
            "pinecone": lambda: PineconeStore(index_config, PineconeConfig()),
            "weaviate": lambda: WeaviateStore(index_config, WeaviateConfig()),
        }
        requested = [name.strip() for name in args.stores.split(",") if name.strip()]
        stores = [backend_map[name]() for name in requested]

    summary = run_ingestion(
        companies=COMPANIES,
        years_back=args.years_back,
        stores=stores,
        dry_run=args.dry_run,
    )

    logger.info(
        "Done: %d filings processed, %d chunks created, %d chunks upserted",
        summary.filings_processed,
        summary.chunks_created,
        summary.chunks_upserted,
    )


if __name__ == "__main__":
    main()
