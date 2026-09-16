from src.ingestion.models import FilingMetadata
from src.retrieval.results import RetrievalResult, result_from_metadata

_METADATA = {
    "company": "Constellation Energy Corporation",
    "ticker": "CEG",
    "cik": "0001868275",
    "form_type": "10-K",
    "fiscal_year": 2023,
    "filing_date": "2024-02-16",
    "accession_number": "0001868275-24-000010",
    "source_url": "https://www.sec.gov/Archives/edgar/data/1868275/000186827524000010",
    "chunk_index": 4,
    "total_chunks": 12,
    "text": "Revenue increased due to higher electricity demand.",
}


def test_result_from_metadata_builds_filing_and_result() -> None:
    result = result_from_metadata(id_="CEG_2023_10-K_0004", score=0.87, metadata=_METADATA)

    assert result == RetrievalResult(
        id="CEG_2023_10-K_0004",
        text="Revenue increased due to higher electricity demand.",
        score=0.87,
        filing=FilingMetadata(
            company="Constellation Energy Corporation",
            ticker="CEG",
            cik="0001868275",
            form_type="10-K",
            fiscal_year=2023,
            filing_date="2024-02-16",
            accession_number="0001868275-24-000010",
            source_url="https://www.sec.gov/Archives/edgar/data/1868275/000186827524000010",
        ),
        chunk_index=4,
        total_chunks=12,
    )


def test_result_from_metadata_coerces_numeric_fields() -> None:
    # Some backends round-trip numbers as floats/strings; coercion should
    # still land on the right int types.
    metadata = {**_METADATA, "fiscal_year": 2023.0, "chunk_index": "4", "total_chunks": 12.0}

    result = result_from_metadata(id_="x", score=0.5, metadata=metadata)

    assert result.filing.fiscal_year == 2023
    assert result.chunk_index == 4
    assert result.total_chunks == 12
