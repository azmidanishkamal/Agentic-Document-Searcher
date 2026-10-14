from dataclasses import replace

from src.agent import prompts
from src.ingestion.models import FilingMetadata
from src.retrieval.results import RetrievalResult

_FILING = FilingMetadata(
    company="Cameco Corporation",
    ticker="CCJ",
    cik="0000002012",
    form_type="40-F",
    fiscal_year=2023,
    filing_date="2024-03-15",
    accession_number="0000002012-24-000001",
    source_url="https://www.sec.gov/Archives/edgar/data/2012/000000201224000001",
)


def _chunk(id_: str, filing: FilingMetadata) -> RetrievalResult:
    return RetrievalResult(
        id=id_,
        text="Net earnings were $361 million.",
        score=0.9,
        filing=filing,
        chunk_index=0,
        total_chunks=1,
    )


def test_excerpt_header_includes_reporting_currency() -> None:
    centrus = replace(_FILING, company="Centrus Energy Corp.", ticker="LEU")

    text = prompts.format_chunks([_chunk("c1", _FILING), _chunk("c2", centrus)])

    assert (
        "[c1] Cameco Corporation (CCJ) 40-F, fiscal year 2023, reporting currency CAD"
        in text
    )
    assert (
        "[c2] Centrus Energy Corp. (LEU) 40-F, fiscal year 2023, reporting currency USD"
        in text
    )


def test_excerpt_header_marks_unmapped_ticker_currency_unknown() -> None:
    other = replace(_FILING, company="Other Corp.", ticker="ZZZ")

    text = prompts.format_chunks([_chunk("c1", other)])

    assert "reporting currency unknown" in text
