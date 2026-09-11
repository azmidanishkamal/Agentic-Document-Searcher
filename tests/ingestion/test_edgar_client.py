from datetime import date

import pytest

from src.ingestion.edgar_client import (
    EdgarClientError,
    FilingRef,
    get_filing_documents,
    list_annual_filings,
)


def test_filing_ref_document_url() -> None:
    filing = FilingRef(
        cik="0001868275",
        accession_number="0001868275-24-000010",
        form_type="10-K",
        filing_date="2024-02-16",
        fiscal_year=2023,
        primary_document="ceg-20231231.htm",
    )
    assert filing.document_url == (
        "https://www.sec.gov/Archives/edgar/data/1868275/000186827524000010/ceg-20231231.htm"
    )


def test_list_annual_filings_filters_by_form_and_year(monkeypatch: pytest.MonkeyPatch) -> None:
    current_year = date.today().year
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "10-K"],
                "reportDate": [f"{current_year - 1}-12-31", "", f"{current_year - 10}-12-31"],
                "filingDate": [f"{current_year}-02-01", f"{current_year}-01-15", f"{current_year - 10}-02-01"],
                "accessionNumber": ["0001-24-000001", "0001-24-000002", "0001-14-000003"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
            }
        }
    }
    monkeypatch.setattr(
        "src.ingestion.edgar_client.get_submissions", lambda cik: submissions
    )

    filings = list_annual_filings("0000000000", ("10-K",), years_back=3)

    assert len(filings) == 1
    assert filings[0].fiscal_year == current_year - 1
    assert filings[0].form_type == "10-K"


def test_get_filing_documents_includes_large_exhibits_and_skips_boilerplate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filing = FilingRef(
        cik="0001009001",
        accession_number="0001193125-26-116229",
        form_type="40-F",
        filing_date="2026-03-01",
        fiscal_year=2025,
        primary_document="d34605d40f.htm",
    )
    index_response = {
        "directory": {
            "item": [
                {"name": "d34605d40f.htm", "size": "609286"},
                {"name": "d34605dex991.htm", "size": "1797756"},  # AIF, large exhibit
                {"name": "d34605dex994.htm", "size": "5551"},  # small boilerplate exhibit
                {"name": "R1.htm", "size": "500000"},  # XBRL viewer fragment
                {"name": "d34605d40f_htm.xml", "size": "3235354"},  # not html
            ]
        }
    }
    monkeypatch.setattr(
        "src.ingestion.edgar_client._rate_limited_get",
        lambda url: type("Resp", (), {"json": lambda self: index_response})(),
    )

    documents = get_filing_documents(filing)

    assert documents == ["d34605d40f.htm", "d34605dex991.htm"]


def test_get_user_agent_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    from src.ingestion import edgar_client

    with pytest.raises(EdgarClientError):
        edgar_client._get_user_agent()
