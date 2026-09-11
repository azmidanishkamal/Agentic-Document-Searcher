"""Client for locating and downloading 10-K/40-F filings from SEC EDGAR.

Uses the public ticker-to-CIK mapping plus the submissions API rather than
hardcoded CIK numbers, so company identity is resolved at run time:
https://www.sec.gov/files/company_tickers.json
https://data.sec.gov/submissions/CIK##########.json
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date

import requests

from src.ingestion.config import (
    EDGAR_ARCHIVES_BASE_URL,
    EDGAR_MIN_REQUEST_INTERVAL_SECONDS,
    EDGAR_SUBMISSIONS_URL,
    EDGAR_TICKER_MAP_URL,
    EDGAR_USER_AGENT_ENV_VAR,
    CompanyConfig,
)

logger = logging.getLogger(__name__)

_last_request_time = 0.0

# Above this size, a supplementary exhibit is treated as substantive report
# content worth ingesting (e.g. a 40-F's MD&A/financial-statement exhibits),
# not boilerplate like an officer certification or auditor consent letter.
_SUPPLEMENTARY_EXHIBIT_SIZE_THRESHOLD_BYTES = 200_000
_INTERACTIVE_DATA_VIEWER_RE = re.compile(r"^R\d+\.htm$", re.IGNORECASE)


class EdgarClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class FilingRef:
    """A single filing located in EDGAR, not yet downloaded."""

    cik: str
    accession_number: str
    form_type: str
    filing_date: str
    fiscal_year: int
    primary_document: str

    @property
    def accession_no_dashes(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def document_url(self) -> str:
        return f"{EDGAR_ARCHIVES_BASE_URL}/{int(self.cik)}/{self.accession_no_dashes}/{self.primary_document}"

    @property
    def index_url(self) -> str:
        """Human-readable filing index page — used as the citable source URL,
        since substantive content can span more than one document (see
        `get_filing_documents`)."""
        return (
            f"{EDGAR_ARCHIVES_BASE_URL}/{int(self.cik)}/{self.accession_no_dashes}/"
            f"{self.accession_number}-index.htm"
        )


def _get_user_agent() -> str:
    user_agent = os.environ.get(EDGAR_USER_AGENT_ENV_VAR)
    if not user_agent:
        raise EdgarClientError(
            f"Set the {EDGAR_USER_AGENT_ENV_VAR} env var to identify requests to SEC EDGAR, "
            'e.g. "Agentic-Document-Searcher research@yourdomain.com" '
            "(SEC requires a descriptive User-Agent with contact info: "
            "https://www.sec.gov/os/webmaster-faq#developers)."
        )
    return user_agent


def _rate_limited_get(url: str) -> requests.Response:
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < EDGAR_MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(EDGAR_MIN_REQUEST_INTERVAL_SECONDS - elapsed)

    headers = {"User-Agent": _get_user_agent(), "Accept-Encoding": "gzip, deflate"}
    response = requests.get(url, headers=headers, timeout=30)
    _last_request_time = time.monotonic()
    response.raise_for_status()
    return response


def resolve_cik(ticker: str) -> str:
    """Look up a company's zero-padded 10-digit CIK from its ticker symbol."""
    response = _rate_limited_get(EDGAR_TICKER_MAP_URL)
    ticker_upper = ticker.upper()
    for entry in response.json().values():
        if entry["ticker"].upper() == ticker_upper:
            return f"{entry['cik_str']:010d}"
    raise EdgarClientError(f"No CIK found for ticker {ticker!r}")


def get_submissions(cik: str) -> dict:
    """Fetch the full submissions history JSON for a CIK."""
    url = EDGAR_SUBMISSIONS_URL.format(cik=int(cik))
    return _rate_limited_get(url).json()


def list_annual_filings(
    cik: str, form_types: tuple[str, ...], years_back: int
) -> list[FilingRef]:
    """Return annual-report filings (10-K/40-F) from the last `years_back` fiscal years."""
    submissions = get_submissions(cik)
    recent = submissions["filings"]["recent"]
    cutoff_year = date.today().year - years_back

    filings: list[FilingRef] = []
    for i, form in enumerate(recent["form"]):
        if form not in form_types:
            continue
        report_date = recent["reportDate"][i]
        if not report_date:
            continue
        fiscal_year = int(report_date[:4])
        if fiscal_year < cutoff_year:
            continue
        primary_document = recent["primaryDocument"][i]
        if not primary_document:
            continue
        filings.append(
            FilingRef(
                cik=cik,
                accession_number=recent["accessionNumber"][i],
                form_type=form,
                filing_date=recent["filingDate"][i],
                fiscal_year=fiscal_year,
                primary_document=primary_document,
            )
        )

    filings.sort(key=lambda f: f.fiscal_year, reverse=True)
    return filings


def get_filing_documents(filing: FilingRef) -> list[str]:
    """Return every substantive HTML document belonging to a filing.

    A US 10-K's primary document already contains the full report, but a
    foreign private issuer's 40-F is a short cover form whose actual
    MD&A/financial-statement content is filed as separate EX-99.x exhibits.
    We list the filing's index and pull in any other large HTML document
    alongside the primary one, skipping small boilerplate exhibits
    (certifications, consents) and XBRL viewer fragments (R###.htm).
    """
    index_url = f"{EDGAR_ARCHIVES_BASE_URL}/{int(filing.cik)}/{filing.accession_no_dashes}/index.json"
    items = _rate_limited_get(index_url).json()["directory"]["item"]

    documents = [filing.primary_document]
    for item in items:
        name = item["name"]
        if name == filing.primary_document:
            continue
        if not name.lower().endswith((".htm", ".html")):
            continue
        if _INTERACTIVE_DATA_VIEWER_RE.match(name):
            continue
        size = int(item.get("size") or 0)
        if size < _SUPPLEMENTARY_EXHIBIT_SIZE_THRESHOLD_BYTES:
            continue
        documents.append(name)
    return documents


def download_document(filing: FilingRef, document_name: str) -> str:
    """Download one raw document (usually HTML) belonging to a filing."""
    url = f"{EDGAR_ARCHIVES_BASE_URL}/{int(filing.cik)}/{filing.accession_no_dashes}/{document_name}"
    logger.info("Downloading %s", url)
    return _rate_limited_get(url).text


def find_company_filings(company: CompanyConfig, years_back: int) -> list[FilingRef]:
    cik = resolve_cik(company.ticker)
    return list_annual_filings(cik, company.form_types, years_back)
