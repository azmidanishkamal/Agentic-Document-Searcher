"""Turn a raw EDGAR filing document (HTML, occasionally plain text) into clean text."""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# SEC filings are inline-XBRL XHTML; BS4's heuristic flags them as "XML parsed
# as HTML", but HTML parsing is exactly what we want here.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_WHITESPACE_RE = re.compile(r"[ \t ]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def html_to_text(raw_document: str) -> str:
    """Strip markup/scripts/styles and collapse whitespace, keeping paragraph breaks."""
    soup = BeautifulSoup(raw_document, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = _WHITESPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()
