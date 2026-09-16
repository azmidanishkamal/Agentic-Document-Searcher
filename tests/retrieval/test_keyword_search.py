from src.retrieval.keyword_search import BM25Index

_BASE_METADATA = {
    "company": "Cameco Corporation",
    "ticker": "CCJ",
    "cik": "0000002012",
    "form_type": "40-F",
    "fiscal_year": 2025,
    "filing_date": "2025-03-01",
    "accession_number": "0000002012-25-000001",
    "source_url": "https://example.com/ccj",
    "chunk_index": 0,
    "total_chunks": 1,
}

_FILLER_TEXTS = [
    "small modular reactor deployment timelines depend on regulatory approval",
    "electricity demand increased due to data center growth",
    "centrifuge enrichment capacity expansion continues",
    "natural gas prices remained volatile during the quarter",
    "the company recorded a net loss for the fiscal year",
]

# Same query terms, but longer documents -- BM25's length normalization means
# the shortest (most tightly on-topic) document should score highest.
_RELEVANT_TEXTS = [
    "uranium supply risk includes mine suspensions in kazakhstan",
    "uranium supply risk includes mine suspensions in kazakhstan and niger",
    "uranium supply risk includes mine suspensions in kazakhstan and niger and canada",
]


def _metadata_docs(texts: list[str]) -> list[tuple[str, dict]]:
    return [(f"doc{i}", {**_BASE_METADATA, "text": text}) for i, text in enumerate(texts)]


def test_bm25_index_ranks_keyword_matches_above_unrelated_text() -> None:
    documents = _metadata_docs(_FILLER_TEXTS) + [
        (f"relevant{i}", {**_BASE_METADATA, "text": text}) for i, text in enumerate(_RELEVANT_TEXTS)
    ]
    index = BM25Index.from_metadata(documents)

    results = index.search("uranium supply risk", top_k=10)

    # Filler docs share no query terms, so BM25 scores them 0 and search()
    # drops them entirely -- only the uranium-relevant docs come back.
    assert {r.id for r in results} == {"relevant0", "relevant1", "relevant2"}
    assert all(r.filing.ticker == "CCJ" for r in results)


def test_bm25_index_search_on_empty_index_returns_empty_list() -> None:
    index = BM25Index.from_metadata([])

    assert index.search("anything", top_k=5) == []


def test_bm25_index_respects_top_k_and_ranks_by_relevance() -> None:
    documents = _metadata_docs(_FILLER_TEXTS) + [
        (f"relevant{i}", {**_BASE_METADATA, "text": text}) for i, text in enumerate(_RELEVANT_TEXTS)
    ]
    index = BM25Index.from_metadata(documents)

    results = index.search("uranium supply risk", top_k=2)

    # Shorter, more tightly-matching docs outrank longer ones with the same
    # terms diluted by extra content.
    assert [r.id for r in results] == ["relevant0", "relevant1"]
