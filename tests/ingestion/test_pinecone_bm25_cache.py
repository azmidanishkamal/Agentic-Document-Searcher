import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.ingestion.config import IndexConfig, PineconeConfig
from src.ingestion.vector_stores.pinecone_store import PineconeStore

_BASE = {
    "company": "Cameco Corporation",
    "ticker": "CCJ",
    "cik": "0000002012",
    "form_type": "40-F",
    "fiscal_year": 2024,
    "filing_date": "2025-03-14",
    "accession_number": "0000002012-25-000001",
    "source_url": "https://www.sec.gov/Archives/edgar/data/2012/000000201225000001",
    "chunk_index": 0,
    "total_chunks": 3,
}
# Three docs so BM25 idf for a term in one doc stays positive.
_DOCS = {
    "c1": {**_BASE, "text": "Mine suspension in Niger disrupted uranium supply."},
    "c2": {**_BASE, "text": "Revenue increased due to higher electricity demand."},
    "c3": {**_BASE, "text": "The company recorded a net loss for the year."},
}
_INDEX_NAME = IndexConfig().name


def _fake_index(docs: dict[str, dict], vector_count: int | None = None) -> MagicMock:
    index = MagicMock()
    index.describe_index_stats.return_value = SimpleNamespace(
        total_vector_count=len(docs) if vector_count is None else vector_count
    )
    index.list.side_effect = lambda: iter(
        [SimpleNamespace(vectors=[SimpleNamespace(id=id_) for id_ in docs])]
    )
    index.fetch.side_effect = lambda ids: SimpleNamespace(
        vectors={
            id_: SimpleNamespace(metadata=docs[id_], values=[0.1] * 4) for id_ in ids
        }
    )
    return index


def _store(index: MagicMock, cache_dir: Path) -> PineconeStore:
    store = PineconeStore(
        index_config=IndexConfig(),
        pinecone_config=PineconeConfig(bm25_cache_dir=cache_dir),
        client=MagicMock(),
    )
    store._index = index
    return store


def _search_ids(store: PineconeStore) -> list[str]:
    return [r.id for r in store._keyword_index().search("uranium Niger", top_k=3)]


def test_cache_miss_scans_index_and_writes_cache_without_embeddings(
    tmp_path: Path,
) -> None:
    index = _fake_index(_DOCS)

    ids = _search_ids(_store(index, tmp_path))

    assert ids == ["c1"]
    index.list.assert_called_once()
    path = tmp_path / f"{_INDEX_NAME}_3.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["index"] == _INDEX_NAME and payload["vector_count"] == 3
    assert [d["id"] for d in payload["documents"]] == ["c1", "c2", "c3"]
    assert payload["documents"][0]["metadata"] == _DOCS["c1"]
    assert "values" not in path.read_text(encoding="utf-8")


def test_cache_hit_skips_list_and_fetch(tmp_path: Path) -> None:
    _search_ids(_store(_fake_index(_DOCS), tmp_path))  # populate cache
    index = _fake_index(_DOCS)

    ids = _search_ids(_store(index, tmp_path))

    assert ids == ["c1"]
    index.list.assert_not_called()
    index.fetch.assert_not_called()


def test_stale_cache_after_reingest_is_rebuilt_and_replaced(tmp_path: Path) -> None:
    _search_ids(_store(_fake_index(_DOCS), tmp_path))  # cache for 3 vectors
    grown = {**_DOCS, "c4": {**_BASE, "text": "Kazatomprom cut uranium output."}}
    index = _fake_index(grown)

    store = _store(index, tmp_path)
    results = store._keyword_index().search("Kazatomprom", top_k=3)

    assert [r.id for r in results] == ["c4"]  # new doc only exists in rebuilt corpus
    index.list.assert_called_once()
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{_INDEX_NAME}_4.json"]


def test_cache_with_mismatched_key_inside_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / f"{_INDEX_NAME}_3.json"
    path.write_text(
        json.dumps({"index": "other-index", "vector_count": 3, "documents": []}),
        encoding="utf-8",
    )
    index = _fake_index(_DOCS)

    assert _search_ids(_store(index, tmp_path)) == ["c1"]
    index.list.assert_called_once()


def test_corrupt_cache_is_treated_as_miss(tmp_path: Path) -> None:
    (tmp_path / f"{_INDEX_NAME}_3.json").write_text("{not json", encoding="utf-8")
    index = _fake_index(_DOCS)

    assert _search_ids(_store(index, tmp_path)) == ["c1"]
    index.list.assert_called_once()


def test_count_mismatch_is_not_cached(tmp_path: Path) -> None:
    # Stats report 5 vectors but only 3 are listable (e.g. stats lag).
    index = _fake_index(_DOCS, vector_count=5)

    assert _search_ids(_store(index, tmp_path)) == ["c1"]
    assert list(tmp_path.iterdir()) == []


def test_cache_disabled_never_touches_disk_or_stats(tmp_path: Path) -> None:
    index = _fake_index(_DOCS)
    store = PineconeStore(
        index_config=IndexConfig(),
        pinecone_config=PineconeConfig(bm25_cache_dir=None),
        client=MagicMock(),
    )
    store._index = index

    assert _search_ids(store) == ["c1"]
    index.describe_index_stats.assert_not_called()
