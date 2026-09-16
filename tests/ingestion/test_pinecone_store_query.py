from types import SimpleNamespace
from unittest.mock import MagicMock

from src.ingestion.config import IndexConfig, PineconeConfig
from src.ingestion.vector_stores.pinecone_store import PineconeStore

_METADATA = {
    "company": "Constellation Energy Corporation",
    "ticker": "CEG",
    "cik": "0001868275",
    "form_type": "10-K",
    "fiscal_year": 2023,
    "filing_date": "2024-02-16",
    "accession_number": "0001868275-24-000010",
    "source_url": "https://www.sec.gov/Archives/edgar/data/1868275/000186827524000010",
    "chunk_index": 0,
    "total_chunks": 3,
    "text": "Revenue increased due to higher electricity demand.",
}


class FakeEmbedder:
    """Stand-in for OpenAIEmbedder that never hits the network."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.embedded_texts: list[str] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [self.vector for _ in texts]


def test_query_embeds_text_and_maps_matches_to_results() -> None:
    match = SimpleNamespace(id="CEG_2023_10-K_0000", score=0.87, metadata=_METADATA)
    fake_index = MagicMock()
    fake_index.query.return_value = SimpleNamespace(matches=[match])
    embedder = FakeEmbedder([0.1, 0.2, 0.3])

    store = PineconeStore(
        index_config=IndexConfig(),
        pinecone_config=PineconeConfig(),
        client=MagicMock(),
        embedder=embedder,
    )
    store._index = fake_index  # skip ensure_index/create_index round trip

    results = store.query("What drove revenue growth?", top_k=3)

    assert embedder.embedded_texts == ["What drove revenue growth?"]
    fake_index.query.assert_called_once_with(vector=[0.1, 0.2, 0.3], top_k=3, include_metadata=True)

    assert len(results) == 1
    result = results[0]
    assert result.id == "CEG_2023_10-K_0000"
    assert result.score == 0.87
    assert result.text == "Revenue increased due to higher electricity demand."
    assert result.filing.ticker == "CEG"
    assert result.chunk_index == 0
    assert result.total_chunks == 3


def test_query_defaults_top_k_to_five() -> None:
    fake_index = MagicMock()
    fake_index.query.return_value = SimpleNamespace(matches=[])
    store = PineconeStore(
        index_config=IndexConfig(),
        pinecone_config=PineconeConfig(),
        client=MagicMock(),
        embedder=FakeEmbedder([0.0]),
    )
    store._index = fake_index

    store.query("anything")

    assert fake_index.query.call_args.kwargs["top_k"] == 5
