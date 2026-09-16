from types import SimpleNamespace
from unittest.mock import MagicMock

from src.ingestion.config import IndexConfig, WeaviateConfig
from src.ingestion.vector_stores.weaviate_store import WeaviateStore

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


def test_query_embeds_text_and_maps_objects_to_results() -> None:
    obj = SimpleNamespace(
        uuid="11111111-1111-1111-1111-111111111111",
        properties=_METADATA,
        metadata=SimpleNamespace(distance=0.2),
    )
    fake_collection = MagicMock()
    fake_collection.query.near_vector.return_value = SimpleNamespace(objects=[obj])
    fake_client = MagicMock()
    fake_client.collections.get.return_value = fake_collection
    embedder = FakeEmbedder([0.4, 0.5])

    store = WeaviateStore(
        index_config=IndexConfig(),
        weaviate_config=WeaviateConfig(),
        client=fake_client,
        embedder=embedder,
    )

    results = store.query("What drove revenue growth?", top_k=2)

    assert embedder.embedded_texts == ["What drove revenue growth?"]
    fake_client.collections.get.assert_called_once_with(WeaviateConfig().collection_name)
    _, kwargs = fake_collection.query.near_vector.call_args
    assert kwargs["near_vector"] == [0.4, 0.5]
    assert kwargs["limit"] == 2

    assert len(results) == 1
    result = results[0]
    assert result.id == "11111111-1111-1111-1111-111111111111"
    assert result.score == 0.8  # 1 - cosine distance
    assert result.text == "Revenue increased due to higher electricity demand."
    assert result.filing.company == "Constellation Energy Corporation"
    assert result.chunk_index == 0
    assert result.total_chunks == 3


def test_query_defaults_top_k_to_five() -> None:
    fake_collection = MagicMock()
    fake_collection.query.near_vector.return_value = SimpleNamespace(objects=[])
    fake_client = MagicMock()
    fake_client.collections.get.return_value = fake_collection

    store = WeaviateStore(
        index_config=IndexConfig(),
        weaviate_config=WeaviateConfig(),
        client=fake_client,
        embedder=FakeEmbedder([0.0]),
    )

    store.query("anything")

    assert fake_collection.query.near_vector.call_args.kwargs["limit"] == 5
