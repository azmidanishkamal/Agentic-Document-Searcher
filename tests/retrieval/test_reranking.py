from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.ingestion.models import FilingMetadata
from src.retrieval.reranking import DEFAULT_COHERE_RERANK_MODEL, CohereReranker
from src.retrieval.results import RetrievalResult

_FILING = FilingMetadata(
    company="Cameco Corporation",
    ticker="CCJ",
    cik="0000002012",
    form_type="40-F",
    fiscal_year=2025,
    filing_date="2025-03-01",
    accession_number="0000002012-25-000001",
    source_url="https://example.com/filing",
)


def _result(id_: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        id=id_, text=text, score=score, filing=_FILING, chunk_index=0, total_chunks=1
    )


def test_rerank_reorders_candidates_by_cohere_relevance_score() -> None:
    candidates = [
        _result("a", "irrelevant filler text", score=0.9),
        _result("b", "uranium supply risk in Niger", score=0.5),
    ]
    fake_client = MagicMock()
    fake_client.rerank.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(index=1, relevance_score=0.95),
            SimpleNamespace(index=0, relevance_score=0.1),
        ]
    )
    reranker = CohereReranker(client=fake_client)

    reranked = reranker.rerank("mine suspension in Niger", candidates, top_k=2)

    assert [r.id for r in reranked] == ["b", "a"]
    assert reranked[0].score == 0.95
    assert reranked[1].score == 0.1
    fake_client.rerank.assert_called_once_with(
        model=DEFAULT_COHERE_RERANK_MODEL,
        query="mine suspension in Niger",
        documents=["irrelevant filler text", "uranium supply risk in Niger"],
        top_n=2,
    )


def test_rerank_truncates_to_top_k() -> None:
    candidates = [_result("a", "text a", 0.1), _result("b", "text b", 0.2)]
    fake_client = MagicMock()
    fake_client.rerank.return_value = SimpleNamespace(
        results=[SimpleNamespace(index=1, relevance_score=0.9)]
    )
    reranker = CohereReranker(client=fake_client)

    reranked = reranker.rerank("query", candidates, top_k=1)

    assert [r.id for r in reranked] == ["b"]
    assert fake_client.rerank.call_args.kwargs["top_n"] == 1


def test_rerank_returns_empty_without_calling_client_when_no_candidates() -> None:
    fake_client = MagicMock()
    reranker = CohereReranker(client=fake_client)

    reranked = reranker.rerank("query", [], top_k=5)

    assert reranked == []
    fake_client.rerank.assert_not_called()


def test_rerank_preserves_other_result_fields() -> None:
    candidates = [_result("a", "text a", 0.1)]
    fake_client = MagicMock()
    fake_client.rerank.return_value = SimpleNamespace(
        results=[SimpleNamespace(index=0, relevance_score=0.42)]
    )
    reranker = CohereReranker(client=fake_client)

    reranked = reranker.rerank("query", candidates, top_k=1)

    assert reranked[0].text == "text a"
    assert reranked[0].filing == _FILING
    assert reranked[0].chunk_index == 0
    assert reranked[0].total_chunks == 1


class _RateLimited(Exception):
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        super().__init__("429")
        self.status_code = 429
        self.headers = headers or {}


def _ok_response() -> SimpleNamespace:
    return SimpleNamespace(results=[SimpleNamespace(index=0, relevance_score=0.9)])


def test_rerank_retries_rate_limits_with_exponential_backoff() -> None:
    fake_client = MagicMock()
    fake_client.rerank.side_effect = [_RateLimited(), _RateLimited(), _ok_response()]
    sleeps: list[float] = []
    reranker = CohereReranker(client=fake_client, sleep=sleeps.append)

    reranked = reranker.rerank("query", [_result("a", "text a", 0.1)], top_k=1)

    assert [r.id for r in reranked] == ["a"]
    assert fake_client.rerank.call_count == 3
    assert sleeps == [5.0, 10.0]


def test_rerank_backoff_honors_retry_after_up_to_the_cap() -> None:
    fake_client = MagicMock()
    fake_client.rerank.side_effect = [
        _RateLimited({"Retry-After": "7"}),
        _RateLimited({"retry-after": "600"}),
        _ok_response(),
    ]
    sleeps: list[float] = []
    reranker = CohereReranker(client=fake_client, sleep=sleeps.append)

    reranker.rerank("query", [_result("a", "text a", 0.1)], top_k=1)

    assert sleeps == [7.0, 60.0]


def test_rerank_gives_up_after_max_attempts() -> None:
    fake_client = MagicMock()
    fake_client.rerank.side_effect = _RateLimited()
    sleeps: list[float] = []
    reranker = CohereReranker(client=fake_client, max_attempts=3, sleep=sleeps.append)

    with pytest.raises(_RateLimited):
        reranker.rerank("query", [_result("a", "text a", 0.1)], top_k=1)

    assert fake_client.rerank.call_count == 3
    assert sleeps == [5.0, 10.0]


def test_rerank_does_not_retry_other_errors() -> None:
    fake_client = MagicMock()
    fake_client.rerank.side_effect = ValueError("bad request")
    sleeps: list[float] = []
    reranker = CohereReranker(client=fake_client, sleep=sleeps.append)

    with pytest.raises(ValueError):
        reranker.rerank("query", [_result("a", "text a", 0.1)], top_k=1)

    assert fake_client.rerank.call_count == 1
    assert sleeps == []


def test_rerank_recognizes_cohere_too_many_requests_error() -> None:
    from cohere.errors import TooManyRequestsError

    fake_client = MagicMock()
    fake_client.rerank.side_effect = [
        TooManyRequestsError(body={"message": "trial key"}, headers={}),
        _ok_response(),
    ]
    sleeps: list[float] = []
    reranker = CohereReranker(client=fake_client, sleep=sleeps.append)

    reranker.rerank("query", [_result("a", "text a", 0.1)], top_k=1)

    assert sleeps == [5.0]
