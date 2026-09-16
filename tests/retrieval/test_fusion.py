from src.ingestion.models import FilingMetadata
from src.retrieval.fusion import default_candidate_pool_size, reciprocal_rank_fusion
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


def _result(id_: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        id=id_, text=f"text for {id_}", score=score, filing=_FILING, chunk_index=0, total_chunks=1
    )


def test_reciprocal_rank_fusion_boosts_items_ranked_in_both_lists() -> None:
    dense = [_result("a", 0.9), _result("b", 0.8), _result("c", 0.7)]
    sparse = [_result("b", 12.0), _result("d", 10.0), _result("a", 8.0)]

    fused = reciprocal_rank_fusion([dense, sparse], top_k=4)

    # "b" is rank 2 in dense and rank 1 in sparse -- present in both lists
    # gives it the highest combined RRF score, ahead of "a" (rank 1 + rank 3).
    assert [r.id for r in fused] == ["b", "a", "d", "c"]


def test_reciprocal_rank_fusion_respects_top_k() -> None:
    dense = [_result("a", 0.9), _result("b", 0.8)]
    sparse = [_result("c", 5.0), _result("d", 4.0)]

    fused = reciprocal_rank_fusion([dense, sparse], top_k=2)

    assert len(fused) == 2


def test_reciprocal_rank_fusion_replaces_original_score() -> None:
    dense = [_result("a", 0.9)]
    fused = reciprocal_rank_fusion([dense], top_k=1, k=60)

    assert fused[0].score == 1.0 / 61


def test_reciprocal_rank_fusion_handles_empty_lists() -> None:
    assert reciprocal_rank_fusion([[], []], top_k=5) == []


def test_default_candidate_pool_size_has_a_floor_and_scales_with_top_k() -> None:
    assert default_candidate_pool_size(1) == 20
    assert default_candidate_pool_size(10) == 40
