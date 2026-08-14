"""Reciprocal Rank Fusion and per-document capping."""

from __future__ import annotations

import pytest

from app.retrieval.fusion import (
    DEFAULT_RRF_K,
    deduplicate_by_document,
    reciprocal_rank_fusion,
)
from app.retrieval.schema import Chunk, ChunkMetadata, ScoredChunk


def chunk(chunk_id: str, doc_id: str | None = None) -> Chunk:
    doc = doc_id or chunk_id.split("#")[0]
    return Chunk(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        metadata=ChunkMetadata(
            doc_id=doc,
            title=f"Document {doc}",
            department="payments",
            document_type="incident",
            access_level="internal",
            created_date="2026-03-14",
            created_ts=63_000_000,
            owner="Payments Platform Team",
            tags=[],
            source_uri=f"corpus://payments/{doc}",
            chunk_idx=0,
            heading_path="Root Cause",
            token_estimate=200,
        ),
    )


def ranked(*chunk_ids: str) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=chunk(c)) for c in chunk_ids]


class TestFusion:
    def test_document_found_by_both_retrievers_outranks_either_alone(self) -> None:
        """The central property: agreement between retrievers is evidence."""
        fused = reciprocal_rank_fusion(ranked("A#0", "B#0"), ranked("C#0", "A#0"))
        assert fused[0].chunk.chunk_id == "A#0"
        assert fused[0].retrievers == ["dense", "sparse"]

    def test_scores_follow_the_rrf_formula(self) -> None:
        fused = reciprocal_rank_fusion(ranked("A#0"), ranked("A#0"), k=60, alpha=0.5)
        expected = 0.5 / 61 + 0.5 / 61
        assert fused[0].fused_score == pytest.approx(expected)

    def test_alpha_one_ignores_the_sparse_list(self) -> None:
        fused = reciprocal_rank_fusion(ranked("A#0"), ranked("B#0"), alpha=1.0)
        by_id = {r.chunk.chunk_id: r.fused_score for r in fused}
        assert by_id["A#0"] == pytest.approx(1 / (DEFAULT_RRF_K + 1))
        assert by_id["B#0"] == pytest.approx(0.0)

    def test_alpha_zero_ignores_the_dense_list(self) -> None:
        fused = reciprocal_rank_fusion(ranked("A#0"), ranked("B#0"), alpha=0.0)
        by_id = {r.chunk.chunk_id: r.fused_score for r in fused}
        assert by_id["A#0"] == pytest.approx(0.0)
        assert by_id["B#0"] == pytest.approx(1 / (DEFAULT_RRF_K + 1))

    def test_alpha_shifts_the_winner(self) -> None:
        """The tuning knob has to actually do something."""
        dense, sparse = ranked("D#0"), ranked("S#0")
        assert reciprocal_rank_fusion(dense, sparse, alpha=0.9)[0].chunk.chunk_id == "D#0"
        assert reciprocal_rank_fusion(dense, sparse, alpha=0.1)[0].chunk.chunk_id == "S#0"

    def test_rank_order_is_preserved_within_a_retriever(self) -> None:
        fused = reciprocal_rank_fusion(ranked("A#0", "B#0", "C#0"), [], alpha=1.0)
        assert [r.chunk.chunk_id for r in fused] == ["A#0", "B#0", "C#0"]

    def test_provenance_is_recorded(self) -> None:
        fused = reciprocal_rank_fusion(ranked("X#0", "A#0"), ranked("A#0"))
        found = next(r for r in fused if r.chunk.chunk_id == "A#0")
        assert found.dense_rank == 2
        assert found.sparse_rank == 1
        assert sorted(found.retrievers) == ["dense", "sparse"]

    def test_chunk_found_by_one_retriever_records_only_that_one(self) -> None:
        fused = reciprocal_rank_fusion(ranked("A#0"), ranked("B#0"))
        only_sparse = next(r for r in fused if r.chunk.chunk_id == "B#0")
        assert only_sparse.retrievers == ["sparse"]
        assert only_sparse.dense_rank is None

    def test_empty_inputs_yield_nothing(self) -> None:
        assert reciprocal_rank_fusion([], []) == []

    def test_one_empty_retriever_is_fine(self) -> None:
        """Happens for real when Pinecone's sparse index is circuit-broken."""
        fused = reciprocal_rank_fusion(ranked("A#0", "B#0"), [])
        assert [r.chunk.chunk_id for r in fused] == ["A#0", "B#0"]

    def test_ties_break_deterministically(self) -> None:
        """Non-deterministic ordering reads as a quality regression in eval."""
        first = reciprocal_rank_fusion(ranked("B#0", "A#0"), ranked("A#0", "B#0"))
        second = reciprocal_rank_fusion(ranked("B#0", "A#0"), ranked("A#0", "B#0"))
        assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]

    def test_larger_k_flattens_the_ranking(self) -> None:
        """k controls how much the top ranks dominate.

        A small k makes rank 1 overwhelmingly important; a large k spreads
        influence down the list. 60 is the conventional middle.
        """

        def score_spread(k: int) -> float:
            fused = reciprocal_rank_fusion(ranked("A#0", "B#0", "C#0"), [], k=k, alpha=1.0)
            return (fused[0].fused_score or 0.0) - (fused[-1].fused_score or 0.0)

        assert score_spread(10) > score_spread(1000)

    @pytest.mark.parametrize("alpha", [-0.1, 1.1, 2.0])
    def test_alpha_outside_the_unit_interval_is_rejected(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            reciprocal_rank_fusion([], [], alpha=alpha)


class TestDeduplicateByDocument:
    def test_one_document_cannot_monopolise_the_results(self) -> None:
        """A 'recurring root causes' question needs many documents, not one."""
        results = [ScoredChunk(chunk=chunk(f"INC-1#{i}", "INC-1")) for i in range(5)]
        results.append(ScoredChunk(chunk=chunk("INC-2#0", "INC-2")))

        kept = deduplicate_by_document(results, max_per_document=2)
        assert len(kept) == 3
        assert sum(1 for r in kept if r.chunk.metadata.doc_id == "INC-1") == 2
        assert any(r.chunk.metadata.doc_id == "INC-2" for r in kept)

    def test_relative_order_is_preserved(self) -> None:
        results = [
            ScoredChunk(chunk=chunk("A#0", "A")),
            ScoredChunk(chunk=chunk("B#0", "B")),
            ScoredChunk(chunk=chunk("A#1", "A")),
        ]
        kept = deduplicate_by_document(results, max_per_document=1)
        assert [r.chunk.chunk_id for r in kept] == ["A#0", "B#0"]

    def test_cap_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            deduplicate_by_document([], max_per_document=0)
