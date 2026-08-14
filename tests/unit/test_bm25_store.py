"""The degraded-mode retriever.

The emphasis here is authorization rather than ranking quality. A fallback path
is a classic place for an access-control bypass — the primary path enforces
access in the vector database's metadata filter, and the fallback quietly does
not, because a local index has no filter language. Losing Pinecone would then
*widen* what every user can read.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.auth.principal import Principal, Role
from app.retrieval.bm25_store import Bm25Index, load_or_none
from app.retrieval.chunking import chunk_corpus
from app.retrieval.schema import Document, DocumentFrontmatter

CORPUS_SPEC = [
    ("PUB-001", "public", "payments", "Published fee schedule for card payments"),
    ("INT-001", "internal", "payments", "Payment authorization failure runbook"),
    ("CON-001", "confidential", "payments", "Payment incident root cause analysis"),
    ("RES-001", "restricted", "people", "Compensation bands and salary structure"),
    ("INT-002", "internal", "people", "Remote working guidance for staff"),
    ("CON-002", "confidential", "security", "Security incident response procedure"),
]


def build_document(doc_id: str, access: str, department: str, subject: str) -> Document:
    body = f"""# {subject}

## Summary

{subject}. This document covers payment authorization, connection pool
behaviour, and the operational response expected of the on-call engineer when
failures occur during the settlement window.

## Detail

Further detail about {subject.lower()}, including escalation paths, the
verification steps required before standing down, and the review cadence that
applies to this material under the governing policy.
"""
    return Document(
        frontmatter=DocumentFrontmatter(
            doc_id=doc_id,
            title=subject,
            department=department,
            document_type="incident" if access == "confidential" else "policy",
            access_level=access,
            created_date=date(2026, 3, 14),
            owner="Test Team",
            tags=["test"],
        ),
        body=body,
        source_path=f"data/corpus/{doc_id}.md",
    )


@pytest.fixture(scope="module")
def index() -> Bm25Index:
    documents = [build_document(*spec) for spec in CORPUS_SPEC]
    return Bm25Index.build(chunk_corpus(documents))


def principal(role: Role, departments: set[str] | None = None) -> Principal:
    return Principal("t", "Test", role, frozenset(departments or {"*"}))


QUERY = "payment authorization failure connection pool settlement"


class TestAuthorization:
    def test_viewer_never_sees_above_internal(self, index: Bm25Index) -> None:
        results = index.search(QUERY, principal(Role.VIEWER), top_k=50)
        levels = {r.chunk.metadata.access_level for r in results}
        assert levels <= {"public", "internal"}

    def test_analyst_never_sees_restricted(self, index: Bm25Index) -> None:
        results = index.search(QUERY, principal(Role.ANALYST), top_k=50)
        levels = {r.chunk.metadata.access_level for r in results}
        assert "restricted" not in levels

    def test_administrator_reaches_everything(self, index: Bm25Index) -> None:
        results = index.search(
            "compensation bands salary structure", principal(Role.ADMINISTRATOR), top_k=50
        )
        assert any(r.chunk.metadata.access_level == "restricted" for r in results)

    def test_restricted_material_is_unreachable_by_a_direct_query(self, index: Bm25Index) -> None:
        """Naming the document exactly must still not surface it."""
        for role in (Role.VIEWER, Role.ANALYST):
            results = index.search(
                "compensation bands and salary structure", principal(role), top_k=50
            )
            assert all(r.chunk.metadata.doc_id != "RES-001" for r in results)

    def test_department_scoping_is_enforced(self, index: Bm25Index) -> None:
        scoped = principal(Role.ANALYST, {"payments"})
        results = index.search("remote working guidance staff", scoped, top_k=50)
        assert {r.chunk.metadata.department for r in results} <= {"payments"}

    def test_principal_is_a_required_argument(self) -> None:
        """There must be no call signature that skips authorization."""
        import inspect

        parameter = inspect.signature(Bm25Index.search).parameters["principal"]
        assert parameter.default is inspect.Parameter.empty

    def test_narrowing_arguments_cannot_widen_access(self, index: Bm25Index) -> None:
        """Asking for a department outside the caller's scope grants nothing."""
        scoped = principal(Role.ANALYST, {"payments"})
        results = index.search(QUERY, scoped, departments={"people", "security"}, top_k=50)
        assert results == []


class TestRanking:
    def test_lexical_match_ranks_first(self, index: Bm25Index) -> None:
        results = index.search("compensation bands salary", principal(Role.ADMINISTRATOR), top_k=5)
        assert results[0].chunk.metadata.doc_id == "RES-001"

    def test_results_are_ordered_by_score(self, index: Bm25Index) -> None:
        results = index.search(QUERY, principal(Role.ADMINISTRATOR), top_k=10)
        scores = [r.sparse_score or 0.0 for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_contiguous_after_filtering(self, index: Bm25Index) -> None:
        """Ranks must renumber after denied chunks are dropped, not leave gaps
        that would reveal how many documents were withheld."""
        results = index.search(QUERY, principal(Role.VIEWER), top_k=10)
        assert [r.sparse_rank for r in results] == list(range(1, len(results) + 1))

    def test_provenance_marks_the_fallback_retriever(self, index: Bm25Index) -> None:
        results = index.search(QUERY, principal(Role.ADMINISTRATOR), top_k=3)
        assert all(r.retrievers == ["bm25"] for r in results)

    def test_top_k_is_respected(self, index: Bm25Index) -> None:
        assert len(index.search(QUERY, principal(Role.ADMINISTRATOR), top_k=2)) <= 2

    @pytest.mark.parametrize("query", ["", "   ", "\n\t"])
    def test_blank_queries_return_nothing(self, index: Bm25Index, query: str) -> None:
        assert index.search(query, principal(Role.ADMINISTRATOR)) == []

    def test_query_matching_nothing_returns_nothing(self, index: Bm25Index) -> None:
        """bm25s pads its result set to k with zero-score rows.

        Passing those through would hand the response agent text bearing no
        lexical relationship to the question — the exact input that produces a
        confident answer citing an unrelated document.
        """
        results = index.search("zzzz qqqq xxxx", principal(Role.ADMINISTRATOR), top_k=5)
        assert results == []

    def test_every_returned_result_has_a_positive_score(self, index: Bm25Index) -> None:
        results = index.search(QUERY, principal(Role.ADMINISTRATOR), top_k=50)
        assert results
        assert all((r.sparse_score or 0.0) > 0 for r in results)


class TestPersistence:
    def test_round_trip_preserves_results(self, index: Bm25Index, tmp_path: Path) -> None:
        index.save(tmp_path / "bm25")
        reloaded = Bm25Index.load(tmp_path / "bm25")

        admin = principal(Role.ADMINISTRATOR)
        before = [r.chunk.chunk_id for r in index.search(QUERY, admin, top_k=5)]
        after = [r.chunk.chunk_id for r in reloaded.search(QUERY, admin, top_k=5)]
        assert before == after

    def test_reloaded_index_still_enforces_authorization(
        self, index: Bm25Index, tmp_path: Path
    ) -> None:
        """Metadata must survive serialisation, or the fallback silently opens up."""
        index.save(tmp_path / "bm25")
        reloaded = Bm25Index.load(tmp_path / "bm25")
        results = reloaded.search("compensation bands salary", principal(Role.VIEWER), top_k=50)
        assert all(r.chunk.metadata.doc_id != "RES-001" for r in results)

    def test_load_or_none_tolerates_a_missing_index(self, tmp_path: Path) -> None:
        """A missing fallback must not stop the API booting — it just removes a
        rung from the degradation ladder."""
        assert load_or_none(tmp_path / "does-not-exist") is None

    def test_load_raises_a_helpful_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="make seed"):
            Bm25Index.load(tmp_path / "missing")

    def test_building_over_nothing_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero chunks"):
            Bm25Index.build([])
