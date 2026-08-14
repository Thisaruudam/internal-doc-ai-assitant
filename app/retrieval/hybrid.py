"""Hybrid retrieval.

Puts the pieces together: build the authorization filter, query the dense and
sparse indexes concurrently across every namespace the caller may read, fuse by
rank, rerank, cap per document, and hand back top-k with full provenance.

Two behaviours worth naming:

**Namespace fan-out.** A principal with access to six departments means twelve
concurrent index queries, not one. They run under a single ``asyncio.gather``,
so the wall-clock cost is one round trip rather than twelve. Fusion then happens
across the union — which is why RRF matters here: scores from different
namespaces are not comparable, but ranks fused per-retriever are.

**Degradation is a path, not an error.** If Pinecone fails, retrieval falls back
to the local BM25 index, records what happened in the result, and the answer says
so. The system is allowed to return a worse answer; it is never allowed to return
a silently worse one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.auth.principal import Principal
from app.graph import events
from app.observability.logging import get_logger
from app.retrieval.bm25_store import Bm25Index
from app.retrieval.filters import build_filter, matches_nothing, permitted_departments
from app.retrieval.fusion import deduplicate_by_document, reciprocal_rank_fusion
from app.retrieval.pinecone_store import AsyncPineconeSearch
from app.retrieval.schema import ScoredChunk

log = get_logger(__name__)

_NODE = "retrieval_agent"

#: Errors that indicate a defect in this code rather than a failing dependency.
#: These propagate instead of degrading, so a bug cannot disguise itself as an
#: outage and reach production wearing a fallback's clothes.
_PROGRAMMING_ERRORS = (TypeError, AttributeError, NameError, ImportError, AssertionError)


@dataclass
class RetrievalResult:
    """What retrieval produced, and how."""

    chunks: list[ScoredChunk] = field(default_factory=list)
    #: True when the answer came from the fallback index rather than Pinecone.
    degraded: bool = False
    degraded_reason: str | None = None
    #: Per-stage counts, surfaced to the activity panel.
    stages: dict[str, int] = field(default_factory=dict)
    #: Namespaces actually queried, after authorization.
    namespaces: list[str] = field(default_factory=list)
    reranked: bool = False

    @property
    def evidence(self) -> dict[str, str]:
        """chunk_id -> text, the shape the grounding validator expects."""
        return {r.chunk.chunk_id: r.chunk.text for r in self.chunks}

    def note(self) -> str | None:
        """A sentence the answer can include when capability was reduced."""
        if not self.degraded:
            return None
        return (
            "Note: the primary search index was unavailable, so this answer is "
            "based on a keyword-only fallback search and may be less complete."
        )


class HybridRetriever:
    """Dense + sparse retrieval with fusion, reranking, and a fallback."""

    def __init__(
        self,
        search: AsyncPineconeSearch | None,
        *,
        bm25: Bm25Index | None = None,
        rrf_k: int = 60,
        alpha: float = 0.5,
        top_k_per_retriever: int = 20,
        top_k_final: int = 8,
        max_per_document: int = 2,
        rerank: bool = True,
    ) -> None:
        self._search = search
        self._bm25 = bm25
        self._rrf_k = rrf_k
        self._alpha = alpha
        self._top_k_per_retriever = top_k_per_retriever
        self._top_k_final = top_k_final
        self._max_per_document = max_per_document
        self._rerank = rerank

    async def retrieve(
        self,
        query: str,
        principal: Principal,
        *,
        departments: set[str] | None = None,
        document_types: set[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Run the pipeline for one query."""
        wanted = top_k or self._top_k_final

        metadata_filter = build_filter(
            principal,
            departments=departments,
            document_types=document_types,
            date_from=date_from,
            date_to=date_to,
        )

        if matches_nothing(metadata_filter):
            # Short-circuit before spending a round trip, and give the panel an
            # honest reason rather than an unexplained empty result.
            events.retrieval_stage(_NODE, "filter", count=0, reason="no accessible scope")
            return RetrievalResult(stages={"filter": 0})

        namespaces = sorted(
            permitted_departments(principal) & (departments or permitted_departments(principal))
        )
        if not namespaces:
            return RetrievalResult(stages={"filter": 0})

        if self._search is None:
            return self._fallback(query, principal, wanted, "search backend not configured")

        try:
            dense, sparse = await self._query_all_namespaces(query, namespaces, metadata_filter)
        except _PROGRAMMING_ERRORS:
            # Deliberately not degraded. A TypeError or AttributeError is a bug
            # in this code, not an outage — and silently serving BM25 results
            # would hide it behind a plausible-looking "Pinecone unavailable"
            # message. This exact case cost real debugging time: a wrong
            # argument to the index client looked like a dependency failure.
            log.exception("retrieval_programming_error")
            raise
        except Exception as exc:  # genuine backend failures degrade to the fallback
            log.warning("pinecone_unavailable", error_type=type(exc).__name__)
            events.degradation(
                _NODE,
                component="pinecone",
                reason=type(exc).__name__,
                fallback="local BM25 index",
            )
            return self._fallback(query, principal, wanted, f"Pinecone: {type(exc).__name__}")

        events.retrieval_stage(_NODE, "dense", count=len(dense), namespaces=len(namespaces))
        events.retrieval_stage(_NODE, "sparse", count=len(sparse), namespaces=len(namespaces))

        fused = reciprocal_rank_fusion(dense, sparse, k=self._rrf_k, alpha=self._alpha)
        events.retrieval_stage(_NODE, "fusion", count=len(fused), alpha=self._alpha)

        capped = deduplicate_by_document(fused, max_per_document=self._max_per_document)

        result = RetrievalResult(
            namespaces=namespaces,
            stages={
                "dense": len(dense),
                "sparse": len(sparse),
                "fusion": len(fused),
                "deduplicated": len(capped),
            },
        )

        candidates = capped[: max(wanted * 3, wanted)]
        if self._rerank and candidates:
            try:
                reranked = await self._search.rerank(query, candidates, top_n=wanted)
                result.chunks = reranked
                result.reranked = True
                events.retrieval_stage(_NODE, "rerank", count=len(reranked))
            except Exception as exc:  # reranking is an enhancement, not a requirement
                # Losing the reranker costs precision, not correctness, so the
                # fused order stands rather than failing the turn.
                log.warning("rerank_unavailable", error_type=type(exc).__name__)
                events.degradation(
                    _NODE,
                    component="reranker",
                    reason=type(exc).__name__,
                    fallback="RRF ordering",
                )
                result.chunks = candidates[:wanted]
        else:
            result.chunks = candidates[:wanted]

        result.stages["final"] = len(result.chunks)
        self._assert_authorized(result.chunks, principal)
        return result

    async def _query_all_namespaces(
        self, query: str, namespaces: list[str], metadata_filter: dict[str, Any]
    ) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
        """Fan out across namespaces, both retrievers, in one gather."""
        assert self._search is not None

        tasks = []
        for namespace in namespaces:
            tasks.append(
                self._search.dense_search(
                    query,
                    namespace=namespace,
                    metadata_filter=metadata_filter,
                    top_k=self._top_k_per_retriever,
                )
            )
            tasks.append(
                self._search.sparse_search(
                    query,
                    namespace=namespace,
                    metadata_filter=metadata_filter,
                    top_k=self._top_k_per_retriever,
                )
            )

        # return_exceptions so the two retrievers fail independently. Losing the
        # sparse index should cost lexical precision, not discard the dense
        # results and drop the whole turn to the fallback — which is what a bare
        # gather does, because the first exception cancels the rest.
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        dense: list[ScoredChunk] = []
        sparse: list[ScoredChunk] = []
        failures: list[str] = []

        for index, batch in enumerate(responses):
            retriever = "dense" if index % 2 == 0 else "sparse"
            if isinstance(batch, BaseException):
                if isinstance(batch, _PROGRAMMING_ERRORS):
                    raise batch
                failures.append(f"{retriever}/{namespaces[index // 2]}")
                continue
            (dense if retriever == "dense" else sparse).extend(batch)

        if failures:
            log.warning("retriever_partial_failure", failed=failures)
            events.degradation(
                _NODE,
                component=f"{len(failures)} index query(s)",
                reason="partial retriever failure",
                fallback="results from the surviving retriever",
            )

        if not dense and not sparse:
            raise RuntimeError("both retrievers failed across every namespace")

        # Re-rank within each retriever across the merged namespaces, so RRF
        # sees one coherent ordering rather than several interleaved ones.
        dense.sort(key=lambda r: -(r.dense_score or 0.0))
        sparse.sort(key=lambda r: -(r.sparse_score or 0.0))
        for rank, item in enumerate(dense, start=1):
            item.dense_rank = rank
        for rank, item in enumerate(sparse, start=1):
            item.sparse_rank = rank

        return dense, sparse

    def _fallback(
        self, query: str, principal: Principal, wanted: int, reason: str
    ) -> RetrievalResult:
        """Serve from the local BM25 index."""
        if self._bm25 is None:
            log.error("no_retrieval_available", reason=reason)
            return RetrievalResult(
                degraded=True,
                degraded_reason=f"{reason}; no fallback index available",
                stages={"final": 0},
            )

        hits = self._bm25.search(query, principal, top_k=wanted)
        self._assert_authorized(hits, principal)
        events.retrieval_stage(_NODE, "bm25_fallback", count=len(hits))

        return RetrievalResult(
            chunks=hits,
            degraded=True,
            degraded_reason=reason,
            stages={"bm25_fallback": len(hits), "final": len(hits)},
        )

    @staticmethod
    def _assert_authorized(results: list[ScoredChunk], principal: Principal) -> None:
        """Final check on the way out.

        The Pinecone filter should already guarantee this. It is re-checked
        because a filter that is silently mis-constructed — a typo, a schema
        drift, a future refactor — leaks without any other signal.
        """
        leaked = [
            r.chunk.chunk_id
            for r in results
            if not principal.may_read(r.chunk.metadata.access_level, r.chunk.metadata.department)
        ]
        if leaked:
            raise AssertionError(
                f"retrieval returned {len(leaked)} unauthorized chunk(s) for role "
                f"{principal.role.value}: {leaked[:5]}"
            )
