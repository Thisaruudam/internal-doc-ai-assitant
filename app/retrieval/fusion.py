"""Reciprocal Rank Fusion.

Dense cosine similarity and learned-sparse scores are not on a comparable scale.
Blending them with a weighted sum therefore needs a normalisation step, and every
normalisation choice — min-max over the returned window, z-score, softmax — is a
tuning liability that behaves differently on an easy query than on a hard one.

RRF sidesteps the problem by consuming only rank order:

    score(d) = Σ  w_r / (k + rank_r(d))

with ``k = 60``, the value from the original Cormack et al. formulation, which
damps the influence of the very top ranks enough that one retriever's confident
mistake cannot dominate the fused list.

``alpha`` weights dense against sparse (1.0 = pure dense, 0.0 = pure sparse) so
the trade-off is a configuration value that can be demonstrated live rather than
a constant buried in the code.
"""

from __future__ import annotations

from app.retrieval.schema import ScoredChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    dense: list[ScoredChunk],
    sparse: list[ScoredChunk],
    *,
    k: int = DEFAULT_RRF_K,
    alpha: float = 0.5,
) -> list[ScoredChunk]:
    """Fuse two ranked lists into one.

    Both inputs are assumed to be in descending relevance order — the rank used
    is the position in the list, not any score the retriever reported.

    Provenance is preserved: the returned chunks carry each retriever's rank and
    score, and the list of retrievers that surfaced them, so the activity panel
    can show *why* a document ranked where it did.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be within [0, 1], got {alpha}")

    merged: dict[str, ScoredChunk] = {}

    def contribute(results: list[ScoredChunk], weight: float, retriever: str) -> None:
        for position, result in enumerate(results):
            chunk_id = result.chunk.chunk_id
            existing = merged.get(chunk_id)

            if existing is None:
                existing = ScoredChunk(chunk=result.chunk, fused_score=0.0)
                merged[chunk_id] = existing

            rank = position + 1
            existing.fused_score = (existing.fused_score or 0.0) + weight / (k + rank)

            if retriever not in existing.retrievers:
                existing.retrievers.append(retriever)

            if retriever == "dense":
                existing.dense_rank = rank
                existing.dense_score = result.dense_score or result.fused_score
            else:
                existing.sparse_rank = rank
                existing.sparse_score = result.sparse_score or result.fused_score

    contribute(dense, alpha, "dense")
    contribute(sparse, 1.0 - alpha, "sparse")

    # Sort by fused score, then by chunk_id so ties are resolved deterministically
    # rather than by dictionary insertion order. A stable order matters: the
    # evaluation harness compares runs, and non-determinism there looks like a
    # quality regression.
    return sorted(
        merged.values(),
        key=lambda result: (-(result.fused_score or 0.0), result.chunk.chunk_id),
    )


def deduplicate_by_document(
    results: list[ScoredChunk], *, max_per_document: int = 2
) -> list[ScoredChunk]:
    """Cap how many chunks any single document contributes.

    Without this, one long incident report can occupy the entire top-k and crowd
    out the other seven incidents a "recurring root causes" question actually
    needs. Order is otherwise preserved.
    """
    if max_per_document < 1:
        raise ValueError("max_per_document must be at least 1")

    seen: dict[str, int] = {}
    kept: list[ScoredChunk] = []

    for result in results:
        doc_id = result.chunk.metadata.doc_id
        count = seen.get(doc_id, 0)
        if count >= max_per_document:
            continue
        seen[doc_id] = count + 1
        kept.append(result)

    return kept
