"""Local BM25 index — the degraded-mode retriever.

When Pinecone is unreachable the system does not stop answering; it falls back to
this index, says so in the response, and emits a ``degradation`` event so the
drop in capability is visible rather than silent.

**The security consideration that shapes this module.** A fallback path is a
classic place for an authorization bypass: the primary path enforces access
control in the vector database's metadata filter, and the fallback quietly
doesn't, because a local index has no filter language. That would mean losing
Pinecone silently *widens* what every user can read — a far worse outcome than
returning nothing.

So authorization here is enforced in two places, and the more important one is
the second:

1. Candidates are filtered against the ``Principal`` before scoring is reported.
2. ``search`` asserts the invariant on the way out, so a future refactor that
   drops step 1 fails loudly instead of leaking.

The index is built at seed time and persisted, so a cold API process can serve
degraded traffic immediately rather than re-reading the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bm25s
import Stemmer

from app.auth.principal import Principal
from app.observability.logging import get_logger
from app.retrieval.schema import Chunk, ScoredChunk

log = get_logger(__name__)

_STEMMER_LANGUAGE = "english"
_CHUNKS_FILENAME = "chunks.jsonl"

#: How many raw candidates to pull before authorization filtering.
#:
#: BM25 cannot filter during scoring, so restricted documents occupy slots in
#: the raw ranking and are removed afterwards. Over-fetching keeps the post-
#: filter result set full for a viewer, who may be denied a large share of hits.
_OVERFETCH_FACTOR = 6

#: Scores at or below this are treated as "no match" and discarded.
#:
#: bm25s always returns exactly ``k`` rows, padding with zero-score entries when
#: fewer documents match. Passing those through would hand the response agent
#: text with no lexical relationship to the question — which is precisely the
#: input that produces a confident answer citing an unrelated document.
_MIN_RELEVANCE_SCORE = 0.0


class Bm25Index:
    """A persisted lexical index over the corpus chunks."""

    def __init__(self, retriever: bm25s.BM25, chunks: list[Chunk]) -> None:
        self._retriever = retriever
        self._chunks = chunks
        self._stemmer = Stemmer.Stemmer(_STEMMER_LANGUAGE)

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def build(cls, chunks: list[Chunk]) -> Bm25Index:
        """Build an index in memory from the chunked corpus."""
        if not chunks:
            raise ValueError("cannot build a BM25 index over zero chunks")

        stemmer = Stemmer.Stemmer(_STEMMER_LANGUAGE)
        tokens = bm25s.tokenize(
            [chunk.text for chunk in chunks],
            stemmer=stemmer,
            show_progress=False,
        )

        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)

        log.info("bm25_index_built", chunk_count=len(chunks))
        return cls(retriever, list(chunks))

    def save(self, directory: str | Path) -> None:
        """Persist the index and the chunk payloads side by side.

        The chunk metadata is written separately rather than through bm25s's own
        corpus support: authorization filtering needs the full typed metadata,
        and round-tripping it through the library's opaque corpus format would
        lose the guarantees ``ChunkMetadata`` provides.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        self._retriever.save(str(path))
        with (path / _CHUNKS_FILENAME).open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(chunk.model_dump_json() + "\n")

        log.info("bm25_index_saved", directory=str(path), chunk_count=len(self._chunks))

    @classmethod
    def load(cls, directory: str | Path) -> Bm25Index:
        path = Path(directory)
        chunks_file = path / _CHUNKS_FILENAME
        if not chunks_file.exists():
            raise FileNotFoundError(f"no BM25 index at {path} — run `make seed` to build it")

        retriever = bm25s.BM25.load(str(path))
        with chunks_file.open(encoding="utf-8") as handle:
            chunks = [Chunk.model_validate(json.loads(line)) for line in handle if line.strip()]

        log.info("bm25_index_loaded", directory=str(path), chunk_count=len(chunks))
        return cls(retriever, chunks)

    # ── retrieval ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        principal: Principal,
        *,
        top_k: int = 20,
        departments: set[str] | None = None,
        document_types: set[str] | None = None,
        min_score: float = _MIN_RELEVANCE_SCORE,
    ) -> list[ScoredChunk]:
        """Rank chunks the caller is entitled to read.

        ``principal`` is required, not optional. Making it a mandatory positional
        argument means there is no call signature that skips authorization.
        """
        if not query.strip() or not self._chunks:
            return []

        candidate_count = min(len(self._chunks), max(top_k * _OVERFETCH_FACTOR, top_k))
        query_tokens = bm25s.tokenize(query, stemmer=self._stemmer, show_progress=False)

        indices, scores = self._retriever.retrieve(
            query_tokens, k=candidate_count, show_progress=False
        )

        results: list[ScoredChunk] = []
        denied = 0

        for position in range(indices.shape[1]):
            score = float(scores[0, position])

            # bm25s pads to k with zero-score rows; those are not results.
            # Scores descend, so the first one at or below the floor ends the
            # useful part of the ranking.
            if score <= min_score:
                break

            chunk = self._chunks[int(indices[0, position])]
            metadata = chunk.metadata

            if not principal.may_read(metadata.access_level, metadata.department):
                denied += 1
                continue
            if departments and metadata.department not in departments:
                continue
            if document_types and metadata.document_type not in document_types:
                continue

            results.append(
                ScoredChunk(
                    chunk=chunk,
                    # Renumbered after filtering, so gaps do not reveal how many
                    # documents were withheld from this caller.
                    sparse_rank=len(results) + 1,
                    sparse_score=score,
                    fused_score=score,
                    retrievers=["bm25"],
                )
            )
            if len(results) >= top_k:
                break

        # Belt and braces. The loop above already filtered; this asserts the
        # property a future refactor might break. Retrieval leaks are silent by
        # nature — nothing looks wrong in a response containing a document the
        # reader was not entitled to.
        _assert_authorized(results, principal)

        if denied:
            log.info(
                "bm25_results_filtered",
                denied=denied,
                returned=len(results),
                role=principal.role.value,
            )
        return results

    def __len__(self) -> int:
        return len(self._chunks)


def _assert_authorized(results: list[ScoredChunk], principal: Principal) -> None:
    leaked = [
        result.chunk.chunk_id
        for result in results
        if not principal.may_read(
            result.chunk.metadata.access_level, result.chunk.metadata.department
        )
    ]
    if leaked:
        # Fail closed and loudly rather than returning a partially-filtered list.
        raise AssertionError(
            f"BM25 fallback returned {len(leaked)} unauthorized chunk(s) for "
            f"role {principal.role.value}: {leaked[:5]}"
        )


def load_or_none(directory: str | Path) -> Bm25Index | None:
    """Load the index if it exists, else ``None``.

    Used at start-up: a missing fallback index must not stop the API from
    booting, it just means the degradation ladder has one fewer rung. The health
    endpoint reports it.
    """
    try:
        return Bm25Index.load(directory)
    except (FileNotFoundError, ValueError, OSError) as exc:
        log.warning("bm25_index_unavailable", directory=str(directory), reason=str(exc))
        return None


def index_stats(index: Bm25Index | None) -> dict[str, Any]:
    return {"available": index is not None, "chunk_count": len(index) if index else 0}
