"""Pinecone dense and sparse indexes.

Two indexes rather than one hybrid index. Pinecone supports both shapes, and the
single-index form is simpler — but separate indexes are the documented path when
you want Pinecone's integrated ``pinecone-sparse-english-v0`` model and
independent reranking, and they make the fusion step explicit rather than hiding
it inside a server-side score blend. Explicit matters here: the activity panel
shows dense rank, sparse rank, and fused rank as separate numbers, which is only
possible if the fusion happens where we can see it.

Provisioning is idempotent. ``ensure_indexes`` creates what is missing and
leaves what exists alone, so ``make seed`` is safe to run repeatedly.

Namespaces partition by department. That gives cheap filtering and, more
usefully, a hard blast radius: a mis-constructed metadata filter can only leak
within one namespace, never across the whole corpus.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

from pinecone import Pinecone, PineconeAsyncio, ServerlessSpec

from app.config import GeminiSettings, PineconeSettings
from app.observability.logging import get_logger
from app.retrieval.embeddings import GeminiEmbedder
from app.retrieval.schema import Chunk, ChunkMetadata, ScoredChunk

log = get_logger(__name__)

#: Pinecone caps a single upsert request. Dense vectors at 1536 dimensions are
#: ~6KB each, so batches are kept well under the 2MB request ceiling.
_DENSE_BATCH = 100
_SPARSE_BATCH = 96

#: Field name the integrated sparse model reads text from.
_TEXT_FIELD = "chunk_text"


def _metadata_for_pinecone(metadata: ChunkMetadata) -> dict[str, Any]:
    """Pinecone metadata accepts strings, numbers, booleans, and string lists."""
    payload = metadata.to_pinecone()
    return {key: value for key, value in payload.items() if value is not None}


class PineconeStore:
    """Provisioning and indexing. Synchronous — used by the seed script."""

    def __init__(self, settings: PineconeSettings, gemini: GeminiSettings) -> None:
        self._settings = settings
        self._gemini = gemini
        self._client = Pinecone(api_key=settings.api_key.get_secret_value())
        self._embedder = GeminiEmbedder(gemini)

    # ── provisioning ────────────────────────────────────────────────────

    def ensure_indexes(self) -> None:
        """Create the dense and sparse indexes if they do not already exist."""
        existing = {index["name"] for index in self._client.list_indexes()}

        if self._settings.dense_index not in existing:
            log.info("creating_dense_index", name=self._settings.dense_index)
            self._client.create_index(
                name=self._settings.dense_index,
                dimension=self._gemini.embedding_dimensions,
                # Cosine, because the embeddings are unit-normalised. Dotproduct
                # would be equivalent for normalised vectors but only cosine
                # stays correct if normalisation is ever missed.
                metric="cosine",
                spec=ServerlessSpec(cloud=self._settings.cloud, region=self._settings.region),
            )
        else:
            log.info("dense_index_exists", name=self._settings.dense_index)

        if self._settings.sparse_index not in existing:
            log.info("creating_sparse_index", name=self._settings.sparse_index)
            # Integrated embedding: Pinecone runs the sparse model itself, so
            # the lexical side needs no local model and no vocabulary to keep
            # in step with the index.
            self._client.create_index_for_model(
                name=self._settings.sparse_index,
                cloud=self._settings.cloud,
                region=self._settings.region,
                embed={
                    "model": self._settings.sparse_model,
                    "field_map": {"text": _TEXT_FIELD},
                },
            )
        else:
            log.info("sparse_index_exists", name=self._settings.sparse_index)

        self._wait_until_ready()

    def _wait_until_ready(self, timeout_s: float = 180.0) -> None:
        """Block until both indexes accept writes.

        Serverless index creation is asynchronous; upserting before the index is
        ready fails with an unhelpful error, so the seed script waits here rather
        than surfacing that to whoever ran `make seed`.
        """
        import time

        deadline = time.monotonic() + timeout_s
        for name in (self._settings.dense_index, self._settings.sparse_index):
            while time.monotonic() < deadline:
                if self._client.describe_index(name).status.get("ready"):
                    break
                time.sleep(2)
            else:
                raise TimeoutError(f"index {name!r} was not ready within {timeout_s:g}s")

    # ── indexing ────────────────────────────────────────────────────────

    async def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """Index every chunk into both stores, namespaced by department.

        Async because the embedder holds a client bound to a single event loop.
        Calling ``asyncio.run`` per namespace would create and tear down a loop
        each time, and the second namespace would find the client's transport
        already closed.
        """
        by_namespace: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_namespace.setdefault(chunk.metadata.department, []).append(chunk)

        dense_index = self._client.Index(self._settings.dense_index)
        sparse_index = self._client.Index(self._settings.sparse_index)
        total = 0

        for namespace, group in sorted(by_namespace.items()):
            vectors = await self._embedder.embed_documents([chunk.text for chunk in group])

            dense_payload = [
                {
                    "id": chunk.chunk_id,
                    "values": vector,
                    "metadata": _metadata_for_pinecone(chunk.metadata) | {"text": chunk.text},
                }
                for chunk, vector in zip(group, vectors, strict=True)
            ]
            # The Pinecone data-plane client here is synchronous. Offloaded to a
            # worker thread so a long seed does not block the loop the embedder
            # is using for its own requests.
            #
            # functools.partial rather than a lambda: a lambda closes over the
            # loop variable by reference, so every deferred call would see the
            # last namespace rather than its own.
            for start in range(0, len(dense_payload), _DENSE_BATCH):
                await asyncio.to_thread(
                    partial(
                        dense_index.upsert,
                        vectors=dense_payload[start : start + _DENSE_BATCH],
                        namespace=namespace,
                        show_progress=False,
                    )
                )

            sparse_payload = [
                {
                    "_id": chunk.chunk_id,
                    _TEXT_FIELD: chunk.text,
                    **_metadata_for_pinecone(chunk.metadata),
                }
                for chunk in group
            ]
            for start in range(0, len(sparse_payload), _SPARSE_BATCH):
                await asyncio.to_thread(
                    partial(
                        sparse_index.upsert_records,
                        namespace=namespace,
                        records=sparse_payload[start : start + _SPARSE_BATCH],
                    )
                )

            total += len(group)
            log.info("namespace_indexed", namespace=namespace, chunks=len(group))

        return total

    def describe(self) -> dict[str, Any]:
        return {
            "dense": self._client.Index(self._settings.dense_index)
            .describe_index_stats()
            .to_dict(),
            "sparse": self._client.Index(self._settings.sparse_index)
            .describe_index_stats()
            .to_dict(),
        }


def _chunk_from_metadata(chunk_id: str, metadata: dict[str, Any], text: str) -> Chunk:
    """Rebuild a typed chunk from what Pinecone stored."""
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            doc_id=str(metadata.get("doc_id", "")),
            title=str(metadata.get("title", "")),
            department=str(metadata.get("department", "")),
            document_type=str(metadata.get("document_type", "")),
            access_level=str(metadata.get("access_level", "")),
            created_date=str(metadata.get("created_date", "")),
            created_ts=int(metadata.get("created_ts", 0)),
            owner=str(metadata.get("owner", "")),
            tags=[str(t) for t in metadata.get("tags", [])],
            source_uri=str(metadata.get("source_uri", "")),
            chunk_idx=int(metadata.get("chunk_idx", 0)),
            heading_path=str(metadata.get("heading_path", "")),
            token_estimate=int(metadata.get("token_estimate", 0)),
        ),
    )


class AsyncPineconeSearch:
    """Query-time client. Async, because dense and sparse run concurrently.

    Held open for the process lifetime rather than constructed per request:
    Pinecone's async client maintains a connection pool, and rebuilding it on
    every question adds a TLS handshake to the critical path.
    """

    def __init__(self, settings: PineconeSettings, gemini: GeminiSettings) -> None:
        self._settings = settings
        self._embedder = GeminiEmbedder(gemini)
        self._client: PineconeAsyncio | None = None
        self._hosts: dict[str, str] = {}

    async def __aenter__(self) -> AsyncPineconeSearch:
        self._client = PineconeAsyncio(api_key=self._settings.api_key.get_secret_value())
        # Index handles are addressed by host, not by name. Resolved once here
        # rather than per query: a describe call on the critical path would add
        # a round trip to every question asked.
        for name in (self._settings.dense_index, self._settings.sparse_index):
            description = await self._client.describe_index(name)
            if not description.host:
                raise RuntimeError(f"index {name!r} reported no host; is it still provisioning?")
            self._hosts[name] = description.host
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _require_client(self) -> PineconeAsyncio:
        if self._client is None:
            raise RuntimeError("AsyncPineconeSearch must be used as an async context manager")
        return self._client

    async def dense_search(
        self,
        query: str,
        *,
        namespace: str,
        metadata_filter: dict[str, Any],
        top_k: int,
    ) -> list[ScoredChunk]:
        vector = await self._embedder.embed_query(query)
        client = self._require_client()

        async with client.IndexAsyncio(host=self._hosts[self._settings.dense_index]) as index:
            response = await index.query(
                top_k=top_k,
                vector=vector,
                namespace=namespace,
                filter=metadata_filter,
                include_metadata=True,
            )

        results: list[ScoredChunk] = []
        for rank, match in enumerate(response.get("matches", []), start=1):
            metadata = dict(match.get("metadata") or {})
            text = str(metadata.pop("text", ""))
            results.append(
                ScoredChunk(
                    chunk=_chunk_from_metadata(match["id"], metadata, text),
                    dense_rank=rank,
                    dense_score=float(match.get("score", 0.0)),
                    retrievers=["dense"],
                )
            )
        return results

    async def sparse_search(
        self,
        query: str,
        *,
        namespace: str,
        metadata_filter: dict[str, Any],
        top_k: int,
    ) -> list[ScoredChunk]:
        client = self._require_client()

        async with client.IndexAsyncio(host=self._hosts[self._settings.sparse_index]) as index:
            response = await index.search(
                namespace=namespace,
                query={
                    "top_k": top_k,
                    "inputs": {"text": query},
                    "filter": metadata_filter,
                },
                fields=["*"],
            )

        results: list[ScoredChunk] = []
        hits = response.get("result", {}).get("hits", [])
        for rank, hit in enumerate(hits, start=1):
            # Hit exposes id/score as attributes; the mapping interface uses the
            # trailing-underscore names, so attribute access is the stable route.
            fields = dict(hit.fields or {})
            text = str(fields.pop(_TEXT_FIELD, ""))
            results.append(
                ScoredChunk(
                    chunk=_chunk_from_metadata(hit.id, fields, text),
                    sparse_rank=rank,
                    sparse_score=float(hit.score),
                    retrievers=["sparse"],
                )
            )
        return results

    async def rerank(
        self, query: str, results: list[ScoredChunk], *, top_n: int
    ) -> list[ScoredChunk]:
        """Reorder fused results with Pinecone's hosted cross-encoder.

        Fusion knows only rank agreement; a cross-encoder reads the query and
        the passage together and is markedly better at the top of the list,
        which is the part that reaches the answer.
        """
        if not results:
            return results

        client = self._require_client()
        documents = [{"id": r.chunk.chunk_id, "text": r.chunk.text[:4000]} for r in results]

        response = await client.inference.rerank(
            model=self._settings.rerank_model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            return_documents=False,
        )

        by_id = {r.chunk.chunk_id: r for r in results}
        reranked: list[ScoredChunk] = []
        for row in response.data:
            index = row["index"] if isinstance(row, dict) else row.index
            score = row["score"] if isinstance(row, dict) else row.score
            scored = by_id.get(documents[index]["id"])
            if scored is None:
                continue
            scored.rerank_score = float(score)
            reranked.append(scored)
        return reranked
