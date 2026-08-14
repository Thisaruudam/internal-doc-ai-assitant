"""Gemini embeddings.

Two details here are easy to get wrong and both materially affect retrieval
quality.

**Asymmetric task types.** ``gemini-embedding-001`` produces different vectors
for the same text depending on whether it is being indexed or queried.
Documents are embedded with ``RETRIEVAL_DOCUMENT`` and questions with
``RETRIEVAL_QUERY``; the model was trained so that a question embedded as a
query lands near the passage that answers it, which is not the same as landing
near passages that resemble it. Using one task type for both is the single most
common way to leave recall on the table.

**Re-normalisation after truncation.** The model natively emits 3072 dimensions
and is Matryoshka-trained, so a prefix of the vector is still meaningful — that
is what makes 1536 dimensions viable. But only the full-length vector is
normalised to unit length. A truncated prefix is not, and feeding unnormalised
vectors into a cosine index quietly distorts every similarity score. Truncated
vectors are therefore re-normalised here.
"""

from __future__ import annotations

import asyncio
import math
from enum import StrEnum

from google import genai
from google.genai import types

from app.config import GeminiSettings
from app.observability.logging import get_logger

log = get_logger(__name__)

#: Gemini caps how many inputs one embed request may carry.
_MAX_BATCH = 100

#: Native output width of gemini-embedding-001.
_NATIVE_DIMENSIONS = 3072


class EmbeddingTask(StrEnum):
    """Why a piece of text is being embedded.

    The task type is part of the input, not a hint — the same string embedded
    under two task types produces two different vectors.
    """

    DOCUMENT = "RETRIEVAL_DOCUMENT"
    QUERY = "RETRIEVAL_QUERY"


def _normalise(vector: list[float]) -> list[float]:
    """Scale to unit length.

    Required whenever the vector has been truncated below the native width:
    cosine similarity on unnormalised vectors is not cosine similarity.
    """
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0.0:
        return vector
    return [component / magnitude for component in vector]


class GeminiEmbedder:
    """Async embedding client with batching and bounded concurrency."""

    def __init__(self, settings: GeminiSettings, *, max_concurrency: int = 4) -> None:
        self._settings = settings
        self._client = genai.Client(api_key=settings.api_key.get_secret_value())
        # Bounded rather than unbounded gather: the free tier rate-limits, and a
        # burst of 40 concurrent requests fails slower than 4 sequential batches.
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def dimensions(self) -> int:
        return self._settings.embedding_dimensions

    async def _embed_batch(self, texts: list[str], task: EmbeddingTask) -> list[list[float]]:
        async with self._semaphore:
            response = await self._client.aio.models.embed_content(
                model=self._settings.embedding_model,
                contents=texts,  # type: ignore[arg-type]
                config=types.EmbedContentConfig(
                    task_type=task.value,
                    output_dimensionality=self.dimensions,
                ),
            )

        vectors: list[list[float]] = []
        for embedding in response.embeddings or []:
            values = list(embedding.values or [])
            if self.dimensions < _NATIVE_DIMENSIONS:
                values = _normalise(values)
            vectors.append(values)
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus passages for indexing."""
        return await self._embed_many(texts, EmbeddingTask.DOCUMENT)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a user question for searching."""
        vectors = await self._embed_many([text], EmbeddingTask.QUERY)
        return vectors[0]

    async def _embed_many(self, texts: list[str], task: EmbeddingTask) -> list[list[float]]:
        if not texts:
            return []

        batches = [texts[i : i + _MAX_BATCH] for i in range(0, len(texts), _MAX_BATCH)]
        results = await asyncio.gather(*(self._embed_batch(batch, task) for batch in batches))

        vectors = [vector for batch in results for vector in batch]
        if len(vectors) != len(texts):
            raise RuntimeError(f"embedding returned {len(vectors)} vectors for {len(texts)} inputs")

        log.info(
            "embeddings_created",
            count=len(vectors),
            task=task.value,
            dimensions=self.dimensions,
        )
        return vectors
