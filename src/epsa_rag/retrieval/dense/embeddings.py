"""OpenAI embedding adapter with locked semantic input formatting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from openai import OpenAI, OpenAIError

from epsa_rag.core.exceptions import EmbeddingError
from epsa_rag.core.models import ParagraphChunk, RetrievalQuery
from epsa_rag.retrieval.config import DenseConfig
from epsa_rag.retrieval.interfaces import FloatMatrix, FloatVector


def format_document(chunk: ParagraphChunk) -> str:
    """Format only the locked semantic title and paragraph content."""

    return f"Title: {chunk.title}\nText: {chunk.paragraph_text}"


class OpenAIEmbeddingProvider:
    """Synchronous, batched adapter for the OpenAI embeddings endpoint."""

    def __init__(
        self,
        config: DenseConfig,
        client: OpenAI | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self._progress_callback = progress_callback
        try:
            self._client = client or OpenAI()
        except OpenAIError as error:
            message = f"unable to initialize OpenAI embedding client: {error}"
            raise EmbeddingError(message) from error

    @property
    def model(self) -> str:
        return self.config.embedding_model

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def embed_documents(self, chunks: Sequence[ParagraphChunk]) -> FloatMatrix:
        """Embed locked document text in exactly the supplied corpus order."""

        return self._embed_texts([format_document(chunk) for chunk in chunks])

    def embed_query(self, query: RetrievalQuery) -> FloatVector:
        """Embed the unmodified complete query text."""

        return cast(FloatVector, self.embed_queries((query,))[0].copy())

    def embed_queries(self, queries: Sequence[RetrievalQuery]) -> FloatMatrix:
        """Batch exact query texts while preserving their supplied order."""

        return self._embed_texts([query.text for query in queries])

    def _embed_texts(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = list(texts[start : start + self.config.batch_size])
            if any(not text for text in batch):
                raise EmbeddingError("embedding input must not be empty")
            try:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self.config.embedding_model,
                    dimensions=self.config.dimensions,
                    encoding_format=self.config.encoding_format,
                )
            except OpenAIError as error:
                raise EmbeddingError(f"OpenAI embedding request failed: {error}") from error
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise EmbeddingError("embedding response indices do not match the request batch")
            vectors.extend(item.embedding for item in ordered)
            if self._progress_callback is not None:
                self._progress_callback(min(start + len(batch), len(texts)), len(texts))

        matrix = np.asarray(vectors, dtype=np.float32)
        expected_shape = (len(texts), self.dimensions)
        if matrix.shape != expected_shape:
            raise EmbeddingError(
                f"embedding matrix has shape {matrix.shape}, expected {expected_shape}"
            )
        if not np.isfinite(matrix).all():
            raise EmbeddingError("embedding response contains non-finite values")
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise EmbeddingError("embedding response contains a zero vector")
        return matrix
