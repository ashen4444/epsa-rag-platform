"""Exact cosine search using an L2-normalized FAISS IndexFlatIP."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from epsa_rag.core.exceptions import FrozenArtifactError, IndexIntegrityError
from epsa_rag.retrieval.config import DenseConfig
from epsa_rag.retrieval.interfaces import FloatMatrix, FloatVector
from epsa_rag.retrieval.io import (
    read_json_object,
    write_json_artifact_exclusive,
)
from epsa_rag.retrieval.models import BackendHit


def normalize_rows(vectors: FloatMatrix, *, dimensions: int) -> FloatMatrix:
    """Return independent contiguous normalized float32 rows."""

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimensions:
        raise IndexIntegrityError(
            f"dense vectors have shape {matrix.shape}; expected (*, {dimensions})"
        )
    if not np.isfinite(matrix).all():
        raise IndexIntegrityError("dense vectors contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise IndexIntegrityError("dense vectors contain a zero vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class FaissFlatIPIndex:
    """Chunk-ID mapping around an exact FAISS inner-product index."""

    def __init__(self, *, index: Any, chunk_ids: Sequence[str], config: DenseConfig) -> None:
        self._index = index
        self._chunk_ids = tuple(chunk_ids)
        self.config = config
        self._validate()

    @classmethod
    def build(
        cls,
        *,
        vectors: FloatMatrix,
        chunk_ids: Sequence[str],
        config: DenseConfig,
    ) -> FaissFlatIPIndex:
        """Normalize vectors and add them in stable corpus order."""

        if not chunk_ids:
            raise IndexIntegrityError("cannot build a dense index from an empty corpus")
        normalized = normalize_rows(vectors, dimensions=config.dimensions)
        if normalized.shape[0] != len(chunk_ids):
            raise IndexIntegrityError("dense vector rows do not align with chunk IDs")
        index = faiss.IndexFlatIP(config.dimensions)
        index.add(normalized)
        return cls(index=index, chunk_ids=chunk_ids, config=config)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self._chunk_ids

    def search_vector(self, vector: FloatVector, *, top_k: int) -> tuple[BackendHit, ...]:
        """Perform exhaustive cosine search with stable score/ID tie-breaking."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = np.asarray(vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.config.dimensions:
            raise IndexIntegrityError(
                f"query vector has shape {query.shape}; expected ({self.config.dimensions},)"
            )
        normalized_query = normalize_rows(
            query.reshape(1, self.config.dimensions), dimensions=self.config.dimensions
        )
        distances, indices = self._index.search(normalized_query, len(self._chunk_ids))
        candidates = [
            (self._chunk_ids[int(index)], float(score))
            for index, score in zip(indices[0], distances[0], strict=True)
            if index >= 0
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return tuple(
            BackendHit(chunk_id=chunk_id, rank=rank, score=score)
            for rank, (chunk_id, score) in enumerate(candidates[:top_k], start=1)
        )

    def save(self, index_path: Path, chunk_ids_path: Path) -> None:
        """Persist FAISS bytes and the exact positional chunk-ID mapping."""

        for path in (index_path, chunk_ids_path):
            if path.exists():
                raise FrozenArtifactError(f"refusing to overwrite retrieval artifact: {path}")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        write_json_artifact_exclusive(
            chunk_ids_path,
            {"schema_version": "1.0", "chunk_ids": self._chunk_ids},
            relative_path=chunk_ids_path.name,
            record_count=len(self._chunk_ids),
        )

    @classmethod
    def load(
        cls,
        index_path: Path,
        chunk_ids_path: Path,
        *,
        config: DenseConfig,
    ) -> FaissFlatIPIndex:
        """Load FAISS bytes and their positional ID mapping."""

        try:
            index = faiss.read_index(str(index_path))
        except RuntimeError as error:
            message = f"unable to read FAISS index {index_path}: {error}"
            raise IndexIntegrityError(message) from error
        payload = read_json_object(chunk_ids_path)
        try:
            if payload.get("schema_version") != "1.0":
                raise ValueError("unsupported dense chunk-ID schema")
            chunk_ids = tuple(str(value) for value in payload["chunk_ids"])
        except (KeyError, TypeError, ValueError) as error:
            raise IndexIntegrityError(f"invalid dense chunk-ID mapping: {error}") from error
        return cls(index=index, chunk_ids=chunk_ids, config=config)

    def _validate(self) -> None:
        if not self._chunk_ids:
            raise IndexIntegrityError("dense index must contain at least one chunk")
        if len(self._chunk_ids) != len(set(self._chunk_ids)):
            raise IndexIntegrityError("dense chunk IDs must be unique")
        if self._index.d != self.config.dimensions:
            raise IndexIntegrityError("FAISS index dimensions do not match configuration")
        if self._index.ntotal != len(self._chunk_ids):
            raise IndexIntegrityError("FAISS vectors do not align with chunk IDs")
        if self._index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise IndexIntegrityError("FAISS index must use inner-product search")
