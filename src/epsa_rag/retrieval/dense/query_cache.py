"""Immutable, content-addressed query-embedding cache."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol
from uuid import uuid4

import numpy as np
from pydantic import AwareDatetime, Field, ValidationError, field_validator

from epsa_rag.core.exceptions import QueryEmbeddingCacheError
from epsa_rag.core.ids import Identifier, stable_digest, validate_identifier
from epsa_rag.core.models import ContractModel, ParagraphChunk, RetrievalQuery
from epsa_rag.data.io import sha256_file, write_json_exclusive
from epsa_rag.retrieval.config import DenseConfig
from epsa_rag.retrieval.interfaces import EmbeddingProvider, FloatMatrix, FloatVector

QueryEmbeddingCacheMode = Literal["disabled", "read-only", "read-write"]
_PATH_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _path_identifier(value: str) -> str:
    validated = validate_identifier(value)
    if _PATH_IDENTIFIER.fullmatch(validated) is None:
        raise ValueError("path identifier may contain only letters, digits, '.', '_', or '-'")
    return validated


class QueryEmbeddingCacheEntry(ContractModel):
    """Integrity metadata for one immutable cached query vector."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_type: Literal["query-embedding"] = "query-embedding"
    cache_implementation: Literal["query-embedding-cache-v1"] = "query-embedding-cache-v1"
    cache_version: Identifier
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: Identifier
    dimensions: int = Field(ge=1)
    vector_dtype: Literal["float32"] = "float32"
    vector_file: Literal["vector.npy"] = "vector.npy"
    vector_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: AwareDatetime

    @field_validator("generated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value


class QueryEmbeddingObservation(ContractModel):
    """Serializable provenance for one query embedding lookup or creation."""

    cache_mode: QueryEmbeddingCacheMode
    source: Literal["cache", "live"]
    cache_version: Identifier | None
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: Identifier
    dimensions: int = Field(ge=1)
    vector_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_entry: str | None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class QueryEmbeddingReference(ContractModel):
    """One frozen benchmark query's link to an immutable cache entry."""

    question_id: Identifier
    query_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class QueryEmbeddingCollectionManifest(ContractModel):
    """Frozen identity and cache coverage for a complete benchmark question collection."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_type: Literal["query-embedding-collection"] = "query-embedding-collection"
    generator: Literal["epsa-rag-query-cache"] = "epsa-rag-query-cache"
    generator_version: Literal["query-embedding-cache-v1"] = "query-embedding-cache-v1"
    generated_at: AwareDatetime
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    cache_version: Identifier
    dataset_version: Identifier
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: Identifier
    dimensions: int = Field(ge=1)
    question_count: int = Field(ge=1)
    questions: tuple[QueryEmbeddingReference, ...]

    @field_validator("generated_at")
    @classmethod
    def require_collection_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value


class BatchQueryEmbeddingProvider(Protocol):
    """Batch boundary used only while creating a frozen benchmark cache."""

    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_queries(self, queries: Sequence[RetrievalQuery]) -> FloatMatrix: ...


class QueryEmbeddingObservationBuffer:
    """Serial handoff from the embedding boundary to one evaluation trace."""

    def __init__(self) -> None:
        self._pending: list[QueryEmbeddingObservation] = []

    def record(self, observation: QueryEmbeddingObservation) -> None:
        self._pending.append(observation)

    def take(self, query: RetrievalQuery) -> QueryEmbeddingObservation | None:
        if not self._pending:
            return None
        observation = self._pending[0]
        if observation.query_text_sha256 != query_text_sha256(query):
            raise QueryEmbeddingCacheError("query embedding observation order mismatch")
        return self._pending.pop(0)

    def require_empty(self) -> None:
        if self._pending:
            raise QueryEmbeddingCacheError("unconsumed query embedding observations")


def query_text_sha256(query: RetrievalQuery) -> str:
    """Hash the exact query text without trimming or normalization."""

    return hashlib.sha256(query.text.encode("utf-8")).hexdigest()


def query_embedding_cache_key(
    query: RetrievalQuery, *, model: str, dimensions: int, cache_version: str
) -> str:
    """Bind a cache identity to exact semantic input and embedding configuration."""

    return stable_digest(
        "query-embedding-cache-key-v1",
        cache_version,
        model,
        str(dimensions),
        query.text,
    )


def vector_sha256(vector: FloatVector) -> str:
    """Hash canonical float32 vector bytes independently of the NumPy file container."""

    canonical = np.asarray(vector, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


class QueryEmbeddingCache:
    """Read and publish immutable query vectors under content-derived paths."""

    def __init__(self, root: Path, *, config: DenseConfig, cache_version: str) -> None:
        # Validate the caller-selected logical version through the shared identifier contract.
        self.root = root
        self.config = config
        try:
            self.cache_version = _path_identifier(cache_version)
        except ValueError as error:
            raise QueryEmbeddingCacheError(f"invalid query cache version: {error}") from error

    def key(self, query: RetrievalQuery) -> str:
        return query_embedding_cache_key(
            query,
            model=self.config.embedding_model,
            dimensions=self.config.dimensions,
            cache_version=self.cache_version,
        )

    def entry_directory(self, key: str) -> Path:
        return self.root / self.cache_version / "entries" / key[:2] / key

    def relative_entry(self, key: str) -> str:
        return self.entry_directory(key).relative_to(self.root).as_posix()

    def collection_manifest_path(self, dataset_version: str) -> Path:
        validated = _path_identifier(dataset_version)
        return self.root / self.cache_version / "collections" / validated / "manifest.json"

    def load(self, query: RetrievalQuery) -> tuple[FloatVector, QueryEmbeddingCacheEntry] | None:
        key = self.key(query)
        directory = self.entry_directory(key)
        if not directory.exists():
            return None
        manifest_path = directory / "manifest.json"
        vector_path = directory / "vector.npy"
        try:
            manifest = QueryEmbeddingCacheEntry.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as error:
            raise QueryEmbeddingCacheError(f"invalid query cache manifest for {key}") from error
        expected = {
            "cache_version": self.cache_version,
            "cache_key": key,
            "query_text_sha256": query_text_sha256(query),
            "model": self.config.embedding_model,
            "dimensions": self.config.dimensions,
        }
        if any(getattr(manifest, name) != value for name, value in expected.items()):
            raise QueryEmbeddingCacheError(f"query cache metadata mismatch for {key}")
        try:
            if sha256_file(vector_path) != manifest.vector_file_sha256:
                raise QueryEmbeddingCacheError(f"query cache vector checksum mismatch for {key}")
            vector = np.load(vector_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise QueryEmbeddingCacheError(f"invalid query cache vector for {key}") from error
        validated = self._validate_vector(vector, key)
        if vector_sha256(validated) != manifest.vector_sha256:
            raise QueryEmbeddingCacheError(f"query cache vector content mismatch for {key}")
        return validated, manifest

    def store(
        self,
        query: RetrievalQuery,
        vector: FloatVector,
        *,
        generated_at: datetime | None = None,
    ) -> QueryEmbeddingCacheEntry:
        key = self.key(query)
        validated = self._validate_vector(vector, key)
        existing = self.load(query)
        if existing is not None:
            existing_vector, manifest = existing
            if vector_sha256(existing_vector) != vector_sha256(validated):
                raise QueryEmbeddingCacheError(
                    f"refusing conflicting query embedding for immutable key {key}"
                )
            return manifest
        target = self.entry_directory(key)
        stage_root = self.root / f".query-embedding-stage-{uuid4().hex}"
        stage = stage_root / "entry"
        stage.mkdir(parents=True)
        try:
            vector_path = stage / "vector.npy"
            with vector_path.open("xb") as stream:
                np.save(stream, validated, allow_pickle=False)
            manifest = QueryEmbeddingCacheEntry(
                cache_version=self.cache_version,
                cache_key=key,
                query_text_sha256=query_text_sha256(query),
                model=self.config.embedding_model,
                dimensions=self.config.dimensions,
                vector_sha256=vector_sha256(validated),
                vector_file_sha256=sha256_file(vector_path),
                generated_at=generated_at or datetime.now(UTC),
            )
            write_json_exclusive(stage / "manifest.json", manifest)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(stage, target)
            except OSError:
                # Another process may have published the same immutable entry first.
                loaded = self.load(query)
                if loaded is None or vector_sha256(loaded[0]) != vector_sha256(validated):
                    raise
                manifest = loaded[1]
            return manifest
        finally:
            if stage_root.exists():
                shutil.rmtree(stage_root)

    def _validate_vector(self, vector: FloatVector, key: str) -> FloatVector:
        value = np.asarray(vector)
        if value.dtype != np.float32 or value.shape != (self.config.dimensions,):
            raise QueryEmbeddingCacheError(
                f"query cache vector for {key} must be float32 shape ({self.config.dimensions},)"
            )
        if not np.isfinite(value).all() or np.linalg.norm(value) == 0:
            raise QueryEmbeddingCacheError(f"query cache vector for {key} is invalid")
        return np.asarray(value, dtype=np.float32).copy()

    def load_collection(
        self,
        *,
        queries: Sequence[RetrievalQuery],
        dataset_version: str,
        dataset_manifest_sha256: str,
        dataset_file_sha256: str,
    ) -> QueryEmbeddingCollectionManifest:
        """Validate a frozen collection and every referenced cache entry."""

        path = self.collection_manifest_path(dataset_version)
        try:
            manifest = QueryEmbeddingCollectionManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as error:
            raise QueryEmbeddingCacheError(
                f"unable to load frozen query collection {dataset_version}"
            ) from error
        expected_identity = (
            self.cache_version,
            dataset_version,
            dataset_manifest_sha256,
            dataset_file_sha256,
            self.config.embedding_model,
            self.config.dimensions,
            len(queries),
        )
        actual_identity = (
            manifest.cache_version,
            manifest.dataset_version,
            manifest.dataset_manifest_sha256,
            manifest.dataset_file_sha256,
            manifest.model,
            manifest.dimensions,
            manifest.question_count,
        )
        if actual_identity != expected_identity or len(manifest.questions) != len(queries):
            raise QueryEmbeddingCacheError("frozen query collection identity mismatch")
        for query, reference in zip(queries, manifest.questions, strict=True):
            loaded = self.load(query)
            expected_reference = (
                query.question_id,
                query_text_sha256(query),
                self.key(query),
                vector_sha256(loaded[0]) if loaded is not None else None,
            )
            actual_reference = (
                reference.question_id,
                reference.query_text_sha256,
                reference.cache_key,
                reference.vector_sha256,
            )
            if (
                query.question_id is None
                or loaded is None
                or actual_reference != expected_reference
            ):
                raise QueryEmbeddingCacheError(
                    f"frozen query collection entry mismatch for {query.question_id or 'unknown'}"
                )
        return manifest

    def build_collection(
        self,
        *,
        queries: Sequence[RetrievalQuery],
        dataset_version: str,
        dataset_manifest_sha256: str,
        dataset_file_sha256: str,
        git_commit_sha: str,
        provider: BatchQueryEmbeddingProvider,
        generated_at: datetime | None = None,
    ) -> QueryEmbeddingCollectionManifest:
        """Populate cache misses in batches, then freeze the exact benchmark collection."""

        if not queries or any(query.question_id is None for query in queries):
            raise QueryEmbeddingCacheError("frozen query collections require identified queries")
        if (
            provider.model != self.config.embedding_model
            or provider.dimensions != self.config.dimensions
        ):
            raise QueryEmbeddingCacheError("query collection provider configuration mismatch")
        path = self.collection_manifest_path(dataset_version)
        if path.exists():
            return self.load_collection(
                queries=queries,
                dataset_version=dataset_version,
                dataset_manifest_sha256=dataset_manifest_sha256,
                dataset_file_sha256=dataset_file_sha256,
            )
        missing = [query for query in queries if self.load(query) is None]
        if missing:
            vectors = provider.embed_queries(missing)
            expected_shape = (len(missing), self.config.dimensions)
            if vectors.dtype != np.float32 or vectors.shape != expected_shape:
                raise QueryEmbeddingCacheError(
                    f"query collection provider returned {vectors.dtype} {vectors.shape}, "
                    f"expected float32 {expected_shape}"
                )
            for query, vector in zip(missing, vectors, strict=True):
                self.store(query, vector, generated_at=generated_at)
        references: list[QueryEmbeddingReference] = []
        for query in queries:
            if query.question_id is None:  # Guarded by collection validation above.
                raise QueryEmbeddingCacheError("frozen query is missing its question ID")
            loaded = self.load(query)
            if loaded is None:  # pragma: no cover - guarded by population above.
                raise QueryEmbeddingCacheError("query cache population did not create every entry")
            vector, _ = loaded
            references.append(
                QueryEmbeddingReference(
                    question_id=query.question_id,
                    query_text_sha256=query_text_sha256(query),
                    cache_key=self.key(query),
                    vector_sha256=vector_sha256(vector),
                )
            )
        manifest = QueryEmbeddingCollectionManifest(
            generated_at=generated_at or datetime.now(UTC),
            git_commit_sha=git_commit_sha,
            cache_version=self.cache_version,
            dataset_version=dataset_version,
            dataset_manifest_sha256=dataset_manifest_sha256,
            dataset_file_sha256=dataset_file_sha256,
            model=self.config.embedding_model,
            dimensions=self.config.dimensions,
            question_count=len(queries),
            questions=tuple(references),
        )
        write_json_exclusive(path, manifest)
        return self.load_collection(
            queries=queries,
            dataset_version=dataset_version,
            dataset_manifest_sha256=dataset_manifest_sha256,
            dataset_file_sha256=dataset_file_sha256,
        )


class CachedQueryEmbeddingProvider:
    """Apply explicit disabled/read-only/read-write behavior around an embedding provider."""

    def __init__(
        self,
        *,
        config: DenseConfig,
        mode: QueryEmbeddingCacheMode,
        cache: QueryEmbeddingCache | None = None,
        delegate: EmbeddingProvider | None = None,
        observer: Callable[[QueryEmbeddingObservation], None] | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if mode == "disabled" and delegate is None:
            raise QueryEmbeddingCacheError("disabled cache mode requires a live provider")
        if mode != "disabled" and cache is None:
            raise QueryEmbeddingCacheError(f"{mode} cache mode requires a cache")
        if mode == "read-write" and delegate is None:
            raise QueryEmbeddingCacheError("read-write cache mode requires a live provider")
        if cache is not None and cache.config.embedding_model != config.embedding_model:
            raise QueryEmbeddingCacheError("query cache/provider model mismatch")
        if cache is not None and cache.config.dimensions != config.dimensions:
            raise QueryEmbeddingCacheError("query cache/provider dimension mismatch")
        if delegate is not None:
            if delegate.model != config.embedding_model:
                raise QueryEmbeddingCacheError("query cache/provider model mismatch")
            if delegate.dimensions != config.dimensions:
                raise QueryEmbeddingCacheError("query cache/provider dimension mismatch")
        self.config = config
        self.mode = mode
        self.cache = cache
        self.delegate = delegate
        self._observer = observer
        self._clock = clock

    @property
    def model(self) -> str:
        return self.config.embedding_model

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def embed_documents(self, chunks: Sequence[ParagraphChunk]) -> FloatMatrix:
        if self.delegate is None:
            raise QueryEmbeddingCacheError("read-only query cache cannot embed documents")
        return self.delegate.embed_documents(chunks)

    def embed_query(self, query: RetrievalQuery) -> FloatVector:
        started = self._clock()
        cache_key = query_embedding_cache_key(
            query,
            model=self.model,
            dimensions=self.dimensions,
            cache_version=self.cache.cache_version if self.cache is not None else "disabled",
        )
        loaded = (
            self.cache.load(query)
            if self.cache is not None and self.mode != "disabled"
            else None
        )
        if loaded is not None:
            vector, _ = loaded
            self._observe(query, vector, source="cache", elapsed=self._clock() - started)
            return vector
        if self.mode == "read-only":
            raise QueryEmbeddingCacheError(f"query embedding cache miss for {cache_key}")
        if self.delegate is None:  # Protected by constructor validation.
            raise QueryEmbeddingCacheError("live embedding provider is unavailable")
        vector = self.delegate.embed_query(query)
        if self.cache is not None:
            self.cache.store(query, vector)
        self._observe(query, vector, source="live", elapsed=self._clock() - started)
        return vector

    def _observe(
        self,
        query: RetrievalQuery,
        vector: FloatVector,
        *,
        source: Literal["cache", "live"],
        elapsed: float,
    ) -> None:
        if self._observer is None:
            return
        cache_version = self.cache.cache_version if self.cache is not None else None
        key = query_embedding_cache_key(
            query,
            model=self.model,
            dimensions=self.dimensions,
            cache_version=cache_version or "disabled",
        )
        self._observer(
            QueryEmbeddingObservation(
                cache_mode=self.mode,
                source=source,
                cache_version=cache_version,
                cache_key=key,
                query_text_sha256=query_text_sha256(query),
                model=self.model,
                dimensions=self.dimensions,
                vector_sha256=vector_sha256(vector),
                cache_entry=self.cache.relative_entry(key) if self.cache is not None else None,
                latency_ms=max(0.0, elapsed * 1000),
            )
        )
