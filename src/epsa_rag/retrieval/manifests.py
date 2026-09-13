"""Reproducibility manifests for persisted retrieval indexes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, NonEmptyText
from epsa_rag.data.manifests import ArtifactFile
from epsa_rag.retrieval.config import BM25Config, DenseConfig


class DependencyVersion(ContractModel):
    """One runtime dependency relevant to index reproducibility."""

    name: Identifier
    version: NonEmptyText


class RetrievalIndexManifest(ContractModel):
    """Integrity and provenance for one immutable Phase 3 index."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_type: Literal["retrieval-index"] = "retrieval-index"
    index_kind: Literal["bm25", "dense"]
    index_version: Identifier
    backend: NonEmptyText
    generated_at: datetime
    generator: Literal["epsa-rag-phase3"] = "epsa-rag-phase3"
    generator_version: Identifier = "hybrid-retrieval-index-v1"
    corpus_version: Identifier
    corpus_manifest_sha256: Identifier
    corpus_file_sha256: Identifier
    chunk_count: int = Field(ge=1)
    chunk_id_order_sha256: Identifier
    configuration: BM25Config | DenseConfig
    configuration_fingerprint: Identifier
    dependencies: tuple[DependencyVersion, ...]
    files: tuple[ArtifactFile, ...]

    @field_validator("generated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Require an unambiguous index creation instant."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value
