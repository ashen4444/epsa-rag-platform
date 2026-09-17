"""Immutable manifest contracts for frozen dataset and corpus versions."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, NonEmptyText
from epsa_rag.data.config import PreparationConfiguration


class ArtifactFile(ContractModel):
    """Integrity metadata for one generated file."""

    relative_path: NonEmptyText
    sha256: Identifier
    byte_count: int = Field(ge=0)
    record_count: int = Field(ge=0)


class SourceManifest(ContractModel):
    """Identity and integrity of the downloaded upstream source."""

    dataset: Literal["HotPotQA"]
    configuration: Literal["distractor"]
    split: Literal["dev", "train"]
    uri: NonEmptyText
    retrieved_from_uri: NonEmptyText
    filename: NonEmptyText
    sha256: Identifier
    byte_count: int = Field(ge=0)
    record_count: int = Field(ge=0)
    fully_validated_record_count: int = Field(ge=0)
    invalid_unselected_question_ids: tuple[Identifier, ...] = ()
    license: Literal["CC BY-SA 4.0"] = "CC BY-SA 4.0"


class GenerationManifest(ContractModel):
    """Generator identity independent of future experiment storage."""

    generator: Literal["epsa-rag-phase2"] = "epsa-rag-phase2"
    generator_version: Literal["hotpotqa-preparation-v1", "hotpotqa-preparation-v2"] = (
        "hotpotqa-preparation-v1"
    )
    generated_at: datetime
    configuration_fingerprint: Identifier

    @field_validator("generated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Require an unambiguous creation instant."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value


class DatasetManifest(ContractModel):
    """Complete provenance for the frozen question benchmark."""

    schema_version: Literal["1.0", "1.1"] = "1.0"
    artifact_type: Literal["dataset"] = "dataset"
    version: Identifier
    source: SourceManifest
    generation: GenerationManifest
    configuration: PreparationConfiguration
    source_question_count: int = Field(ge=0)
    eligible_question_count: int | None = Field(default=None, ge=0)
    selected_question_count: int = Field(ge=0)
    question_ids: tuple[Identifier, ...]
    files: tuple[ArtifactFile, ...]


class CorpusManifest(ContractModel):
    """Complete provenance for the frozen global paragraph corpus."""

    schema_version: Literal["1.0", "1.1"] = "1.0"
    artifact_type: Literal["corpus"] = "corpus"
    version: Identifier
    source_dataset_version: Identifier
    source: SourceManifest
    generation: GenerationManifest
    configuration: PreparationConfiguration
    candidate_paragraph_count: int = Field(ge=0)
    unique_paragraph_count: int = Field(ge=0)
    duplicate_paragraph_count: int = Field(ge=0)
    chunk_ids: tuple[Identifier, ...]
    files: tuple[ArtifactFile, ...]
