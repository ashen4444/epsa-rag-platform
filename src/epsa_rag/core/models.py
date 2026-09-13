"""Backend-independent structural and retrieval data contracts."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from epsa_rag.core.ids import Identifier


def validate_non_empty_text(value: str) -> str:
    """Reject blank text without altering source whitespace."""

    if not value.strip():
        raise ValueError("text must not be blank")
    return value


NonEmptyText = Annotated[str, Field(min_length=1), AfterValidator(validate_non_empty_text)]


class ContractModel(BaseModel):
    """Strict, immutable base for cross-component data contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Sentence(ContractModel):
    """A sentence retaining its native zero-based paragraph index."""

    index: Annotated[int, Field(ge=0)]
    text: str


class ParagraphChunk(ContractModel):
    """Inference-safe paragraph structure with no gold evaluation labels."""

    chunk_id: Identifier
    title: NonEmptyText
    paragraph_text: NonEmptyText
    sentences: Annotated[tuple[Sentence, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_sentence_order(self) -> ParagraphChunk:
        """Require sentence indices to be unique and strictly increasing."""

        indices = [sentence.index for sentence in self.sentences]
        if indices != list(range(len(self.sentences))):
            raise ValueError("sentence indices must be contiguous and zero-based")
        if self.paragraph_text != "".join(sentence.text for sentence in self.sentences):
            raise ValueError("paragraph_text must equal the exact ordered sentence concatenation")
        return self


class RetrievalQuery(ContractModel):
    """Backend-independent input to a future retriever implementation."""

    text: NonEmptyText
    question_id: Identifier | None = None


class RankedParagraphChunk(ContractModel):
    """Canonical ranked result shared by future retrieval backends and EPSA."""

    chunk: ParagraphChunk
    rank: Annotated[int, Field(ge=1)]
    score: float
    source_scores: dict[Identifier, float] = Field(default_factory=dict)
    source_ranks: dict[Identifier, Annotated[int, Field(ge=1)]] = Field(
        default_factory=dict
    )

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        """Reject non-finite fused scores."""

        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    @field_validator("source_scores")
    @classmethod
    def validate_source_scores(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject non-finite backend scores and detach caller-owned dictionaries."""

        if any(not math.isfinite(score) for score in value.values()):
            raise ValueError("source scores must be finite")
        return value.copy()

    @field_validator("source_ranks")
    @classmethod
    def detach_source_ranks(cls, value: dict[str, int]) -> dict[str, int]:
        """Detach caller-owned rank dictionaries."""

        return value.copy()
