"""Source, benchmark, and evaluation-only data contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, NonEmptyText, ParagraphChunk

NonNegativeIndex = Annotated[int, Field(ge=0)]


class HotPotQASourceExample(BaseModel):
    """Validated representation of one official HotPotQA JSON object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: Identifier = Field(validation_alias=AliasChoices("_id", "question_id"))
    question: NonEmptyText
    answer: NonEmptyText
    question_type: Literal["bridge", "comparison"] = Field(alias="type")
    level: Literal["easy", "medium", "hard"]
    supporting_facts: tuple[tuple[NonEmptyText, NonNegativeIndex], ...]
    context: tuple[tuple[NonEmptyText, tuple[str, ...]], ...]

    @model_validator(mode="after")
    def validate_evidence_references(self) -> HotPotQASourceExample:
        """Ensure every supporting fact resolves to a native source sentence."""

        title_to_sentences: dict[str, tuple[str, ...]] = {}
        for title, sentences in self.context:
            if not sentences:
                raise ValueError(f"context paragraph {title!r} must contain at least one sentence")
            if title in title_to_sentences:
                raise ValueError(f"context title {title!r} appears more than once")
            title_to_sentences[title] = sentences

        if not self.supporting_facts:
            raise ValueError("supporting_facts must not be empty")
        for title, sentence_index in self.supporting_facts:
            referenced_sentences = title_to_sentences.get(title)
            if referenced_sentences is None:
                raise ValueError(f"supporting title {title!r} is absent from context")
            if sentence_index >= len(referenced_sentences):
                raise ValueError(
                    f"supporting sentence index {sentence_index} is out of range for {title!r}"
                )
        return self


class QuestionInput(ContractModel):
    """The complete inference-visible benchmark input."""

    question_id: Identifier
    text: NonEmptyText


class SupportingFactLabel(ContractModel):
    """Evaluation-only supporting sentence provenance."""

    chunk_id: Identifier
    title: NonEmptyText
    sentence_index: NonNegativeIndex
    evidence_unit_id: Identifier


class EvaluationLabels(ContractModel):
    """Gold information that must never be passed into inference features."""

    answer: NonEmptyText
    question_type: Literal["bridge", "comparison"]
    difficulty: Literal["easy", "medium", "hard"]
    supporting_facts: tuple[SupportingFactLabel, ...]


class BenchmarkExample(ContractModel):
    """One benchmark record with an explicit inference/evaluation boundary."""

    inference: QuestionInput
    evaluation: EvaluationLabels


class PreparedArtifacts(ContractModel):
    """In-memory result produced before immutable artifact serialization."""

    examples: tuple[BenchmarkExample, ...]
    corpus: tuple[ParagraphChunk, ...]
    candidate_paragraph_count: int = Field(ge=0)
    duplicate_paragraph_count: int = Field(ge=0)
