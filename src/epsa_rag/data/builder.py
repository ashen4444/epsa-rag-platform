"""Deterministic selection and global corpus construction."""

from __future__ import annotations

from collections.abc import Sequence

from epsa_rag.core.exceptions import SourceValidationError
from epsa_rag.core.ids import make_content_id, make_evidence_unit_id, stable_digest
from epsa_rag.core.models import ParagraphChunk, Sentence
from epsa_rag.data.config import PreparationConfig
from epsa_rag.data.models import (
    BenchmarkExample,
    EvaluationLabels,
    HotPotQASourceExample,
    PreparedArtifacts,
    QuestionInput,
    SupportingFactLabel,
)


def select_examples(
    examples: Sequence[HotPotQASourceExample],
    config: PreparationConfig,
) -> tuple[HotPotQASourceExample, ...]:
    """Select questions by a stable seed-and-ID hash rank."""

    if config.question_count > len(examples):
        raise SourceValidationError(
            f"requested {config.question_count} questions from a source containing {len(examples)}"
        )
    seed = str(config.selection_seed)
    ranked = sorted(
        examples,
        key=lambda example: (stable_digest(seed, example.question_id), example.question_id),
    )
    return tuple(ranked[: config.question_count])


def build_artifacts(examples: Sequence[HotPotQASourceExample]) -> PreparedArtifacts:
    """Build separated benchmark records and one deterministic global corpus."""

    corpus_by_id: dict[str, ParagraphChunk] = {}
    benchmark_examples: list[BenchmarkExample] = []
    candidate_count = 0

    for example in examples:
        local_chunks: dict[str, ParagraphChunk] = {}
        for title, sentence_texts in example.context:
            candidate_count += 1
            chunk = _make_chunk(title, sentence_texts)
            existing = corpus_by_id.get(chunk.chunk_id)
            if existing is not None and existing != chunk:
                raise SourceValidationError(f"chunk hash collision detected: {chunk.chunk_id}")
            corpus_by_id[chunk.chunk_id] = chunk
            local_chunks[title] = chunk

        supporting_labels = tuple(
            _make_supporting_label(local_chunks, title, sentence_index)
            for title, sentence_index in example.supporting_facts
        )
        benchmark_examples.append(
            BenchmarkExample(
                inference=QuestionInput(question_id=example.question_id, text=example.question),
                evaluation=EvaluationLabels(
                    answer=example.answer,
                    question_type=example.question_type,
                    difficulty=example.level,
                    supporting_facts=supporting_labels,
                ),
            )
        )

    corpus = tuple(corpus_by_id[chunk_id] for chunk_id in sorted(corpus_by_id))
    return PreparedArtifacts(
        examples=tuple(benchmark_examples),
        corpus=corpus,
        candidate_paragraph_count=candidate_count,
        duplicate_paragraph_count=candidate_count - len(corpus),
    )


def _make_chunk(title: str, sentence_texts: tuple[str, ...]) -> ParagraphChunk:
    chunk_id = make_content_id("chunk", title, *sentence_texts)
    sentences = tuple(
        Sentence(index=index, text=text) for index, text in enumerate(sentence_texts)
    )
    return ParagraphChunk(
        chunk_id=chunk_id,
        title=title,
        paragraph_text="".join(sentence_texts),
        sentences=sentences,
    )


def _make_supporting_label(
    local_chunks: dict[str, ParagraphChunk],
    title: str,
    sentence_index: int,
) -> SupportingFactLabel:
    try:
        chunk = local_chunks[title]
    except KeyError as error:
        raise SourceValidationError(f"supporting title {title!r} was not built") from error
    return SupportingFactLabel(
        chunk_id=chunk.chunk_id,
        title=title,
        sentence_index=sentence_index,
        evidence_unit_id=make_evidence_unit_id(chunk.chunk_id, sentence_index),
    )

