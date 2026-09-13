from __future__ import annotations

import pytest

from epsa_rag.core.exceptions import SourceValidationError
from epsa_rag.data.builder import build_artifacts, select_examples
from epsa_rag.data.config import PreparationConfig
from epsa_rag.data.models import HotPotQASourceExample


def source_example(question_id: str, *, duplicate: bool = False) -> HotPotQASourceExample:
    shared_title = "Shared" if duplicate else f"Title {question_id}"
    shared_sentences = ["First.", " Second."] if duplicate else [f"Text {question_id}."]
    return HotPotQASourceExample.model_validate(
        {
            "_id": question_id,
            "question": f"Question {question_id}?",
            "answer": f"Answer {question_id}",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": [[shared_title, 0]],
            "context": [[shared_title, shared_sentences]],
        }
    )


def test_selection_is_deterministic_and_independent_of_source_order() -> None:
    examples = tuple(source_example(f"q{index}") for index in range(6))
    config = PreparationConfig(question_count=3, selection_seed=17)

    first = select_examples(examples, config)
    second = select_examples(tuple(reversed(examples)), config)

    assert [item.question_id for item in first] == [item.question_id for item in second]
    assert len(first) == 3


def test_selection_rejects_an_oversized_sample() -> None:
    with pytest.raises(SourceValidationError, match="requested 2"):
        select_examples((source_example("q1"),), PreparationConfig(question_count=2))


def test_build_artifacts_deduplicates_exact_chunks_and_separates_gold() -> None:
    artifacts = build_artifacts(
        (source_example("q1", duplicate=True), source_example("q2", duplicate=True))
    )

    assert artifacts.candidate_paragraph_count == 2
    assert artifacts.duplicate_paragraph_count == 1
    assert len(artifacts.corpus) == 1
    assert artifacts.corpus[0].paragraph_text == "First. Second."
    assert artifacts.corpus[0].sentences[1].text == " Second."

    record = artifacts.examples[0]
    assert record.inference.model_dump() == {"question_id": "q1", "text": "Question q1?"}
    assert record.evaluation.answer == "Answer q1"
    label = record.evaluation.supporting_facts[0]
    assert label.evidence_unit_id == f"{label.chunk_id}::s0"

