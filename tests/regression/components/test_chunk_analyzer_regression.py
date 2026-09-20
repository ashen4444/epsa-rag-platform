"""Historical rule behavior and its documented limitations stay explicit."""

from __future__ import annotations

from epsa_rag.epsa.chunk_analysis import (
    ChunkAnalyzerConfig,
    RuleBasedCandidateChunkEvidenceAnalyzer,
)
from epsa_rag.epsa.question_analysis import RuleBasedQuestionAnalyzer


def _analyze(question: str, title: str, paragraph: str) -> object:
    return RuleBasedCandidateChunkEvidenceAnalyzer().analyze(
        {"chunk_id": "chunk:regression", "title": title, "paragraph_text": paragraph},
        RuleBasedQuestionAnalyzer().analyze(question),
    )


def test_york_new_york_partial_match_is_documented_legacy_behavior() -> None:
    evidence = _analyze("Where was York founded?", "New York", "New York was founded in 1624.")
    assert evidence.is_title_match  # type: ignore[attr-defined]
    assert "New York" in evidence.question_entity_overlap  # type: ignore[attr-defined]


def test_bare_in_inception_triggers_legacy_located_hint_in_chunk_only() -> None:
    evidence = _analyze(
        "Which actor appeared in Inception?", "Actor",
        "The actor appeared in Inception."
    )
    assert "located" in {hint.relation for hint in evidence.relation_hints}  # type: ignore[attr-defined]


def test_title_christopher_nolan_excludes_true_bridge_and_can_admit_london() -> None:
    evidence = _analyze(
        "Where was the director of Inception born?", "Christopher Nolan",
        "Christopher Nolan directed Inception. Christopher Nolan was born in London."
    )
    bridges = {item.text for item in evidence.potential_bridge_entities}  # type: ignore[attr-defined]
    assert "Christopher Nolan" not in bridges
    assert "London" in bridges
    decisions = {item.entity: item.reason for item in evidence.metadata.bridge_decisions}  # type: ignore[attr-defined]
    assert decisions["Christopher Nolan"] == "document_title"
    assert decisions["London"] == "candidate"


def test_director_directed_equivalence_is_retained() -> None:
    evidence = _analyze(
        "Who was the director of Inception?", "Inception",
        "Inception was directed by Christopher Nolan."
    )
    assert "director" in evidence.question_token_overlap  # type: ignore[attr-defined]


def test_v1_frozen_configuration_and_legacy_span_shape_remain_stable() -> None:
    analyzer = RuleBasedCandidateChunkEvidenceAnalyzer()
    evidence = analyzer.analyze(
        {"chunk_id": "chunk:v1", "title": "Inception",
         "paragraph_text": "Inception was directed by Christopher Nolan in 1970."}
    )

    assert ChunkAnalyzerConfig().fingerprint() == (
        "90fcf61b3ab785fee467b7eb0d4b187771b8359cdae9cef6ff65cd01ecde7faa"
    )
    assert evidence.metadata.version == "rule_based_v1"
    assert "span_scope" not in evidence.model_dump_json()
    assert any(item.relation == "directed" for item in evidence.relation_hints)
