"""Focused Component 02 behavior and source-boundary tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from epsa_rag.core.exceptions import ChunkAnalysisError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.epsa.chunk_analysis import (
    CandidateChunkEvidence,
    CanonicalRetrievedChunk,
    ChunkAnalyzerConfig,
    RuleBasedCandidateChunkEvidenceAnalyzer,
)
from epsa_rag.epsa.chunk_analysis.chunk_adapter import adapt_retrieved_chunk
from epsa_rag.epsa.chunk_analysis.protocols import ChunkAnalyzerProtocol
from epsa_rag.epsa.question_analysis import RuleBasedQuestionAnalyzer
from epsa_rag.instrumentation import InMemoryInstrumentationSink, TraceContext

QUESTION = "Where was the director of Inception born?"
PARAGRAPH = (
    "Inception is a 2010 science fiction film directed by Christopher Nolan. "
    "Nolan was born in London."
)


def _chunk(**changes: object) -> dict[str, object]:
    item: dict[str, object] = {
        "chunk_id": "chunk:inception",
        "doc_title": "Inception",
        "paragraph_index": 0,
        "paragraph_text": PARAGRAPH,
        "sentences": [{"sentence_id": 0, "text": PARAGRAPH}],
        "question_id": "q-1",
        "rank": 2,
        "fusion_score": 0.42,
        "is_supporting_doc": True,
        "gold_answer": "SECRET GOLD",
    }
    item.update(changes)
    return item


def _analysis() -> object:
    return RuleBasedQuestionAnalyzer().analyze(QUESTION)


def test_features_provenance_and_source_text_are_structured_and_immutable() -> None:
    analyzer = RuleBasedCandidateChunkEvidenceAnalyzer()
    evidence = analyzer.analyze(_chunk(), _analysis())  # type: ignore[arg-type]

    assert evidence.chunk_id == "chunk:inception"
    assert evidence.retrieval_rank == 2
    assert evidence.retrieval_score == 0.42
    assert evidence.paragraph_text == PARAGRAPH
    assert evidence.sentences[0].text == PARAGRAPH
    assert evidence.source_question_id == "q-1"
    assert evidence.entities[0].text == "Inception"
    assert evidence.entities[0].source == "doc_title"
    assert [e.text for e in evidence.entities].count("Inception") == 1
    assert "Christopher Nolan" in {e.text for e in evidence.entities}
    assert {hint.relation for hint in evidence.relation_hints} >= {"directed", "born"}
    assert "Inception" in evidence.question_entity_overlap
    assert evidence.is_title_match
    assert "director" in evidence.question_token_overlap
    assert evidence.question_token_overlap_score > 0
    assert "Christopher Nolan" in {e.text for e in evidence.potential_bridge_entities}
    assert evidence.metadata.configuration_fingerprint == analyzer.config.fingerprint()
    assert "SECRET GOLD" not in evidence.model_dump_json()
    assert CandidateChunkEvidence.model_validate_json(evidence.model_dump_json()) == evidence
    with pytest.raises(ValidationError):
        evidence.doc_title = "Changed"  # type: ignore[misc]
    assert isinstance(analyzer, ChunkAnalyzerProtocol)


@dataclass
class _ObjectChunk:
    id: str
    title: str
    text: str
    sentences: list[dict[str, object]]


def test_adapter_resolves_aliases_wrappers_and_explicit_retrieval_provenance() -> None:
    obj = _ObjectChunk(
        id="chunk:object", title="Inception",
        text="Title: Inception\nParagraph: Inception was directed by Christopher Nolan.",
        sentences=[{"sentence_id": 0, "text": "Inception was directed by Christopher Nolan."}],
    )
    wrapped = {"document": obj, "rank": 5, "score": 0.1}
    evidence = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(
        wrapped, _analysis(), retrieval_rank=3, retrieval_score=0.8  # type: ignore[arg-type]
    )

    assert evidence.chunk_id == "chunk:object"
    assert evidence.doc_title == "Inception"
    assert evidence.paragraph_text == "Inception was directed by Christopher Nolan."
    assert evidence.retrieval_rank == 3
    assert evidence.retrieval_score == 0.8
    assert evidence.sentences[0].index == 0


def test_canonical_ranked_chunk_and_missing_optional_text_fields() -> None:
    paragraph = ParagraphChunk(
        chunk_id="chunk:canonical",
        title="Inception",
        paragraph_text="Inception was directed by Christopher Nolan.",
        sentences=(Sentence(index=0, text="Inception was directed by Christopher Nolan."),),
    )
    ranked = RankedParagraphChunk(chunk=paragraph, rank=1, score=0.4)
    canonical = adapt_retrieved_chunk(ranked)
    missing_chunk_text = adapt_retrieved_chunk(
        {"chunk_id": "chunk:minimal", "title": "", "paragraph_text": "A body."}
    )

    assert canonical.sentences == paragraph.sentences
    assert canonical.retrieval_rank == 1
    assert canonical.retrieval_score == 0.4
    assert missing_chunk_text.chunk_text == "A body."
    assert missing_chunk_text.doc_title == ""


def test_canonical_chunk_retains_or_overrides_retrieval_provenance() -> None:
    canonical = CanonicalRetrievedChunk(
        chunk_id="chunk:canonical-input",
        doc_title="Paris",
        paragraph_text="Paris is in France.",
        chunk_text="Title: Paris\nParagraph: Paris is in France.",
        retrieval_rank=2,
        retrieval_score=0.4,
    )

    overridden = adapt_retrieved_chunk(canonical, retrieval_rank=1, retrieval_score=0.7)

    assert overridden.chunk_id == canonical.chunk_id
    assert overridden.retrieval_rank == 1
    assert overridden.retrieval_score == 0.7


def test_adapter_normalizes_string_and_legacy_sentence_objects() -> None:
    adapted = adapt_retrieved_chunk(
        {
            "chunk_id": "chunk:sentence-aliases",
            "paragraph_text": "One. Two.",
            "sentences": ["One. ", {"text": "Two."}],
        }
    )

    assert [(sentence.index, sentence.text) for sentence in adapted.sentences] == [
        (0, "One. "),
        (1, "Two."),
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"rank": 0},
        {"rank": "not-a-rank"},
        {"fusion_score": float("nan")},
        {"fusion_score": float("inf")},
        {"paragraph_index": -1},
    ],
)
def test_invalid_rank_score_and_index_fail_explicitly(changes: dict[str, object]) -> None:
    with pytest.raises(ChunkAnalysisError):
        RuleBasedCandidateChunkEvidenceAnalyzer().analyze(_chunk(**changes))


def test_missing_identity_or_text_fails_and_failure_event_is_emitted() -> None:
    sink = InMemoryInstrumentationSink()
    analyzer = RuleBasedCandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    context = TraceContext.start(run_id="chunk-test", question_id="q-1")
    with pytest.raises(ChunkAnalysisError):
        analyzer.analyze({"title": "Inception", "text": "A body."}, trace_context=context)
    with pytest.raises(ChunkAnalysisError):
        analyzer.analyze({"chunk_id": "chunk:empty", "title": ""}, trace_context=context)

    assert [event.event_type for event in sink.events] == [
        "epsa.chunk_analysis.failed", "epsa.chunk_analysis.failed"
    ]


def test_completed_event_preserves_full_evidence_and_latency() -> None:
    sink = InMemoryInstrumentationSink()
    analyzer = RuleBasedCandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    context = TraceContext.start(run_id="chunk-test", question_id="q-1")
    evidence = analyzer.analyze(_chunk(), _analysis(), trace_context=context)  # type: ignore[arg-type]

    event = sink.events[0]
    assert event.event_type == "epsa.chunk_analysis.completed"
    assert event.context == context
    assert event.payload["evidence"] == evidence.model_dump(mode="json")
    assert event.payload["latency_ms"] >= 0  # type: ignore[operator]


def test_answer_patterns_keep_year_as_date_and_number_and_preserve_generic_entities() -> None:
    evidence = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(
        _chunk(paragraph_text="Christopher Nolan was born in London on January 12, 1970.")
    )
    pairs = {(c.answer_type.value, c.text) for c in evidence.answer_type_candidates}

    assert ("DATE", "1970") in pairs
    assert ("NUMBER", "1970") in pairs
    assert ("DATE", "January 12, 1970") in pairs
    assert ("LOCATION", "London") in pairs
    assert ("ENTITY", "Christopher Nolan") in pairs


def test_config_is_versioned_and_rule_scores_are_not_probabilities() -> None:
    config = ChunkAnalyzerConfig()
    assert config.mode == "rule_based_v1"
    assert len(config.fingerprint()) == 64
    with pytest.raises(ValidationError):
        ChunkAnalyzerConfig(number_score=float("nan"))
