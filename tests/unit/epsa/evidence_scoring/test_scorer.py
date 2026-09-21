"""Focused frozen-behavior tests for EPSA Component 04."""

from __future__ import annotations

import pytest

from epsa_rag.core.exceptions import EvidenceScoringError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import QuestionInput
from epsa_rag.epsa.chunk_analysis import RuleBasedCandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.chunk_analysis.models import CanonicalRetrievedChunk
from epsa_rag.epsa.evidence_scoring import EvidenceScorerV1Config, RuleBasedEvidenceScorerV1
from epsa_rag.epsa.evidence_scoring.protocols import EvidenceScorerProtocol
from epsa_rag.epsa.evidence_units import RuleBasedEvidenceUnitExtractor
from epsa_rag.epsa.question_analysis import AnswerType, RuleBasedQuestionAnalyzer
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.evaluation.components.evidence_scoring import evaluate_evidence_scoring
from epsa_rag.instrumentation import InMemoryInstrumentationSink, TraceContext


def _inputs():
    analysis = RuleBasedQuestionAnalyzer().analyze("Who directed Inception?")
    sentence = "Inception was directed by Christopher Nolan."
    chunk = CanonicalRetrievedChunk(
        chunk_id="inception",
        doc_title="Inception",
        paragraph_index=0,
        paragraph_text=sentence,
        chunk_text=sentence,
        sentences=(Sentence(index=0, text=sentence),),
        retrieval_rank=1,
        retrieval_score=0.9,
        source_question_id="question-one",
    )
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    unit = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, chunk, analysis)[0]
    return analysis, unit


def test_exact_frozen_formula_breakdown_and_provenance() -> None:
    analysis, unit = _inputs()
    scored = RuleBasedEvidenceScorerV1().score(unit, analysis)

    assert scored.evidence_unit is unit
    assert scored.evidence_unit.evidence_unit_id == "inception::s0"
    assert scored.final_score == 1.0
    assert scored.score_breakdown.model_dump() == {
        "entity_match_score": 1.0,
        "relation_match_score": 1.0,
        "answer_type_match_score": 1.0,
        "token_overlap_score": 1.0,
        "title_match_score": 1.0,
        "retrieval_score_component": 1.0,
        "bridge_entity_score": 1.0,
        "noise_penalty": 0.0,
    }
    assert scored.metadata.version == "research_v1"
    assert len(scored.metadata.configuration_fingerprint) == 64


def test_frozen_feature_edge_cases_and_clipping() -> None:
    analysis, unit = _inputs()
    scorer = RuleBasedEvidenceScorerV1()
    generic = scorer.score(
        unit.model_copy(update={"answer_type_candidates": (AnswerType.DATE,)}), analysis
    )
    assert generic.score_breakdown.answer_type_match_score == 0.0
    entity_analysis = RuleBasedQuestionAnalyzer().analyze("What is Inception?")
    entity = scorer.score(
        unit.model_copy(update={"answer_type_candidates": (AnswerType.DATE,)}), entity_analysis
    )
    assert entity.score_breakdown.answer_type_match_score == 0.6
    fallback = scorer.score(
        unit.model_copy(update={"retrieval_rank": None, "retrieval_score": 0.7}), analysis
    )
    assert fallback.score_breakdown.retrieval_score_component == 0.7
    short_text = "Yes."
    short_resolution = unit.metadata.resolution.model_copy(
        update={"resolved_text": short_text, "changed": True, "reason": "test"}
    )
    short = scorer.score(
        unit.model_copy(
            update={
                "resolved_text": short_text,
                "entities": (),
                "relation_hints": (),
                "metadata": unit.metadata.model_copy(update={"resolution": short_resolution}),
            }
        ),
        analysis,
    )
    assert short.score_breakdown.noise_penalty == 0.35
    assert short.final_score == 0.0


def test_substring_entity_match_and_rank_order_are_frozen_behavior() -> None:
    analysis, unit = _inputs()
    scorer = RuleBasedEvidenceScorerV1()
    resolved_text = "The Inceptionist spoke."
    resolution = unit.metadata.resolution.model_copy(
        update={"resolved_text": resolved_text, "changed": True, "reason": "test"}
    )
    substring = scorer.score(
        unit.model_copy(
            update={
                "entities": (),
                "resolved_text": resolved_text,
                "metadata": unit.metadata.model_copy(update={"resolution": resolution}),
            }
        ),
        analysis,
    )
    assert substring.score_breakdown.entity_match_score == 1.0
    rank_two = scorer.score(
        unit.model_copy(update={"retrieval_rank": 2, "retrieval_score": 0.99}), analysis
    )
    rank_ten = scorer.score(
        unit.model_copy(update={"retrieval_rank": 10, "retrieval_score": 0.99}), analysis
    )
    assert rank_two.score_breakdown.retrieval_score_component == 0.5
    assert rank_ten.score_breakdown.retrieval_score_component == 0.1


def test_order_rank_events_and_strict_boundary() -> None:
    analysis, unit = _inputs()
    sink = InMemoryInstrumentationSink()
    scorer = RuleBasedEvidenceScorerV1(instrumentation_sink=sink)
    second = unit.model_copy(update={"evidence_unit_id": "inception::s1", "sentence_id": 1})
    context = TraceContext.start(run_id="scorer-test", question_id="question-one")
    scored = scorer.score_many((unit, second), analysis, trace_context=context)

    assert isinstance(scorer, EvidenceScorerProtocol)
    assert [item.evidence_unit.evidence_unit_id for item in scored] == [
        "inception::s0",
        "inception::s1",
    ]
    assert [event.event_type for event in sink.events] == [
        "epsa.evidence_scoring.completed",
        "epsa.evidence_scoring.completed",
    ]
    assert "is_supporting_sentence" not in sink.events[0].model_dump_json()
    with pytest.raises(EvidenceScoringError, match="Component 03 contract"):
        scorer.score(object(), analysis, trace_context=context)  # type: ignore[arg-type]
    with pytest.raises(EvidenceScoringError, match="Component 01 contract"):
        scorer.score(unit, object(), trace_context=context)  # type: ignore[arg-type]
    assert sink.events[-1].event_type == "epsa.evidence_scoring.failed"


def test_inference_evaluator_runs_components_01_through_04_without_gold_fields() -> None:
    text = "Inception was directed by Christopher Nolan."
    ranked = RankedParagraphChunk(
        chunk=ParagraphChunk(
            chunk_id="inception-eval",
            title="Inception",
            paragraph_text=text,
            sentences=(Sentence(index=0, text=text),),
        ),
        rank=1,
        score=0.9,
    )
    input_item = InferenceRetrieval(
        question=QuestionInput(question_id="q-eval", text="Who directed Inception?"),
        chunks=(ranked,),
        retriever_version="retriever-v1",
    )
    summary, traces, events = evaluate_evidence_scoring((input_item,), run_id="score-eval")

    assert summary.completed_questions == 1
    assert summary.diagnostics.scored_evidence_units == 1
    assert summary.diagnostics.final_score_histogram == {"one": 1}
    assert traces[0].scored_evidence_units[0].evidence_unit.evidence_unit_id == "inception-eval::s0"
    assert [event.event_type for event in events] == [
        "epsa.question_analysis.completed",
        "epsa.chunk_analysis.completed",
        "epsa.evidence_units.completed",
        "epsa.evidence_scoring.completed",
    ]
    assert "supporting" not in traces[0].model_dump_json()


def test_evaluator_requires_inputs_and_config_rejects_weight_drift() -> None:
    with pytest.raises(ValueError, match="requires inference inputs"):
        evaluate_evidence_scoring((), run_id="score-eval-empty")
    with pytest.raises(ValueError, match="weights must sum"):
        EvidenceScorerV1Config(entity_match_weight=0.24)
