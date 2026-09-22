"""Focused behavioral tests for reconstructed EPSA Component 09."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from epsa_rag.core.exceptions import NextHopQueryGenerationError
from epsa_rag.core.models import Sentence
from epsa_rag.epsa.chunk_analysis import RuleBasedCandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.chunk_analysis.models import CanonicalRetrievedChunk
from epsa_rag.epsa.evidence_graph import EvidenceGraphBuilderV1
from epsa_rag.epsa.evidence_path_search import EvidencePathSearcherV1
from epsa_rag.epsa.evidence_scoring import RuleBasedEvidenceScorerV1
from epsa_rag.epsa.evidence_units import RuleBasedEvidenceUnitExtractor
from epsa_rag.epsa.next_hop_query import (
    HistoricalAdaptedNextHopQueryGeneratorConfig,
    NextHopQueryGeneratorProtocol,
    NextHopQuerySource,
    NextHopQueryType,
    QueryReasonCode,
    ReconstructedNextHopQueryGeneratorConfig,
    RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1,
    RuleBasedNextHopQueryGeneratorReconstructedV1,
)
from epsa_rag.epsa.question_analysis import QuestionType, RelationHint, RuleBasedQuestionAnalyzer
from epsa_rag.epsa.sufficiency_decision import (
    DecisionReasonCode,
    RuleBasedSufficiencyEngineV1,
)
from epsa_rag.instrumentation import InMemoryInstrumentationSink, TraceContext


def _bridge_inputs():
    analysis = RuleBasedQuestionAnalyzer().analyze("Where was the director of Inception born?")
    first_text = "Inception was directed by Christopher Nolan."
    first_chunk = CanonicalRetrievedChunk(
        chunk_id="inception",
        doc_title="Inception",
        paragraph_index=0,
        paragraph_text=first_text,
        chunk_text=first_text,
        sentences=(Sentence(index=0, text=first_text),),
        retrieval_rank=1,
        retrieval_score=0.9,
        source_question_id="component09-bridge",
    )
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(first_chunk, analysis)
    first = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, first_chunk, analysis)[0]
    first = first.model_copy(
        update={
            "entities": ("Inception", "Christopher Nolan"),
            "relation_hints": ("directed",),
            "question_entity_overlap": ("Inception",),
        }
    )
    second_text = "Christopher Nolan was born in London."
    resolution = first.metadata.resolution.model_copy(
        update={"original_text": second_text, "resolved_text": second_text, "changed": False}
    )
    second = first.model_copy(
        update={
            "evidence_unit_id": "christopher-nolan::s0",
            "chunk_id": "christopher-nolan",
            "doc_title": "Christopher Nolan",
            "sentence_text": second_text,
            "resolved_text": second_text,
            "entities": ("Christopher Nolan", "London"),
            "relation_hints": ("born",),
            "question_entity_overlap": (),
            "retrieval_rank": 2,
            "retrieval_score": 0.8,
            "metadata": first.metadata.model_copy(update={"resolution": resolution}),
        }
    )
    scored = RuleBasedEvidenceScorerV1().score_many((first, second), analysis)
    graph = EvidenceGraphBuilderV1().build(analysis, scored)
    paths = tuple(EvidencePathSearcherV1().search_paths(graph, analysis))
    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, paths)
    return analysis, graph, paths, decision


def _factoid_inputs():
    analysis = RuleBasedQuestionAnalyzer().analyze("Who directed Inception?")
    text = "Inception was directed by Christopher Nolan."
    chunk = CanonicalRetrievedChunk(
        chunk_id="inception-factoid",
        doc_title="Inception",
        paragraph_index=0,
        paragraph_text=text,
        chunk_text=text,
        sentences=(Sentence(index=0, text=text),),
        retrieval_rank=1,
        retrieval_score=0.9,
        source_question_id="component09-factoid",
    )
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    unit = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, chunk, analysis)[0]
    scored = RuleBasedEvidenceScorerV1().score_many((unit,), analysis)
    graph = EvidenceGraphBuilderV1().build(analysis, scored)
    paths = tuple(EvidencePathSearcherV1().search_paths(graph, analysis))
    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, paths)
    return analysis, graph, paths, decision


def _insufficient(decision):
    return decision.model_copy(
        update={
            "sufficient": False,
            "missing_evidence": "Additional evidence is required.",
            "decision_reason": DecisionReasonCode.FACTOID_RULES_UNSATISFIED,
        }
    )


def test_bridge_query_is_deterministic_and_preserves_provenance() -> None:
    analysis, graph, paths, decision = _bridge_inputs()
    partial = next(path for path in paths if path.metadata.path_kind == "bridge_candidate")
    partial = partial.model_copy(update={"relation_chain": ("directed",)})
    decision = _insufficient(RuleBasedSufficiencyEngineV1().decide(analysis, graph, (partial,)))
    generator = RuleBasedNextHopQueryGeneratorReconstructedV1()

    first = generator.generate(analysis, decision, graph, (partial,))
    second = generator.generate(analysis, decision, graph, (partial,))

    assert isinstance(generator, NextHopQueryGeneratorProtocol)
    assert first == second
    assert first.query == "Christopher Nolan born"
    assert first.query_type is NextHopQueryType.BRIDGE_ENTITY_RELATION
    assert first.source is NextHopQuerySource.PATH_BRIDGE_ENTITY
    assert first.target_entity == "Christopher Nolan"
    assert first.missing_relation == "born"
    assert first.expected_answer_type is analysis.expected_answer_type
    assert first.confidence == 0.65
    assert first.metadata.reason_code is QueryReasonCode.BRIDGE_ENTITY_RELATION
    assert first.metadata.selected_path_id == partial.path_id
    assert first.metadata.candidate_path_ids == (partial.path_id,)
    assert first.metadata.source_graph == graph.metadata
    assert first.metadata.source_sufficiency_decision == decision.metadata
    assert first.metadata.version == "research_v1_reconstructed"


def test_seed_fallback_and_bridge_priority_are_stable() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    query = RuleBasedNextHopQueryGeneratorReconstructedV1().generate(
        analysis, _insufficient(decision), graph, paths
    )

    assert query.query == "Inception directed"
    assert query.query_type is NextHopQueryType.SEED_ENTITY_RELATION
    assert query.source is NextHopQuerySource.QUESTION_SEED
    assert query.confidence == 0.40
    assert query.metadata.reason_code is QueryReasonCode.SEED_ENTITY_RELATION


def test_comparison_prefers_the_unresolved_target() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze("Which is longer, River A or River B?")
    relation_start = analysis.normalized_question.index("longer")
    relation = RelationHint(
        relation="length",
        matched_text="longer",
        start=relation_start,
        end=relation_start + len("longer"),
        confidence=0.8,
    )
    analysis = analysis.model_copy(
        update={
            "question_type": QuestionType.COMPARISON,
            "comparison_targets": analysis.seed_entities,
            "required_relation_hints": (relation,),
        }
    )
    _, _, factoid_paths, _ = _factoid_inputs()
    base_path = factoid_paths[0]
    graph = EvidenceGraphBuilderV1().build(analysis, ())
    comparison_metadata = base_path.metadata.model_copy(
        update={
            "path_kind": "comparison_target_partial",
            "source_graph": graph.metadata,
            "does_not_compare_values_yet": True,
            "comparison_target": analysis.comparison_targets[0].text,
            "bridge_entity": None,
        }
    )
    path = base_path.model_copy(
        update={
            "path_id": "river-a-partial",
            "question_type": QuestionType.COMPARISON,
            "answer_type": analysis.expected_answer_type,
            "relation_chain": ("length",),
            "metadata": comparison_metadata,
        }
    )
    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, (path,))

    query = RuleBasedNextHopQueryGeneratorReconstructedV1().generate(
        analysis, decision, graph, (path,)
    )

    assert query.query == "River B length"
    assert query.query_type is NextHopQueryType.COMPARISON_TARGET_RELATION
    assert query.source is NextHopQuerySource.PATH_COMPARISON_TARGET
    assert query.confidence == 0.50
    assert query.metadata.reason_code is QueryReasonCode.COMPARISON_TARGET_RELATION


def test_sufficient_or_pathless_decisions_return_explicit_no_query() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    generator = RuleBasedNextHopQueryGeneratorReconstructedV1()

    sufficient = generator.generate(analysis, decision, graph, paths)
    pathless = generator.generate(
        analysis,
        RuleBasedSufficiencyEngineV1().decide(analysis, graph, ()),
        graph,
        (),
    )

    assert sufficient.query is None
    assert sufficient.query_type is NextHopQueryType.NO_QUERY
    assert sufficient.source is NextHopQuerySource.NO_QUERY
    assert sufficient.confidence == 0.0
    assert sufficient.metadata.reason_code is QueryReasonCode.SUFFICIENT_DECISION
    assert pathless.query is None
    assert pathless.metadata.reason_code is QueryReasonCode.NO_CANDIDATE_PATHS


def test_contracts_are_immutable_and_exclude_runtime_gold_fields() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    query = RuleBasedNextHopQueryGeneratorReconstructedV1().generate(
        analysis, _insufficient(decision), graph, paths
    )

    with pytest.raises(ValidationError):
        query.query = "replacement"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ReconstructedNextHopQueryGeneratorConfig(seed_query_confidence=0.5)
    serialized = query.model_dump_json()
    assert "supporting_facts" not in serialized
    assert "gold_answer" not in serialized
    assert "evaluation_only" not in serialized


def test_invalid_inputs_raise_component_error_and_emit_failure_event() -> None:
    _analysis, graph, paths, decision = _factoid_inputs()
    sink = InMemoryInstrumentationSink()
    generator = RuleBasedNextHopQueryGeneratorReconstructedV1(instrumentation_sink=sink)
    trace = TraceContext.start(run_id="component09-test", question_id="component09-q")

    with pytest.raises(NextHopQueryGenerationError, match="question analysis"):
        generator.generate(object(), decision, graph, paths, trace_context=trace)  # type: ignore[arg-type]

    assert [event.event_type for event in sink.events] == ["epsa.next_hop_query.failed"]


def test_completed_instrumentation_contains_only_safe_operational_diagnostics() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    sink = InMemoryInstrumentationSink()
    generator = RuleBasedNextHopQueryGeneratorReconstructedV1(instrumentation_sink=sink)

    generator.generate(
        analysis,
        _insufficient(decision),
        graph,
        paths,
        trace_context=TraceContext.start(run_id="component09-test", question_id="component09-q"),
    )

    event = sink.events[0]
    assert event.event_type == "epsa.next_hop_query.completed"
    assert event.source_version == "research_v1_reconstructed"
    assert set(event.payload) == {
        "latency_ms",
        "query_available",
        "query_type",
        "source",
        "reason_code",
        "confidence",
        "selected_path_present",
    }


def test_recovered_historical_policy_expands_relation_and_uses_historical_confidence() -> None:
    analysis, graph, paths, decision = _bridge_inputs()
    partial = next(path for path in paths if path.metadata.path_kind == "bridge_candidate")
    partial = partial.model_copy(update={"relation_chain": ("directed",)})
    decision = _insufficient(RuleBasedSufficiencyEngineV1().decide(analysis, graph, (partial,)))

    query = RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1().generate(
        analysis, decision, graph, (partial,)
    )

    assert query.query == "Christopher Nolan born birthplace"
    assert query.query_type is NextHopQueryType.BRIDGE_COMPLETION
    assert query.source is NextHopQuerySource.QUESTION_ANALYSIS_AND_SUFFICIENCY_DECISION
    assert query.confidence == 0.9
    assert query.metadata.reason_code is QueryReasonCode.HISTORICAL_BRIDGE_COMPLETION
    assert query.metadata.version == "research_v1_historical_adapted"
    assert query.metadata.historical_rules_status == (
        "recovered_historical_policy_adapted_to_current_contracts"
    )


def test_recovered_historical_policy_retains_seed_fallback_without_a_path() -> None:
    analysis, graph, _paths, _decision = _factoid_inputs()
    pathless = _insufficient(RuleBasedSufficiencyEngineV1().decide(analysis, graph, ()))

    query = RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1().generate(
        analysis, pathless, graph, ()
    )

    assert query.query == "Inception directed director person"
    assert query.query_type is NextHopQueryType.RELATION_COMPLETION
    assert query.metadata.selected_path_id is None
    assert query.confidence == 0.85


def test_historical_configuration_is_immutable() -> None:
    config = HistoricalAdaptedNextHopQueryGeneratorConfig()

    with pytest.raises(ValidationError):
        config.mode = "research_v1_reconstructed"  # type: ignore[misc]
