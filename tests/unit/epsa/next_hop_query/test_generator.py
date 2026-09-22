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
    NextHopQuery,
    NextHopQueryGeneratorProtocol,
    NextHopQuerySource,
    NextHopQueryType,
    QueryReasonCode,
    ReconstructedNextHopQueryGeneratorConfig,
    RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1,
    RuleBasedNextHopQueryGeneratorReconstructedV1,
    historical_generator,
)
from epsa_rag.epsa.next_hop_query import generator as reconstructed_generator
from epsa_rag.epsa.question_analysis import (
    AnswerType,
    QuestionType,
    RelationHint,
    RuleBasedQuestionAnalyzer,
)
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


def test_historical_helper_rules_cover_relation_fallbacks_and_query_building() -> None:
    config = HistoricalAdaptedNextHopQueryGeneratorConfig()

    assert historical_generator._norm("Born_Place") == "born place"
    assert historical_generator._first_relation_from_text("required relation capital.") == "capital"
    assert historical_generator._first_relation_from_text("The director was born there") == "born"
    assert historical_generator._first_relation_from_text("") is None
    assert (
        historical_generator._choose_missing_relation("", ("born", "directed"), ("born",))
        == "directed"
    )
    assert historical_generator._choose_missing_relation("", ("born",), ("born",)) == "born"
    assert historical_generator._choose_missing_relation("", (), ()) is None
    assert (
        historical_generator._build_query(
            entities=("Inception", "Inception"),
            relation="directed",
            expected_answer_type="PERSON",
            config=config,
        )
        == "Inception directed director person"
    )
    assert (
        historical_generator._build_query(
            entities=(),
            relation=None,
            expected_answer_type="UNKNOWN",
            config=config,
        )
        is None
    )
    assert (
        historical_generator._build_query(
            entities=("Paris",),
            relation="located",
            expected_answer_type="LOCATION",
            config=config,
        )
        == "Paris located location"
    )
    assert historical_generator._confidence(True, True, True, True) == 0.9
    assert historical_generator._confidence(False, False, False, False) == 0.2


def test_historical_helpers_cover_path_and_reason_variants() -> None:
    analysis, _graph, paths, decision = _factoid_inputs()
    path = paths[0]

    assert historical_generator._best_fallback_path(()) is None
    assert historical_generator._best_fallback_path(paths) == path
    assert (
        historical_generator._choose_target_entity(None, ("Inception",), None, QuestionType.FACTOID)
        == "Inception"
    )
    assert (
        historical_generator._choose_target_entity(path, ("Inception",), None, QuestionType.FACTOID)
        is not None
    )
    assert historical_generator._reason_text(_insufficient(decision)).startswith(
        "Generated from missing evidence:"
    )
    missing = _insufficient(decision).model_copy(update={"missing_evidence": None})
    assert historical_generator._reason_text(missing).startswith(
        "Generated from insufficient decision:"
    )
    assert reconstructed_generator._usable_text(" value ")
    assert not reconstructed_generator._usable_text(None)
    assert reconstructed_generator._normalise("A  VALUE") == "a value"
    assert reconstructed_generator._first_usable_entity(analysis.seed_entities) is not None


def test_generators_cover_no_query_and_failure_paths() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    sink = InMemoryInstrumentationSink()
    historical = RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1(instrumentation_sink=sink)
    trace = TraceContext.start(run_id="component09-test", question_id="component09-q")
    sufficient = historical.generate(analysis, decision, graph, paths, trace_context=trace)

    assert sufficient.query is None
    assert sufficient.source is NextHopQuerySource.SUFFICIENCY_DECISION
    with pytest.raises(NextHopQueryGenerationError):
        historical.generate(  # type: ignore[arg-type]
            analysis, decision, object(), paths, trace_context=trace
        )
    assert sink.events[-1].event_type == "epsa.next_hop_query.failed"


def test_reconstructed_output_helpers_and_contract_validation_failures() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    generator = RuleBasedNextHopQueryGeneratorReconstructedV1()

    no_query = generator._no_query(
        analysis,
        _insufficient(decision),
        graph,
        paths,
        QueryReasonCode.NO_GROUNDED_TARGET,
        "no grounded target",
    )
    assert no_query.query is None
    query = generator._query(
        analysis,
        _insufficient(decision),
        graph,
        paths,
        target=" Inception ",
        relation=" directed ",
        path=paths[0],
        query_type=NextHopQueryType.SEED_ENTITY_RELATION,
        source=NextHopQuerySource.QUESTION_SEED,
        reason_code=QueryReasonCode.SEED_ENTITY_RELATION,
        reason="test",
        confidence=0.4,
    )
    assert query.query == "Inception directed"
    for invalid in (object(),):
        with pytest.raises(NextHopQueryGenerationError):
            generator._validate_inputs(invalid, decision, graph, paths)
        with pytest.raises(NextHopQueryGenerationError):
            generator._validate_inputs(analysis, invalid, graph, paths)
        with pytest.raises(NextHopQueryGenerationError):
            generator._validate_inputs(analysis, decision, invalid, paths)
    with pytest.raises(NextHopQueryGenerationError):
        generator._validate_inputs(analysis, decision, graph, "not-paths")


def test_historical_special_and_no_signal_branches() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    decision = _insufficient(decision)
    generator = RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1()

    unavailable = generator._special_query(
        analysis,
        decision,
        graph,
        paths,
        paths[0],
        entities=(),
        relation=None,
        expected_answer_type="UNKNOWN",
        include_answer_type=False,
        query_type=NextHopQueryType.YES_NO_RELATION_CHECK,
        reason_code=QueryReasonCode.HISTORICAL_YES_NO_RELATION_CHECK,
        unavailable="unavailable",
    )
    assert unavailable.query is None
    special = generator._special_query(
        analysis,
        decision,
        graph,
        paths,
        paths[0],
        entities=("Inception",),
        relation="directed",
        expected_answer_type="PERSON",
        include_answer_type=True,
        query_type=NextHopQueryType.YES_NO_RELATION_CHECK,
        reason_code=QueryReasonCode.HISTORICAL_YES_NO_RELATION_CHECK,
        unavailable="unavailable",
    )
    assert special.query == "Inception directed director person"
    no_signal_analysis = analysis.model_copy(
        update={
            "seed_entities": (),
            "required_relation_hints": (),
            "expected_answer_type": AnswerType.UNKNOWN,
        }
    )
    no_signal = generator._generate(
        no_signal_analysis,
        decision.model_copy(update={"best_path": None}),
        graph,
        (),
    )
    assert no_signal.query is None


def test_reconstructed_validation_rejects_mismatched_component_provenance() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    validate = RuleBasedNextHopQueryGeneratorReconstructedV1._validate_inputs
    mismatched_graph = graph.model_copy(update={"question_type": QuestionType.BRIDGE})
    with pytest.raises(NextHopQueryGenerationError, match="question type"):
        validate(analysis, decision, mismatched_graph, paths)
    mismatched_answer_graph = graph.model_copy(
        update={
            "metadata": graph.metadata.model_copy(update={"expected_answer_type": AnswerType.DATE})
        }
    )
    with pytest.raises(NextHopQueryGenerationError, match="answer type"):
        validate(analysis, decision, mismatched_answer_graph, paths)
    mismatched_decision = decision.model_copy(update={"question_type": QuestionType.BRIDGE})
    with pytest.raises(NextHopQueryGenerationError, match="decision question type"):
        validate(analysis, mismatched_decision, graph, paths)
    with pytest.raises(NextHopQueryGenerationError, match="duplicated"):
        validate(analysis, decision, graph, (paths[0], paths[0]))


def test_generator_branch_variants_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    decision = _insufficient(decision)
    reconstructed = RuleBasedNextHopQueryGeneratorReconstructedV1()
    monkeypatch.setattr(reconstructed, "_bridge_query", lambda *_args: None)
    monkeypatch.setattr(reconstructed, "_comparison_query", lambda *_args: None)
    monkeypatch.setattr(reconstructed, "_seed_query", lambda *_args: None)
    assert reconstructed._generate(analysis, decision, graph, paths).query is None
    assert reconstructed._generate(analysis, decision, graph, ()).query is None

    historical = RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1()
    yes_no = analysis.model_copy(update={"question_type": QuestionType.YES_NO})
    assert historical._generate(yes_no, decision, graph, paths).query is not None
    target = analysis.seed_entities[0]
    comparison = analysis.model_copy(
        update={"question_type": QuestionType.COMPARISON, "comparison_targets": (target, target)}
    )
    assert historical._generate(comparison, decision, graph, paths).query is not None


def test_all_cross_component_validation_guards() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    validate = RuleBasedNextHopQueryGeneratorReconstructedV1._validate_inputs
    bad_graph_metadata = graph.metadata.model_copy(update={"required_relation_hints": ("other",)})
    with pytest.raises(NextHopQueryGenerationError, match="relation hints"):
        validate(
            analysis, decision, graph.model_copy(update={"metadata": bad_graph_metadata}), paths
        )
    with pytest.raises(NextHopQueryGenerationError, match="decision answer type"):
        validate(
            analysis, decision.model_copy(update={"answer_type": AnswerType.DATE}), graph, paths
        )
    bad_decision_metadata = decision.metadata.model_copy(
        update={"source_graph": bad_graph_metadata}
    )
    with pytest.raises(NextHopQueryGenerationError, match="decision graph provenance"):
        validate(
            analysis, decision.model_copy(update={"metadata": bad_decision_metadata}), graph, paths
        )
    with pytest.raises(NextHopQueryGenerationError, match="Component 06"):
        validate(analysis, decision, graph, (object(),))
    wrong_question_path = paths[0].model_copy(update={"question_type": QuestionType.BRIDGE})
    with pytest.raises(NextHopQueryGenerationError, match="candidate path question type"):
        validate(analysis, decision, graph, (wrong_question_path,))
    wrong_path_metadata = paths[0].metadata.model_copy(update={"source_graph": bad_graph_metadata})
    with pytest.raises(NextHopQueryGenerationError, match="candidate path graph provenance"):
        validate(
            analysis,
            decision,
            graph,
            (paths[0].model_copy(update={"metadata": wrong_path_metadata}),),
        )
    wrong_ids = decision.metadata.model_copy(update={"candidate_path_ids": ("different",)})
    with pytest.raises(NextHopQueryGenerationError, match="decision provenance"):
        validate(analysis, decision.model_copy(update={"metadata": wrong_ids}), graph, paths)


def test_exception_wrapping_and_remaining_helper_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    trace = TraceContext.start(run_id="component09-errors", question_id="q")
    for generator in (
        RuleBasedNextHopQueryGeneratorReconstructedV1(
            instrumentation_sink=InMemoryInstrumentationSink()
        ),
        RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1(
            instrumentation_sink=InMemoryInstrumentationSink()
        ),
    ):
        monkeypatch.setattr(
            generator, "_generate", lambda *_args: (_ for _ in ()).throw(ValueError("bad"))
        )
        with pytest.raises(NextHopQueryGenerationError, match="bad"):
            generator.generate(analysis, decision, graph, paths, trace_context=trace)
        monkeypatch.setattr(
            generator, "_generate", lambda *_args: (_ for _ in ()).throw(RuntimeError("bad"))
        )
        with pytest.raises(NextHopQueryGenerationError, match="generation failed"):
            generator.generate(analysis, decision, graph, paths, trace_context=trace)

    reconstructed = RuleBasedNextHopQueryGeneratorReconstructedV1()
    assert reconstructed.config.mode == "research_v1_reconstructed"
    assert reconstructed._comparison_query(analysis, paths) is None
    comparison = analysis.model_copy(update={"question_type": QuestionType.COMPARISON})
    assert reconstructed._comparison_query(comparison, paths) is None
    no_seed = analysis.model_copy(update={"seed_entities": ()})
    assert reconstructed._seed_query(no_seed, paths) is None
    no_relation = analysis.model_copy(update={"required_relation_hints": ()})
    assert reconstructed._seed_query(no_relation, paths) is None
    assert reconstructed_generator._relation_for_path(no_relation, paths[0]) is None
    assert historical_generator._choose_missing_relation("born evidence missing", (), ()) == "born"


def test_historical_answer_factoid_and_target_fallbacks() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    decision = _insufficient(decision).model_copy(update={"best_path": None})
    generator = RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1()
    assert generator.config.mode == "research_v1_historical_adapted"
    answer_only = analysis.model_copy(update={"required_relation_hints": ()})
    answer_query = generator._generate(answer_only, decision, graph, ())
    assert answer_query.query_type is NextHopQueryType.ANSWER_TYPE_COMPLETION
    factoid = answer_only.model_copy(update={"expected_answer_type": AnswerType.UNKNOWN})
    factoid_query = generator._generate(factoid, decision, graph, ())
    assert factoid_query.query_type is NextHopQueryType.FACTOID_COMPLETION
    bridge_path = paths[0].model_copy(
        update={"metadata": paths[0].metadata.model_copy(update={"bridge_entity": "Bridge"})}
    )
    assert (
        historical_generator._choose_target_entity(
            bridge_path, ("Inception",), None, QuestionType.BRIDGE
        )
        == "Bridge"
    )
    empty_path = paths[0].model_copy(update={"entity_chain": ()})
    assert (
        historical_generator._choose_target_entity(
            empty_path, ("Inception",), None, QuestionType.FACTOID
        )
        == "Inception"
    )


def test_next_hop_query_model_rejects_inconsistent_states() -> None:
    analysis, graph, paths, decision = _factoid_inputs()
    query = RuleBasedNextHopQueryGeneratorReconstructedV1().generate(
        analysis, _insufficient(decision), graph, paths
    )
    payload = query.model_dump(mode="json")
    with pytest.raises(ValidationError):
        NextHopQuery.model_validate({**payload, "confidence": 0.1234567})
    with pytest.raises(ValidationError):
        NextHopQuery.model_validate({**payload, "target_entity": None})
    with pytest.raises(ValidationError):
        NextHopQuery.model_validate({**payload, "source": "path_bridge_entity"})
    no_query = query.model_copy(update={"query": None, "query_type": NextHopQueryType.NO_QUERY})
    with pytest.raises(ValidationError):
        NextHopQuery.model_validate(no_query.model_dump(mode="json"))
