"""Focused behavioral tests for Component 07 research-v1."""

from __future__ import annotations

import json

import pytest

from epsa_rag.core.exceptions import (
    FrozenArtifactError,
    QuestionAnalysisError,
    SufficiencyDecisionError,
)
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import QuestionInput
from epsa_rag.epsa.chunk_analysis import RuleBasedCandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.chunk_analysis.models import CanonicalRetrievedChunk
from epsa_rag.epsa.evidence_graph import EvidenceGraphBuilderV1
from epsa_rag.epsa.evidence_path_search import EvidencePathSearcherV1
from epsa_rag.epsa.evidence_scoring import RuleBasedEvidenceScorerV1
from epsa_rag.epsa.evidence_units import RuleBasedEvidenceUnitExtractor
from epsa_rag.epsa.question_analysis import AnswerType, QuestionType, RuleBasedQuestionAnalyzer
from epsa_rag.epsa.sufficiency_decision import (
    DecisionReasonCode,
    GuardCode,
    RuleBasedSufficiencyEngineV1,
    SufficiencyDecision,
    SufficiencyDecisionEngineProtocol,
)
from epsa_rag.epsa.sufficiency_decision import engine as engine_module
from epsa_rag.evaluation.components import sufficiency_decision as evaluator_component
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.evaluation.components.sufficiency_decision import (
    evaluate_sufficiency_decisions,
    write_sufficiency_decision_evaluation,
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
        source_question_id="question-one",
    )
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(first_chunk, analysis)
    first = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, first_chunk, analysis)[0]
    first = first.model_copy(
        update={
            "entities": ("Inception", "Christopher Nolan"),
            "relation_hints": ("directed",),
            "answer_type_candidates": (AnswerType.PERSON,),
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
            "answer_type_candidates": (AnswerType.LOCATION,),
            "question_entity_overlap": (),
            "retrieval_rank": 2,
            "retrieval_score": 0.8,
            "metadata": first.metadata.model_copy(update={"resolution": resolution}),
        }
    )
    scored = RuleBasedEvidenceScorerV1().score_many((first, second), analysis)
    graph = EvidenceGraphBuilderV1().build(analysis, scored)
    return analysis, graph, EvidencePathSearcherV1().search_paths(graph, analysis)


def _single_hop_inputs(question: str, answer_type: AnswerType):
    analysis = RuleBasedQuestionAnalyzer().analyze(question)
    text = "Inception was directed by Christopher Nolan."
    chunk = CanonicalRetrievedChunk(
        chunk_id="inception-direct",
        doc_title="Inception",
        paragraph_index=0,
        paragraph_text=text,
        chunk_text=text,
        sentences=(Sentence(index=0, text=text),),
        retrieval_rank=1,
        retrieval_score=0.9,
        source_question_id="single-hop",
    )
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    unit = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, chunk, analysis)[0]
    unit = unit.model_copy(
        update={
            "entities": ("Inception", "Christopher Nolan"),
            "relation_hints": ("directed",),
            "answer_type_candidates": (answer_type,),
            "question_entity_overlap": ("Inception",),
        }
    )
    graph = EvidenceGraphBuilderV1().build(
        analysis, RuleBasedEvidenceScorerV1().score_many((unit,), analysis)
    )
    return analysis, graph, EvidencePathSearcherV1().search_paths(graph, analysis)


def test_complete_bridge_is_sufficient_and_preserves_component_06_provenance() -> None:
    analysis, graph, paths = _bridge_inputs()
    sink = InMemoryInstrumentationSink()
    engine = RuleBasedSufficiencyEngineV1(instrumentation_sink=sink)

    decision = engine.decide(
        analysis,
        graph,
        paths,
        trace_context=TraceContext.start(run_id="sufficiency-test", question_id="question-one"),
    )

    assert isinstance(engine, SufficiencyDecisionEngineProtocol)
    assert decision.sufficient is True
    assert decision.decision_reason is DecisionReasonCode.SUFFICIENT_BRIDGE
    assert decision.answer_candidate == "London"
    assert decision.selected_evidence_unit_ids == ("inception::s0", "christopher-nolan::s0")
    assert decision.best_path is not None
    assert decision.best_path.metadata.source_graph == graph.metadata
    assert decision.metadata.candidate_path_ids == tuple(path.path_id for path in paths)
    assert decision.metadata.confidence_kind == "uncalibrated_heuristic"
    assert [event.event_type for event in sink.events] == ["epsa.sufficiency_decision.completed"]


def test_no_paths_returns_insufficient_without_inventing_an_answer() -> None:
    analysis, graph, _ = _bridge_inputs()

    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, ())

    assert decision.sufficient is False
    assert decision.confidence == 0.0
    assert decision.decision_reason is DecisionReasonCode.NO_CANDIDATE_PATHS
    assert decision.answer_candidate is None
    assert decision.selected_evidence_unit_ids == ()


def test_comparison_research_v1_remains_insufficient() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze("Which is longer, River A or River B?")
    analysis = analysis.model_copy(
        update={
            "question_type": QuestionType.COMPARISON,
            "comparison_targets": analysis.seed_entities,
        }
    )
    text = "River A has a length of 100 km."
    chunk = CanonicalRetrievedChunk(
        chunk_id="river-a",
        doc_title="River A",
        paragraph_index=0,
        paragraph_text=text,
        chunk_text=text,
        sentences=(Sentence(index=0, text=text),),
        retrieval_rank=1,
        retrieval_score=0.9,
        source_question_id="comparison-one",
    )
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    unit = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, chunk, analysis)[0]
    unit = unit.model_copy(
        update={
            "entities": ("River A", "100 km"),
            "relation_hints": ("length",),
            "answer_type_candidates": (AnswerType.NUMBER,),
            "question_entity_overlap": ("River A",),
        }
    )
    graph = EvidenceGraphBuilderV1().build(
        analysis, RuleBasedEvidenceScorerV1().score_many((unit,), analysis)
    )
    paths = EvidencePathSearcherV1().search_paths(graph, analysis)

    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, paths)

    assert decision.sufficient is False
    assert decision.decision_reason is DecisionReasonCode.COMPARISON_UNSUPPORTED_RESEARCH_V1


def test_yes_no_evidence_is_sufficient_without_selecting_polarity() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze("Was Inception directed by Christopher Nolan?")
    text = "Inception was directed by Christopher Nolan."
    chunk = CanonicalRetrievedChunk(
        chunk_id="inception-yes-no",
        doc_title="Inception",
        paragraph_index=0,
        paragraph_text=text,
        chunk_text=text,
        sentences=(Sentence(index=0, text=text),),
        retrieval_rank=1,
        retrieval_score=0.9,
        source_question_id="yes-no",
    )
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    unit = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, chunk, analysis)[0]
    unit = unit.model_copy(
        update={
            "entities": ("Inception", "Christopher Nolan"),
            "relation_hints": ("directed",),
            "answer_type_candidates": (AnswerType.BOOLEAN,),
            "question_entity_overlap": ("Inception", "Christopher Nolan"),
        }
    )
    graph = EvidenceGraphBuilderV1().build(
        analysis, RuleBasedEvidenceScorerV1().score_many((unit,), analysis)
    )
    paths = EvidencePathSearcherV1().search_paths(graph, analysis)

    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, paths)

    assert decision.sufficient is True
    assert decision.decision_reason is DecisionReasonCode.SUFFICIENT_YES_NO_EVIDENCE
    assert decision.answer_candidate is None
    assert decision.metadata.does_not_generate_yes_no_polarity is True


def test_inference_evaluator_runs_components_01_through_07_without_gold_fields() -> None:
    text = "Inception was directed by Christopher Nolan."
    inputs = (
        InferenceRetrieval(
            question=QuestionInput(question_id="q-eval", text="Who directed Inception?"),
            chunks=(
                RankedParagraphChunk(
                    chunk=ParagraphChunk(
                        chunk_id="inception-eval",
                        title="Inception",
                        paragraph_text=text,
                        sentences=(Sentence(index=0, text=text),),
                    ),
                    rank=1,
                    score=0.9,
                ),
            ),
            retriever_version="retriever-v1",
        ),
    )

    summary, traces, events = evaluate_sufficiency_decisions(inputs, run_id="sufficiency-eval")

    assert summary.completed_questions == 1
    assert summary.diagnostics.decisions == 1
    assert traces[0].decision is not None
    assert traces[0].decision.sufficient is True
    assert "supporting" not in traces[0].model_dump_json()
    assert [event.event_type for event in events][-1] == "epsa.sufficiency_decision.completed"


def test_bridge_and_factoid_failure_reasons_are_deterministic() -> None:
    analysis, graph, paths = _bridge_inputs()
    bridge = next(path for path in paths if path.answer_candidate == "London")
    incomplete = bridge.model_copy(
        update={
            "evidence_unit_ids": bridge.evidence_unit_ids[:1],
            "scored_evidence_units": bridge.scored_evidence_units[:1],
        }
    )
    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, (incomplete,))

    assert decision.sufficient is False
    assert decision.decision_reason is DecisionReasonCode.BRIDGE_RULES_UNSATISFIED
    assert decision.rule_trace[-1].rule_code.value == "evidence_units"

    factoid_analysis = RuleBasedQuestionAnalyzer().analyze("Who directed Inception?")
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
        source_question_id="factoid-one",
    )
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, factoid_analysis)
    unit = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, chunk, factoid_analysis)[
        0
    ]
    graph = EvidenceGraphBuilderV1().build(
        factoid_analysis, RuleBasedEvidenceScorerV1().score_many((unit,), factoid_analysis)
    )
    path = EvidencePathSearcherV1().search_paths(graph, factoid_analysis)[0]
    generic = path.model_copy(update={"answer_candidate": "person"})
    decision = RuleBasedSufficiencyEngineV1().decide(factoid_analysis, graph, (generic,))

    assert decision.sufficient is False
    assert decision.decision_reason is DecisionReasonCode.FACTOID_RULES_UNSATISFIED
    assert decision.rule_trace[-1].rule_code.value == "answer_candidate"


def test_guard_utilities_cover_surface_anchor_and_independent_evidence_cases() -> None:
    assert engine_module._specific_bridge("Christopher Nolan") is True
    assert engine_module._specific_bridge("British") is False
    assert engine_module._specific_answer("London") is True
    assert engine_module._specific_answer("United") is False
    assert engine_module._answer_surface_matches("200 km", AnswerType.NUMBER) is True
    assert engine_module._answer_surface_matches("London", AnswerType.NUMBER) is False
    assert engine_module._answer_surface_matches("January 1990", AnswerType.DATE) is True
    assert engine_module._answer_surface_matches("Christopher Nolan", AnswerType.PERSON) is True

    analysis, graph, paths = _bridge_inputs()
    path = next(path for path in paths if path.answer_candidate == "London")
    assert engine_module._coverage(path, graph)[0] is True
    assert engine_module._strong_anchor_coverage(path, analysis)[0] is True
    quoted = analysis.model_copy(update={"raw_question": 'Where was "Missing Title" born?'})
    assert engine_module._strong_anchor_coverage(path, quoted)[0] is False


def test_evaluator_export_is_immutable_and_contains_only_inference_fields(tmp_path) -> None:
    text = "Inception was directed by Christopher Nolan."
    inputs = (
        InferenceRetrieval(
            question=QuestionInput(question_id="export-q", text="Who directed Inception?"),
            chunks=(
                RankedParagraphChunk(
                    chunk=ParagraphChunk(
                        chunk_id="export-inception",
                        title="Inception",
                        paragraph_text=text,
                        sentences=(Sentence(index=0, text=text),),
                    ),
                    rank=1,
                    score=0.9,
                ),
            ),
            retriever_version="retriever-v1",
        ),
    )
    summary, traces, events = evaluate_sufficiency_decisions(inputs, run_id="export-run")
    manifest = write_sufficiency_decision_evaluation(tmp_path, summary, traces, events)

    assert {file.relative_path for file in manifest.files} == {
        "run.json",
        "traces.jsonl",
        "events.jsonl",
    }
    payload = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))
    assert "supporting" not in json.dumps(payload)
    with pytest.raises(FrozenArtifactError):
        write_sufficiency_decision_evaluation(tmp_path, summary, traces, events)


def test_engine_rejects_invalid_component_contracts_and_emits_failure_event() -> None:
    analysis, graph, paths = _bridge_inputs()
    sink = InMemoryInstrumentationSink()
    engine = RuleBasedSufficiencyEngineV1(instrumentation_sink=sink)
    context = TraceContext.start(run_id="invalid-contract", question_id="question-one")

    assert engine.config.schema_version == "sufficiency-decision-v1"
    with pytest.raises(SufficiencyDecisionError, match="question analysis"):
        engine.decide("not-analysis", graph, paths, trace_context=context)  # type: ignore[arg-type]
    with pytest.raises(SufficiencyDecisionError, match="evidence graph"):
        engine.decide(analysis, "not-graph", paths, trace_context=context)  # type: ignore[arg-type]
    with pytest.raises(SufficiencyDecisionError, match="sequence"):
        engine.decide(analysis, graph, "not-paths", trace_context=context)  # type: ignore[arg-type]

    assert [event.event_type for event in sink.events] == [
        "epsa.sufficiency_decision.failed",
        "epsa.sufficiency_decision.failed",
        "epsa.sufficiency_decision.failed",
    ]


@pytest.mark.parametrize(
    ("name", "replacement", "expected_guard"),
    [
        ("_bridge_entity", lambda *_args: None, GuardCode.BRIDGE_ENTITY),
        ("_specific_bridge", lambda *_args: False, GuardCode.BRIDGE_SPECIFICITY),
        (
            "_bridge_grounded_answer_side",
            lambda *_args: False,
            GuardCode.BRIDGE_ANSWER_SIDE_GROUNDING,
        ),
        (
            "_answer_failure",
            lambda *_args: engine_module._fail(GuardCode.ANSWER_CANDIDATE, "forced"),
            GuardCode.ANSWER_CANDIDATE,
        ),
        (
            "_relation_coverage",
            lambda *_args: (0, ("directed",)),
            GuardCode.RELATION_COVERAGE,
        ),
        (
            "_strong_anchor_coverage",
            lambda *_args: (False, {}),
            GuardCode.QUESTION_ANCHOR_COVERAGE,
        ),
        ("_role_coverage", lambda *_args: (False, {}), GuardCode.ROLE_COVERAGE),
        (
            "_coverage",
            lambda *_args: (False, {"covered": 1}),
            GuardCode.INDEPENDENT_EVIDENCE_COVERAGE,
        ),
    ],
)
def test_bridge_policy_reports_each_conservative_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: object,
    expected_guard: GuardCode,
) -> None:
    analysis, graph, paths = _bridge_inputs()
    monkeypatch.setattr(engine_module, name, replacement)

    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, paths)

    assert decision.sufficient is False
    assert decision.decision_reason is DecisionReasonCode.BRIDGE_RULES_UNSATISFIED
    assert decision.rule_trace[-1].rule_code is expected_guard


@pytest.mark.parametrize(
    ("name", "replacement", "expected_guard"),
    [
        ("_seed_connected", lambda *_args: False, GuardCode.SEED_CONNECTION),
        (
            "_answer_failure",
            lambda *_args: engine_module._fail(GuardCode.ANSWER_CANDIDATE, "forced"),
            GuardCode.ANSWER_CANDIDATE,
        ),
        ("_role_coverage", lambda *_args: (False, {}), GuardCode.ROLE_COVERAGE),
        (
            "_relation_coverage",
            lambda *_args: (0, ("directed",)),
            GuardCode.RELATION_COVERAGE,
        ),
        (
            "_strong_anchor_coverage",
            lambda *_args: (False, {}),
            GuardCode.QUESTION_ANCHOR_COVERAGE,
        ),
    ],
)
def test_factoid_policy_reports_each_conservative_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: object,
    expected_guard: GuardCode,
) -> None:
    analysis, graph, paths = _single_hop_inputs("Who directed Inception?", AnswerType.PERSON)
    monkeypatch.setattr(engine_module, name, replacement)

    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, paths)

    assert decision.sufficient is False
    assert decision.decision_reason is DecisionReasonCode.FACTOID_RULES_UNSATISFIED
    assert decision.rule_trace[-1].rule_code is expected_guard


def test_factoid_and_yes_no_reject_missing_or_incomplete_evidence() -> None:
    factoid_analysis, factoid_graph, factoid_paths = _single_hop_inputs(
        "Who directed Inception?", AnswerType.PERSON
    )
    empty_factoid = factoid_paths[0].model_copy(
        update={"evidence_unit_ids": (), "scored_evidence_units": ()}
    )
    with pytest.raises(SufficiencyDecisionError, match="candidate paths require"):
        RuleBasedSufficiencyEngineV1().decide(factoid_analysis, factoid_graph, (empty_factoid,))

    yes_no_analysis, yes_no_graph, yes_no_paths = _single_hop_inputs(
        "Was Inception directed by Christopher Nolan?", AnswerType.BOOLEAN
    )
    empty_yes_no = yes_no_paths[0].model_copy(
        update={"evidence_unit_ids": (), "scored_evidence_units": ()}
    )
    with pytest.raises(SufficiencyDecisionError, match="candidate paths require"):
        RuleBasedSufficiencyEngineV1().decide(yes_no_analysis, yes_no_graph, (empty_yes_no,))


@pytest.mark.parametrize(
    ("candidate", "answer_type", "expected"),
    [
        ("anything", AnswerType.UNKNOWN, True),
        ("42", AnswerType.NUMBER, True),
        ("London", AnswerType.NUMBER, False),
        ("January 1990", AnswerType.DATE, True),
        ("London", AnswerType.LOCATION, True),
        ("lowercase", AnswerType.LOCATION, False),
        ("Christopher Nolan", AnswerType.PERSON, True),
        ("Nolan", AnswerType.PERSON, False),
        ("Example University", AnswerType.ORGANIZATION, True),
        ("x", AnswerType.TITLE_OR_WORK, False),
    ],
)
def test_answer_surface_rules_cover_supported_answer_types(
    candidate: str, answer_type: AnswerType, expected: bool
) -> None:
    assert engine_module._answer_surface_matches(candidate, answer_type) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("person", False),
        ("January", False),
        ("United", False),
        ("Director of", False),
        ("Christopher Nolan", True),
    ],
)
def test_specific_answer_and_miscellaneous_guard_helpers(value: str | None, expected: bool) -> None:
    assert engine_module._specific_answer(value) is expected

    analysis, graph, paths = _bridge_inputs()
    path = next(path for path in paths if path.answer_candidate == "London")
    assert engine_module._seed_connected(path, graph, analysis) is True
    no_seed_analysis = analysis.model_copy(update={"seed_entities": ()})
    assert engine_module._seed_connected(path, graph, no_seed_analysis)
    assert engine_module._bridge_entity(path, analysis) == "Christopher Nolan"
    assert engine_module._bridge_grounded_answer_side(path, graph, "Christopher Nolan") is True
    assert engine_module._bridge_grounded_answer_side(
        path.model_copy(update={"scored_evidence_units": ()}), graph, "Christopher Nolan"
    ) is False
    assert engine_module._relation_coverage(path, analysis)[0] == 2
    assert engine_module._role_coverage(path, graph, analysis, "Christopher Nolan")[0] is True
    assert engine_module._coverage(path, graph)[0] is True
    missing_evidence_path = path.model_copy(update={"evidence_unit_ids": ("missing",)})
    assert engine_module._coverage(missing_evidence_path, graph)[0] is False
    assert engine_module._multi_fact_question(
        RuleBasedQuestionAnalyzer().analyze("Who was the director of Inception?")
    ) is True
    assert engine_module._multi_fact_question(
        RuleBasedQuestionAnalyzer().analyze("Who directed Inception?")
    ) is False
    assert engine_module._selected_chunk_ids(graph, path.evidence_unit_ids) == (
        "inception",
        "christopher-nolan",
    )


def test_engine_rejects_graph_and_path_provenance_mismatches() -> None:
    analysis, graph, paths = _bridge_inputs()
    engine = RuleBasedSufficiencyEngineV1()

    mismatched_type = graph.model_copy(update={"question_type": QuestionType.FACTOID})
    with pytest.raises(SufficiencyDecisionError, match="question type"):
        engine.decide(analysis, mismatched_type, paths)

    mismatched_answer = graph.model_copy(
        update={
            "metadata": graph.metadata.model_copy(
                update={"expected_answer_type": AnswerType.PERSON}
            )
        }
    )
    with pytest.raises(SufficiencyDecisionError, match="answer type"):
        engine.decide(analysis, mismatched_answer, paths)

    mismatched_relations = graph.model_copy(
        update={"metadata": graph.metadata.model_copy(update={"required_relation_hints": ()})}
    )
    with pytest.raises(SufficiencyDecisionError, match="relation hints"):
        engine.decide(analysis, mismatched_relations, paths)

    with pytest.raises(SufficiencyDecisionError, match="Component 06 contract"):
        engine._validate_inputs(analysis, graph, (object(),))

    mismatched_path_type = paths[0].model_copy(update={"question_type": QuestionType.FACTOID})
    with pytest.raises(SufficiencyDecisionError, match="candidate path question type"):
        engine.decide(analysis, graph, (mismatched_path_type,))

    mismatched_source = paths[0].model_copy(
        update={
            "metadata": paths[0].metadata.model_copy(
                update={"source_graph": graph.metadata.model_copy(update={"version": "other"})}
            )
        }
    )
    with pytest.raises(SufficiencyDecisionError, match="graph provenance"):
        engine.decide(analysis, graph, (mismatched_source,))


def test_factoid_generic_and_multi_fact_guards_are_conservative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis, graph, paths = _single_hop_inputs("Who directed Inception?", AnswerType.PERSON)
    engine = RuleBasedSufficiencyEngineV1()
    generic = analysis.model_copy(
        update={"expected_answer_type": AnswerType.ENTITY, "required_relation_hints": ()}
    )
    generic_decision = engine._factoid(generic, graph, paths)
    assert generic_decision.rule_trace[-1].rule_code is GuardCode.GENERIC_FACTOID_COVERAGE

    multi = RuleBasedQuestionAnalyzer().analyze("Who was the director of Inception?")
    multi_decision = engine._factoid(multi, graph, paths)
    assert multi_decision.rule_trace[-1].rule_code is GuardCode.MULTI_FACT_COVERAGE

    monkeypatch.setattr(engine_module, "_coverage", lambda *_args: (True, {"covered": 2}))
    sufficient = engine._factoid(multi, graph, paths)
    assert sufficient.sufficient is True
    assert sufficient.rule_trace[-1].rule_code is GuardCode.MULTI_FACT_COVERAGE


@pytest.mark.parametrize(
    ("name", "replacement", "expected_guard"),
    [
        ("_seed_connected", lambda *_args: False, GuardCode.SEED_CONNECTION),
        (
            "_relation_coverage",
            lambda *_args: (0, ("directed",)),
            GuardCode.RELATION_COVERAGE,
        ),
    ],
)
def test_yes_no_policy_reports_connection_and_relation_failures(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: object,
    expected_guard: GuardCode,
) -> None:
    analysis, graph, paths = _single_hop_inputs(
        "Was Inception directed by Christopher Nolan?", AnswerType.BOOLEAN
    )
    monkeypatch.setattr(engine_module, name, replacement)

    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, paths)

    assert decision.sufficient is False
    assert decision.decision_reason is DecisionReasonCode.YES_NO_RULES_UNSATISFIED
    assert decision.rule_trace[-1].rule_code is expected_guard


def test_remaining_helper_and_model_validation_branches() -> None:
    analysis, graph, paths = _bridge_inputs()
    path = next(path for path in paths if path.answer_candidate == "London")
    chain_only = path.model_copy(update={"node_ids": (), "entity_chain": ("Inception",)})
    assert engine_module._seed_connected(chain_only, graph, analysis) is True
    fallback_bridge = path.model_copy(
        update={"metadata": path.metadata.model_copy(update={"bridge_entity": None})}
    )
    assert engine_module._bridge_entity(fallback_bridge, analysis) == "Christopher Nolan"
    assert engine_module._answer_failure(path, graph, AnswerType.PERSON) is not None
    wrong_type = path.model_copy(update={"answer_type": AnswerType.PERSON})
    assert engine_module._answer_failure(wrong_type, graph, AnswerType.LOCATION) is not None
    assert engine_module._graph_path_answer_type_matches(path, graph, AnswerType.ENTITY) is True
    assert (
        engine_module._graph_path_answer_type_matches(wrong_type, graph, AnswerType.LOCATION)
        is False
    )
    digit_anchor = analysis.model_copy(update={"raw_question": "Was 2 Fast directed by Nolan?"})
    assert engine_module._strong_anchor_coverage(path, digit_anchor)[0] is False

    decision = RuleBasedSufficiencyEngineV1().decide(analysis, graph, paths)
    payload = decision.model_dump()
    with pytest.raises(ValueError, match="rounded"):
        SufficiencyDecision.model_validate({**payload, "confidence": 0.1234567})
    with pytest.raises(ValueError, match="candidate path IDs"):
        SufficiencyDecision.model_validate(
            {**payload, "metadata": {**payload["metadata"], "candidate_path_ids": ()}}
        )
    with pytest.raises(ValueError, match="best path"):
        SufficiencyDecision.model_validate(
            {
                **payload,
                "best_path": None,
                "candidate_paths_considered": (),
                "metadata": {**payload["metadata"], "candidate_path_ids": ()},
            }
        )
    yes_no = payload | {"question_type": QuestionType.YES_NO, "answer_candidate": "yes"}
    with pytest.raises(ValueError, match="yes/no polarity"):
        SufficiencyDecision.model_validate(yes_no)


def test_unexpected_engine_and_evaluator_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis, graph, paths = _bridge_inputs()
    sink = InMemoryInstrumentationSink()
    engine = RuleBasedSufficiencyEngineV1(instrumentation_sink=sink)
    def raise_unexpected(*_args: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "_bridge", raise_unexpected)
    with pytest.raises(SufficiencyDecisionError, match="sufficiency decision failed"):
        engine.decide(
            analysis,
            graph,
            paths,
            trace_context=TraceContext.start(run_id="unexpected", question_id="question-one"),
        )
    assert sink.events[-1].event_type == "epsa.sufficiency_decision.failed"

    with pytest.raises(ValueError, match="requires inference inputs"):
        evaluator_component.evaluate_sufficiency_decisions((), run_id="empty")

    def fail_analysis(*_args: object, **_kwargs: object) -> object:
        raise QuestionAnalysisError("forced")

    monkeypatch.setattr(evaluator_component.RuleBasedQuestionAnalyzer, "analyze", fail_analysis)
    summary, traces, events = evaluator_component.evaluate_sufficiency_decisions(
        (
            InferenceRetrieval(
                question=QuestionInput(question_id="failed-eval", text="Who directed Inception?"),
                chunks=(),
                retriever_version="retriever-v1",
            ),
        ),
        run_id="failed-eval",
        config=evaluator_component.SufficiencyDecisionEvaluationConfig(
            retain_instrumentation_events=False
        ),
    )
    assert summary.failed_questions == 1
    assert traces[0].error_type == "QuestionAnalysisError"
    assert events == ()
