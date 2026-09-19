"""Focused behavioral tests for EPSA Component 01."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import epsa_rag.epsa.question_analysis.analyzer as analyzer_module
from epsa_rag.core.exceptions import QuestionAnalysisError
from epsa_rag.epsa.question_analysis import (
    AnalysisMetadata,
    AnswerType,
    AnswerTypeCandidate,
    EntityMention,
    QuestionAnalysis,
    QuestionAnalyzerConfig,
    QuestionType,
    RelationHint,
    RuleBasedQuestionAnalyzer,
)
from epsa_rag.epsa.question_analysis.protocols import QuestionAnalyzerProtocol
from epsa_rag.instrumentation import InMemoryInstrumentationSink, TraceContext


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Is Paris in France?", AnswerType.BOOLEAN),
        ("How many children did Albert Einstein have?", AnswerType.NUMBER),
        ("When was Inception released?", AnswerType.DATE),
        ("What film did Christopher Nolan direct?", AnswerType.TITLE_OR_WORK),
        ("Which university did Ada Lovelace attend?", AnswerType.ORGANIZATION),
        ("Where was Christopher Nolan born?", AnswerType.LOCATION),
        ("Who wrote Inception?", AnswerType.PERSON),
        ("What is the capital of France?", AnswerType.ENTITY),
        ("Tell me about Inception.", AnswerType.UNKNOWN),
    ],
)
def test_answer_type_rules_use_documented_precedence(question: str, expected: AnswerType) -> None:
    assert RuleBasedQuestionAnalyzer().analyze(question).expected_answer_type is expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which of A and B was born earlier?", QuestionType.COMPARISON),
        ("Was Christopher Nolan born in London?", QuestionType.YES_NO),
        ("Where was the director of Inception born?", QuestionType.BRIDGE),
        ("Where is Paris located?", QuestionType.FACTOID),
    ],
)
def test_question_type_rules_use_documented_order(question: str, expected: QuestionType) -> None:
    assert RuleBasedQuestionAnalyzer().analyze(question).question_type is expected


def test_normalization_entities_relations_and_spans_are_inspectable() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        "  Where   was the Director of \u201cInception\u201d born?  "
    )

    assert analysis.raw_question == "  Where   was the Director of \u201cInception\u201d born?  "
    assert analysis.normalized_question == 'where was the director of "inception" born?'
    entity_details = [
        (entity.text, entity.source, entity.confidence) for entity in analysis.seed_entities
    ]
    assert entity_details == [
        ("Inception", "quoted_string", 0.98),
        ("Director", "title_like_token", 0.7),
    ]
    assert [(hint.relation, hint.matched_text) for hint in analysis.required_relation_hints] == [
        ("directed", "director"),
        ("born", "born"),
    ]
    for entity in analysis.seed_entities:
        assert analysis.normalized_question[entity.start : entity.end] == entity.text.lower()
    for hint in analysis.required_relation_hints:
        assert analysis.normalized_question[hint.start : hint.end] == hint.matched_text
    assert analysis.metadata.version == "rule_based_v1"
    assert len(analysis.metadata.configuration_fingerprint) == 64


def test_question_leads_are_not_seed_entities() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze("Where was Inception released?")

    assert {entity.normalized for entity in analysis.seed_entities} == {"inception"}


def test_multi_token_and_single_title_entities_are_extracted_deterministically() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        "Where did Christopher Nolan attend University of Oxford?"
    )

    assert "christopher nolan" in {entity.normalized for entity in analysis.seed_entities}
    assert "university of oxford" in {entity.normalized for entity in analysis.seed_entities}
    assert analysis == RuleBasedQuestionAnalyzer().analyze(
        "Where did Christopher Nolan attend University of Oxford?"
    )


def test_exact_normalized_entity_deduplication_keeps_highest_priority_pass() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze('Who directed "Inception"?')

    inception = [entity for entity in analysis.seed_entities if entity.normalized == "inception"]
    assert len(inception) == 1
    assert inception[0].source == "quoted_string"
    assert inception[0].confidence == 0.98


def test_bare_in_at_no_longer_create_false_location_hints() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        "Which actor appeared in Inception at the premiere?"
    )

    assert analysis.required_relation_hints == ()
    assert analysis.question_type is QuestionType.FACTOID


def test_comparison_target_extraction_prefers_explicit_which_of_targets() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        "Which of Christopher Nolan and Steven Spielberg was born earlier?"
    )

    assert analysis.question_type is QuestionType.COMPARISON
    assert [(target.text, target.confidence) for target in analysis.comparison_targets] == [
        ("Christopher Nolan", 0.8),
        ("Steven Spielberg", 0.8),
    ]


def test_between_comparison_targets_and_seed_fallback_are_conservative() -> None:
    between = RuleBasedQuestionAnalyzer().analyze(
        "Who was born earlier between Ada Lovelace and Grace Hopper?"
    )
    fallback = RuleBasedQuestionAnalyzer().analyze("Which of the two, Ada or Grace, is older?")

    assert [target.text for target in between.comparison_targets] == [
        "Ada Lovelace",
        "Grace Hopper",
    ]
    assert [target.text for target in fallback.comparison_targets] == ["Ada", "Grace"]


def test_json_serialization_immutability_and_config_fingerprint_are_stable() -> None:
    config = QuestionAnalyzerConfig()
    analysis = RuleBasedQuestionAnalyzer(config=config).analyze("Where is Paris located?")

    restored = type(analysis).model_validate_json(analysis.model_dump_json())
    assert restored == analysis
    assert json.loads(analysis.model_dump_json())["question_type"] == "factoid"
    assert json.loads(analysis.model_dump_json())["expected_answer_type"] == "LOCATION"
    assert config.fingerprint() == analysis.metadata.configuration_fingerprint
    with pytest.raises(ValidationError):
        analysis.question_type = QuestionType.BRIDGE
    with pytest.raises(ValidationError):
        config.version = "other"  # type: ignore[assignment]


def test_contract_validation_rejects_invalid_spans_scores_and_inconsistent_analysis() -> None:
    metadata = AnalysisMetadata(configuration_fingerprint="a" * 64)
    candidate = AnswerTypeCandidate(
        answer_type=AnswerType.LOCATION, text="LOCATION", confidence=0.75
    )
    with pytest.raises(ValidationError, match="whitespace-collapsed"):
        EntityMention(
            text="Paris",
            normalized="PARIS",
            source="title_like_token",
            start=0,
            end=5,
            confidence=0.7,
        )
    with pytest.raises(ValidationError, match="end must be greater"):
        RelationHint(
            relation="located",
            matched_text="in",
            start=2,
            end=2,
            confidence=0.8,
        )
    with pytest.raises(ValidationError, match="exactly one"):
        QuestionAnalysis(
            raw_question="Where is Paris?",
            normalized_question="where is paris?",
            question_type=QuestionType.FACTOID,
            expected_answer_type=AnswerType.LOCATION,
            seed_entities=(),
            required_relation_hints=(),
            comparison_targets=(),
            answer_type_candidates=(),
            metadata=metadata,
        )
    with pytest.raises(ValidationError, match="must match"):
        QuestionAnalysis(
            raw_question="Where is Paris?",
            normalized_question="where is paris?",
            question_type=QuestionType.FACTOID,
            expected_answer_type=AnswerType.LOCATION,
            seed_entities=(),
            required_relation_hints=(),
            comparison_targets=(),
            answer_type_candidates=(
                candidate.model_copy(update={"answer_type": AnswerType.PERSON}),
            ),
            metadata=metadata,
        )


def test_protocol_and_broader_bridge_rule_are_available_to_downstream_components() -> None:
    analyzer = RuleBasedQuestionAnalyzer()

    assert isinstance(analyzer, QuestionAnalyzerProtocol)
    assert analyzer.config.version == "rule_based_v1"
    assert (
        analyzer.analyze("Where did the actor from New York work?").question_type
        is QuestionType.BRIDGE
    )


def test_internal_parsing_failure_is_wrapped_and_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = InMemoryInstrumentationSink()
    analyzer = RuleBasedQuestionAnalyzer(instrumentation_sink=sink)
    context = TraceContext.start(run_id="question-analyzer-test", question_id="q-2")

    def fail(_: str) -> AnswerType:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(analyzer_module, "infer_answer_type", fail)
    with pytest.raises(QuestionAnalysisError, match="rule-based question analysis failed"):
        analyzer.analyze("Where is Paris?", trace_context=context)

    assert sink.events[0].event_type == "epsa.question_analysis.failed"


@pytest.mark.parametrize("question", ["", "   ", None, 1])
def test_invalid_input_fails_explicitly(question: object) -> None:
    with pytest.raises(QuestionAnalysisError, match="question must be"):
        RuleBasedQuestionAnalyzer().analyze(question)  # type: ignore[arg-type]


def test_structured_success_and_failure_events_cross_the_existing_sink_boundary() -> None:
    sink = InMemoryInstrumentationSink()
    analyzer = RuleBasedQuestionAnalyzer(instrumentation_sink=sink)
    context = TraceContext.start(run_id="question-analyzer-test", question_id="q-1")

    analysis = analyzer.analyze("Where is Paris located?", trace_context=context)
    with pytest.raises(QuestionAnalysisError):
        analyzer.analyze("", trace_context=context)

    assert [event.event_type for event in sink.events] == [
        "epsa.question_analysis.completed",
        "epsa.question_analysis.failed",
    ]
    assert sink.events[0].payload["analysis"] == analysis.model_dump(mode="json")
    assert sink.events[1].payload["error_type"] == "QuestionAnalysisError"
