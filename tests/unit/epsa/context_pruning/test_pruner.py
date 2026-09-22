"""Focused research-v1 behavioral tests for EPSA Component 08."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from epsa_rag.core.exceptions import ContextPruningError
from epsa_rag.core.models import Sentence
from epsa_rag.epsa.chunk_analysis import RuleBasedCandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.chunk_analysis.models import CanonicalRetrievedChunk
from epsa_rag.epsa.context_pruning import (
    ContextPrunerProtocol,
    PrunedContext,
    PruningStrategy,
    ResearchContextPrunerV1,
    ResearchContextPrunerV1Config,
)
from epsa_rag.epsa.evidence_graph import EvidenceGraphBuilderV1
from epsa_rag.epsa.evidence_scoring import RuleBasedEvidenceScorerV1, ScoredEvidenceUnit
from epsa_rag.epsa.evidence_units import RuleBasedEvidenceUnitExtractor
from epsa_rag.epsa.question_analysis import QuestionType, RuleBasedQuestionAnalyzer
from epsa_rag.epsa.sufficiency_decision import RuleBasedSufficiencyEngineV1, SufficiencyDecision
from epsa_rag.instrumentation import InMemoryInstrumentationSink, TraceContext


def _scored_units() -> tuple[ScoredEvidenceUnit, ...]:
    analysis = RuleBasedQuestionAnalyzer().analyze("Who wrote Atlas?")
    sentences = tuple(Sentence(index=index, text=f"Sentence {index}.") for index in range(7))
    chunk = CanonicalRetrievedChunk(
        chunk_id="atlas",
        doc_title="Atlas",
        paragraph_index=0,
        paragraph_text="".join(sentence.text for sentence in sentences),
        chunk_text="".join(sentence.text for sentence in sentences),
        sentences=sentences,
        retrieval_rank=2,
        retrieval_score=0.9,
        source_question_id="context-pruner-question",
    )
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    units = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, chunk, analysis)
    return RuleBasedEvidenceScorerV1().score_many(units, analysis)


def _decision(
    evidence: tuple[ScoredEvidenceUnit, ...],
    *,
    sufficient: bool,
    question_type: QuestionType,
    requested_ids: tuple[str, ...],
) -> SufficiencyDecision:
    analysis = RuleBasedQuestionAnalyzer().analyze("Who wrote Atlas?")
    graph = EvidenceGraphBuilderV1().build(analysis, evidence)
    base = RuleBasedSufficiencyEngineV1().decide(analysis, graph, ())
    return base.model_copy(
        update={
            "sufficient": sufficient,
            "question_type": question_type,
            "selected_evidence_unit_ids": requested_ids,
        }
    )


def test_deduplicates_requested_ids_and_reports_missing_ids() -> None:
    evidence = _scored_units()
    selected_id = evidence[2].evidence_unit.evidence_unit_id
    decision = _decision(
        evidence,
        sufficient=True,
        question_type=QuestionType.FACTOID,
        requested_ids=(selected_id, selected_id, "missing::s0"),
    )

    context = ResearchContextPrunerV1().prune(decision, evidence)

    assert context.selected_evidence_unit_ids == (selected_id,)
    assert context.diagnostics.requested_evidence_unit_ids == (selected_id, "missing::s0")
    assert context.diagnostics.missing_requested_evidence_unit_ids == ("missing::s0",)
    assert context.removed_evidence_unit_ids == tuple(
        unit.evidence_unit.evidence_unit_id
        for unit in evidence
        if unit.evidence_unit != evidence[2].evidence_unit
    )


def test_insufficient_and_non_bridge_decisions_do_not_expand() -> None:
    evidence = _scored_units()
    selected_id = evidence[3].evidence_unit.evidence_unit_id
    pruner = ResearchContextPrunerV1()

    insufficient = pruner.prune(
        _decision(
            evidence,
            sufficient=False,
            question_type=QuestionType.BRIDGE,
            requested_ids=(selected_id,),
        ),
        evidence,
    )
    factoid = pruner.prune(
        _decision(
            evidence,
            sufficient=True,
            question_type=QuestionType.FACTOID,
            requested_ids=(selected_id,),
        ),
        evidence,
    )

    assert insufficient.selected_evidence_unit_ids == (selected_id,)
    assert insufficient.pruning_strategy is PruningStrategy.PARTIAL_EVIDENCE_SENTENCE
    assert factoid.selected_evidence_unit_ids == (selected_id,)
    assert factoid.pruning_strategy is PruningStrategy.SUFFICIENT_PATH_SENTENCE


def test_sufficient_bridge_expands_only_immediate_same_chunk_neighbors() -> None:
    evidence = _scored_units()
    decision = _decision(
        evidence,
        sufficient=True,
        question_type=QuestionType.BRIDGE,
        requested_ids=(evidence[2].evidence_unit.evidence_unit_id,),
    )

    context = ResearchContextPrunerV1().prune(decision, evidence)

    assert context.selected_evidence_unit_ids == tuple(
        evidence[index].evidence_unit.evidence_unit_id for index in (1, 2, 3)
    )
    assert context.pruning_strategy is PruningStrategy.SUFFICIENT_BRIDGE_NEIGHBOR_SENTENCE
    assert context.diagnostics.neighbor_evidence_unit_count == 2


def test_bridge_expansion_stops_at_historical_six_unit_cap_in_input_order() -> None:
    evidence = _scored_units()
    decision = _decision(
        evidence,
        sufficient=True,
        question_type=QuestionType.BRIDGE,
        requested_ids=tuple(evidence[index].evidence_unit.evidence_unit_id for index in (1, 3, 5)),
    )

    context = ResearchContextPrunerV1().prune(decision, evidence)

    assert context.selected_evidence_unit_ids == tuple(
        evidence[index].evidence_unit.evidence_unit_id for index in range(6)
    )
    assert evidence[6].evidence_unit.evidence_unit_id in context.removed_evidence_unit_ids


def test_sorting_rendering_and_character_token_estimate_are_exact() -> None:
    evidence = _scored_units()
    first = evidence[5].model_copy(
        update={"evidence_unit": evidence[5].evidence_unit.model_copy(update={"retrieval_rank": 1})}
    )
    second = evidence[1]
    decision = _decision(
        (first, second),
        sufficient=True,
        question_type=QuestionType.FACTOID,
        requested_ids=(second.evidence_unit.evidence_unit_id, first.evidence_unit.evidence_unit_id),
    )

    context = ResearchContextPrunerV1().prune(decision, (second, first))
    expected = (
        "[Title: Atlas | Chunk: atlas | Sentence: 5]\nSentence 5.\n\n"
        "[Title: Atlas | Chunk: atlas | Sentence: 1]\nSentence 1."
    )

    assert context.selected_evidence_unit_ids == (
        first.evidence_unit.evidence_unit_id,
        second.evidence_unit.evidence_unit_id,
    )
    assert context.selected_context_text == expected
    assert context.estimated_context_tokens == math.ceil(len(expected) / 4)


def test_empty_context_and_all_strategy_labels() -> None:
    evidence = _scored_units()
    empty = ResearchContextPrunerV1().prune(
        _decision(
            evidence,
            sufficient=False,
            question_type=QuestionType.FACTOID,
            requested_ids=("missing::s0",),
        ),
        evidence,
    )

    assert empty.pruning_strategy is PruningStrategy.EMPTY_EVIDENCE
    assert empty.selected_context_text == ""
    assert empty.estimated_context_tokens == 0
    assert set(PruningStrategy) == {
        PruningStrategy.SUFFICIENT_BRIDGE_NEIGHBOR_SENTENCE,
        PruningStrategy.SUFFICIENT_PATH_SENTENCE,
        PruningStrategy.PARTIAL_EVIDENCE_SENTENCE,
        PruningStrategy.EMPTY_EVIDENCE,
    }


def test_contracts_are_immutable_and_reject_invalid_inputs() -> None:
    evidence = _scored_units()
    decision = _decision(
        evidence,
        sufficient=True,
        question_type=QuestionType.FACTOID,
        requested_ids=(evidence[0].evidence_unit.evidence_unit_id,),
    )
    context = ResearchContextPrunerV1().prune(decision, evidence)

    with pytest.raises(ValidationError):
        context.selected_context_text = "replacement"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ResearchContextPrunerV1Config(max_expanded_units=5)
    with pytest.raises(ContextPruningError, match="sufficiency decision"):
        ResearchContextPrunerV1().prune(object(), evidence)  # type: ignore[arg-type]
    with pytest.raises(ContextPruningError, match="Component 04"):
        ResearchContextPrunerV1().prune(decision, (object(),))  # type: ignore[arg-type]


def test_pruner_wraps_invalid_and_unexpected_internal_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _scored_units()
    decision = _decision(
        evidence,
        sufficient=True,
        question_type=QuestionType.FACTOID,
        requested_ids=(evidence[0].evidence_unit.evidence_unit_id,),
    )
    pruner = ResearchContextPrunerV1()

    def invalid(*_args: object, **_kwargs: object) -> object:
        raise ValueError("invalid context")

    monkeypatch.setattr(pruner, "_prune", invalid)
    with pytest.raises(ContextPruningError, match="invalid context"):
        pruner.prune(decision, evidence)

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(pruner, "_prune", unexpected)
    with pytest.raises(ContextPruningError, match="context pruning failed"):
        pruner.prune(decision, evidence)


def test_protocol_instrumentation_and_no_gold_inference_fields() -> None:
    evidence = _scored_units()
    decision = _decision(
        evidence,
        sufficient=True,
        question_type=QuestionType.FACTOID,
        requested_ids=(evidence[0].evidence_unit.evidence_unit_id,),
    )
    sink = InMemoryInstrumentationSink()
    pruner = ResearchContextPrunerV1(instrumentation_sink=sink)

    pruner.prune(
        decision,
        evidence,
        trace_context=TraceContext.start(run_id="context-pruner-test", question_id="question-one"),
    )

    assert isinstance(pruner, ContextPrunerProtocol)
    assert [event.event_type for event in sink.events] == ["epsa.context_pruning.completed"]
    assert "gold" not in " ".join(PrunedContext.model_fields).casefold()
