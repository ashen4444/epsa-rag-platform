"""Focused Component 06 research-v1 behavioral tests."""

from __future__ import annotations

import pytest

from epsa_rag.core.exceptions import EvidencePathSearchError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import QuestionInput
from epsa_rag.epsa.chunk_analysis import RuleBasedCandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.chunk_analysis.models import CanonicalRetrievedChunk
from epsa_rag.epsa.evidence_graph import EvidenceGraphBuilderV1
from epsa_rag.epsa.evidence_path_search import (
    EvidencePathSearcherProtocol,
    EvidencePathSearcherV1,
    EvidencePathSearcherV1Config,
)
from epsa_rag.epsa.evidence_scoring import RuleBasedEvidenceScorerV1
from epsa_rag.epsa.evidence_units import RuleBasedEvidenceUnitExtractor
from epsa_rag.epsa.question_analysis import AnswerType, QuestionType, RuleBasedQuestionAnalyzer
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.evaluation.components.evidence_path_search import evaluate_evidence_paths
from epsa_rag.instrumentation import InMemoryInstrumentationSink, TraceContext


def _scored_units():
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
    scorer = RuleBasedEvidenceScorerV1()
    return analysis, (scorer.score(first, analysis), scorer.score(second, analysis))


def test_bridge_path_preserves_all_upstream_provenance_and_emits_event() -> None:
    analysis, scored = _scored_units()
    graph = EvidenceGraphBuilderV1().build(analysis, scored)
    sink = InMemoryInstrumentationSink()
    searcher = EvidencePathSearcherV1(instrumentation_sink=sink)

    paths = searcher.search_paths(
        graph,
        analysis,
        trace_context=TraceContext.start(run_id="path-test", question_id="question-one"),
    )

    assert isinstance(searcher, EvidencePathSearcherProtocol)
    path = next(path for path in paths if path.answer_candidate == "London")
    assert path.question_type is QuestionType.BRIDGE
    assert path.evidence_unit_ids == ("inception::s0", "christopher-nolan::s0")
    assert path.metadata.source_graph == graph.metadata
    assert path.metadata.configuration_fingerprint == searcher.config.fingerprint()
    assert [item.metadata.version for item in path.scored_evidence_units] == [
        "research_v1",
        "research_v1",
    ]
    assert path.metadata.makes_sufficiency_decision is False
    assert path.score > 0
    assert [event.event_type for event in sink.events] == ["epsa.evidence_path_search.completed"]


def test_no_path_and_non_positive_max_paths_are_safe_and_deterministic() -> None:
    analysis, scored = _scored_units()
    graph = EvidenceGraphBuilderV1().build(analysis, scored)
    searcher = EvidencePathSearcherV1()

    assert searcher.search_paths(graph, analysis, 0) == []
    assert searcher.search_paths(graph, analysis, -3) == []
    assert [path.model_dump_json() for path in searcher.search_paths(graph, analysis)] == [
        path.model_dump_json() for path in searcher.search_paths(graph, analysis)
    ]


def test_paths_preserve_long_component_05_edge_identifiers() -> None:
    analysis, scored = _scored_units()
    long_entity = "Long Entity " + ("x" * 300)
    long_unit = scored[1].evidence_unit.model_copy(
        update={"entities": ("Christopher Nolan", long_entity)}
    )
    long_scored = scored[1].model_copy(update={"evidence_unit": long_unit})
    graph = EvidenceGraphBuilderV1().build(analysis, (scored[0], long_scored))

    paths = EvidencePathSearcherV1().search_paths(graph, analysis)

    assert paths
    assert any(len(edge_id) > 255 for path in paths for edge_id in path.edge_ids)


def test_yes_no_paths_never_decide_polarity() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze("Was Inception directed by Christopher Nolan?")
    assert analysis.question_type is QuestionType.YES_NO
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
    extracted = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, chunk, analysis)[0]
    unit = extracted.model_copy(
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

    assert paths
    assert paths[0].answer_candidate is None
    assert all(path.metadata.does_not_decide_yes_no for path in paths)
    assert all(path.answer_type is AnswerType.BOOLEAN for path in paths)


def test_contract_mismatch_and_mutable_research_constants_are_rejected() -> None:
    analysis, scored = _scored_units()
    graph = EvidenceGraphBuilderV1().build(analysis, scored)
    searcher = EvidencePathSearcherV1()

    with pytest.raises(EvidencePathSearchError, match="max_paths"):
        searcher.search_paths(graph, analysis, True)  # type: ignore[arg-type]
    with pytest.raises(EvidencePathSearchError, match="question analysis"):
        searcher.search_paths(graph, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must remain frozen"):
        EvidencePathSearcherV1Config(relation_match_weight=0.1)


def test_inference_evaluator_runs_components_01_through_06_without_gold_fields() -> None:
    text = "Inception was directed by Christopher Nolan."
    input_item = InferenceRetrieval(
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
    )

    summary, traces, events = evaluate_evidence_paths((input_item,), run_id="path-eval")

    assert summary.completed_questions == 1
    assert summary.diagnostics.paths_missing_scored_evidence_provenance == 0
    assert traces[0].evidence_graph is not None
    assert "supporting" not in traces[0].model_dump_json()
    assert [event.event_type for event in events] == [
        "epsa.question_analysis.completed",
        "epsa.chunk_analysis.completed",
        "epsa.evidence_units.completed",
        "epsa.evidence_scoring.completed",
        "epsa.evidence_graph.completed",
        "epsa.evidence_path_search.completed",
    ]
