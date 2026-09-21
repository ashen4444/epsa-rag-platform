"""Focused behavioral tests for EPSA Component 05 research-v1."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from epsa_rag.core.exceptions import EvidenceGraphBuildError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import QuestionInput
from epsa_rag.epsa.chunk_analysis import RuleBasedCandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.chunk_analysis.models import CanonicalRetrievedChunk
from epsa_rag.epsa.evidence_graph import (
    EvidenceGraph,
    EvidenceGraphBuilderV1,
    EvidenceGraphBuilderV1Config,
    EvidenceGraphMetadata,
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    stable_edge_id,
    stable_node_id,
    stable_sentence_node_id,
)
from epsa_rag.epsa.evidence_graph.protocols import EvidenceGraphBuilderProtocol
from epsa_rag.epsa.evidence_scoring import RuleBasedEvidenceScorerV1
from epsa_rag.epsa.evidence_units import RuleBasedEvidenceUnitExtractor
from epsa_rag.epsa.question_analysis import (
    AnswerType,
    QuestionType,
    RuleBasedQuestionAnalyzer,
)
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.evaluation.components.evidence_graph import evaluate_evidence_graphs
from epsa_rag.instrumentation import InMemoryInstrumentationSink, TraceContext


def _scored_units():
    analysis = RuleBasedQuestionAnalyzer().analyze("Where was the director of Inception born?")
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
    first_unit = RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, chunk, analysis)[0]
    first = first_unit.model_copy(
        update={
            "entities": ("Inception", "Christopher Nolan"),
            "relation_hints": ("directed",),
            "answer_type_candidates": (AnswerType.PERSON,),
            "question_entity_overlap": ("Inception",),
        }
    )
    second_text = "Christopher Nolan was born in London."
    second_resolution = first.metadata.resolution.model_copy(
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
            "metadata": first.metadata.model_copy(update={"resolution": second_resolution}),
        }
    )
    scorer = RuleBasedEvidenceScorerV1()
    return analysis, (scorer.score(first, analysis), scorer.score(second, analysis))


def test_builder_creates_documented_research_v1_graph_and_preserves_provenance() -> None:
    analysis, scored = _scored_units()
    graph = EvidenceGraphBuilderV1().build(analysis, scored)

    assert isinstance(EvidenceGraphBuilderV1(), EvidenceGraphBuilderProtocol)
    assert graph.metadata.version == "research_v1"
    assert graph.metadata.makes_sufficiency_decision is False
    assert graph.metadata.inference_gold_labels_present is False
    assert stable_node_id("entity", "Inception") in {node.node_id for node in graph.nodes}
    assert stable_node_id("entity", "Christopher Nolan") in {node.node_id for node in graph.nodes}
    assert stable_node_id("entity", "London") in {node.node_id for node in graph.nodes}
    assert stable_node_id("chunk", "inception") in {node.node_id for node in graph.nodes}
    assert stable_node_id("title", "Inception") in {node.node_id for node in graph.nodes}
    assert stable_sentence_node_id("inception::s0") in {node.node_id for node in graph.nodes}

    assert {edge.edge_type for edge in graph.edges} == set(GraphEdgeType)
    sentence = graph.node_by_id(stable_sentence_node_id("inception::s0"))
    assert sentence.node_type is GraphNodeType.SENTENCE
    assert sentence.scored_evidence is not None
    assert sentence.scored_evidence.evidence_unit.evidence_unit_id == "inception::s0"
    assert sentence.scored_evidence.evidence_unit.chunk_id == "inception"
    assert sentence.scored_evidence.metadata.version == "research_v1"
    assert "supporting" not in graph.model_dump_json()


def test_ids_serialization_and_duplicate_edge_weight_are_deterministic() -> None:
    analysis, scored = _scored_units()
    low, original_high = scored
    high = original_high.model_copy(update={"final_score": 0.99})
    builder = EvidenceGraphBuilderV1()
    graph = builder.build(analysis, (low, high, high))

    assert stable_node_id("entity", "Christopher Nolan") == "entity::christopher_nolan"
    assert stable_sentence_node_id("inception::s0") == "sentence::inception::s0"
    assert stable_edge_id(
        "sentence::inception::s0",
        "entity::inception",
        "sentence_mentions_entity",
        "inception::s0",
    ).startswith("edge::sentence_mentions_entity::")
    assert graph.model_dump_json() == builder.build(analysis, (low, high, high)).model_dump_json()
    nolan_mention = next(
        edge
        for edge in graph.edges
        if edge.edge_type is GraphEdgeType.SENTENCE_MENTIONS_ENTITY
        and edge.source_id == "sentence::christopher-nolan::s0"
        and edge.target_id == "entity::christopher_nolan"
    )
    assert nolan_mention.weight == 0.99
    assert [node.node_id for node in graph.nodes] == sorted(node.node_id for node in graph.nodes)
    assert [edge.edge_id for edge in graph.edges] == sorted(edge.edge_id for edge in graph.edges)


def test_builder_merges_duplicate_evidence_edges_at_the_highest_weight() -> None:
    analysis, scored = _scored_units()
    original = scored[0]
    low = original.model_copy(update={"final_score": 0.2})
    high = original.model_copy(update={"final_score": 0.8})

    graph = EvidenceGraphBuilderV1().build(analysis, (low, high))

    chunk_to_sentence = next(
        edge for edge in graph.edges if edge.edge_type is GraphEdgeType.CHUNK_TO_SENTENCE
    )
    assert chunk_to_sentence.weight == 0.8
    assert len(
        [edge for edge in graph.edges if edge.edge_type is GraphEdgeType.CHUNK_TO_SENTENCE]
    ) == 1


def test_config_and_graph_contracts_reject_invalid_state() -> None:
    analysis, scored = _scored_units()
    with pytest.raises(ValidationError, match="minimum anchor weight"):
        EvidenceGraphBuilderV1Config(minimum_anchor_weight=0.2)
    assert EvidenceGraphBuilderV1().config.minimum_anchor_weight == 0.1

    with pytest.raises(ValidationError, match="sentence nodes require"):
        GraphNode(node_id="sentence::missing", node_type=GraphNodeType.SENTENCE, label="missing")
    with pytest.raises(ValidationError, match="only sentence nodes"):
        GraphNode(
            node_id="chunk::invalid",
            node_type=GraphNodeType.CHUNK,
            label="invalid",
            scored_evidence=scored[0],
        )
    with pytest.raises(ValidationError, match="finite number"):
        GraphEdge(
            edge_id="edge::invalid",
            source_id="entity::one",
            target_id="entity::two",
            edge_type=GraphEdgeType.POSSIBLE_BRIDGE,
            weight=float("inf"),
        )

    metadata = EvidenceGraphMetadata(
        configuration_fingerprint="a" * 64,
        expected_answer_type=AnswerType.LOCATION,
        required_relation_hints=(),
        num_scored_evidence_units=0,
    )
    node = GraphNode(node_id="entity::one", node_type=GraphNodeType.ENTITY, label="one")
    with pytest.raises(ValidationError, match="seed nodes must exist"):
        EvidenceGraph(
            nodes=(node,),
            edges=(),
            question_type=QuestionType.FACTOID,
            seed_entity_node_ids=("entity::missing",),
            evidence_unit_node_ids=(),
            entity_node_ids=(),
            metadata=metadata,
        )

    graph = EvidenceGraphBuilderV1().build(analysis, scored)
    with pytest.raises(KeyError, match="missing"):
        graph.node_by_id("entity::missing")


def test_graph_contract_rejects_duplicate_unordered_and_dangling_references() -> None:
    metadata = EvidenceGraphMetadata(
        configuration_fingerprint="b" * 64,
        expected_answer_type=AnswerType.LOCATION,
        required_relation_hints=(),
        num_scored_evidence_units=0,
    )
    one = GraphNode(node_id="entity::one", node_type=GraphNodeType.ENTITY, label="one")
    two = GraphNode(node_id="entity::two", node_type=GraphNodeType.ENTITY, label="two")
    edge = GraphEdge(
        edge_id="edge::one",
        source_id=one.node_id,
        target_id=two.node_id,
        edge_type=GraphEdgeType.POSSIBLE_BRIDGE,
        weight=0.5,
    )
    common = {
        "question_type": QuestionType.FACTOID,
        "seed_entity_node_ids": (),
        "evidence_unit_node_ids": (),
        "entity_node_ids": (),
        "metadata": metadata,
    }

    with pytest.raises(ValidationError, match="node IDs must be unique"):
        EvidenceGraph(nodes=(one, one), edges=(), **common)
    with pytest.raises(ValidationError, match="edge IDs must be unique"):
        EvidenceGraph(nodes=(one, two), edges=(edge, edge), **common)
    with pytest.raises(ValidationError, match="nodes must be ordered"):
        EvidenceGraph(nodes=(two, one), edges=(), **common)

    later_edge = edge.model_copy(update={"edge_id": "edge::two"})
    with pytest.raises(ValidationError, match="edges must be ordered"):
        EvidenceGraph(nodes=(one, two), edges=(later_edge, edge), **common)
    dangling = edge.model_copy(update={"target_id": "entity::missing"})
    with pytest.raises(ValidationError, match="edges must reference"):
        EvidenceGraph(nodes=(one, two), edges=(dangling,), **common)
    with pytest.raises(ValidationError, match="evidence sentence nodes"):
        EvidenceGraph(
            nodes=(one,),
            edges=(),
            evidence_unit_node_ids=("sentence::missing",),
            **{key: value for key, value in common.items() if key != "evidence_unit_node_ids"},
        )
    with pytest.raises(ValidationError, match="entity nodes must exist"):
        EvidenceGraph(
            nodes=(one,),
            edges=(),
            entity_node_ids=("entity::missing",),
            **{key: value for key, value in common.items() if key != "entity_node_ids"},
        )
    with pytest.raises(ValueError, match="graph edge weight must be finite"):
        GraphEdge.require_finite_weight(float("inf"))


def test_builder_rejects_invalid_contracts_and_emits_failure_event() -> None:
    analysis, scored = _scored_units()
    sink = InMemoryInstrumentationSink()
    builder = EvidenceGraphBuilderV1(instrumentation_sink=sink)
    context = TraceContext.start(run_id="graph-test", question_id="question-one")

    with pytest.raises(EvidenceGraphBuildError, match="Component 04 contract"):
        builder.build(analysis, (object(),), trace_context=context)  # type: ignore[arg-type]
    with pytest.raises(EvidenceGraphBuildError, match="Component 01 contract"):
        builder.build(object(), scored, trace_context=context)  # type: ignore[arg-type]
    with pytest.raises(EvidenceGraphBuildError, match="Component 04 sequence"):
        builder.build(analysis, "not a sequence", trace_context=context)  # type: ignore[arg-type]
    assert [event.event_type for event in sink.events] == [
        "epsa.evidence_graph.failed",
        "epsa.evidence_graph.failed",
        "epsa.evidence_graph.failed",
    ]


def test_inference_evaluator_runs_components_01_through_05_without_gold_fields() -> None:
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
    summary, traces, events = evaluate_evidence_graphs((input_item,), run_id="graph-eval")

    assert summary.completed_questions == 1
    assert summary.diagnostics.graphs == 1
    assert summary.diagnostics.sentence_nodes_missing_scored_evidence == 0
    assert traces[0].evidence_graph is not None
    assert "supporting" not in traces[0].model_dump_json()
    assert [event.event_type for event in events] == [
        "epsa.question_analysis.completed",
        "epsa.chunk_analysis.completed",
        "epsa.evidence_units.completed",
        "epsa.evidence_scoring.completed",
        "epsa.evidence_graph.completed",
    ]


def test_evaluator_requires_inference_inputs() -> None:
    with pytest.raises(ValueError, match="requires inference inputs"):
        evaluate_evidence_graphs((), run_id="graph-eval-empty")
