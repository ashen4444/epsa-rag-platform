"""Inference-only structural diagnostics for EPSA Components 01 through 05."""

from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.exceptions import (
    ChunkAnalysisError,
    EvidenceGraphBuildError,
    EvidenceScoringError,
    EvidenceUnitExtractionError,
    QuestionAnalysisError,
)
from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel
from epsa_rag.epsa.chunk_analysis import RuleBasedV2CandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.chunk_analysis.models import CandidateChunkEvidence
from epsa_rag.epsa.evidence_graph import EvidenceGraph, EvidenceGraphBuilderV1, GraphNodeType
from epsa_rag.epsa.evidence_scoring import RuleBasedEvidenceScorerV1, ScoredEvidenceUnit
from epsa_rag.epsa.evidence_units import EvidenceUnit, RuleBasedV2EvidenceUnitExtractor
from epsa_rag.epsa.question_analysis import QuestionAnalysis, RuleBasedQuestionAnalyzer
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.instrumentation import (
    InMemoryInstrumentationSink,
    InstrumentationEvent,
    NoOpInstrumentationSink,
    TraceContext,
)


class EvidenceGraphEvaluationConfig(ConfigModel):
    evaluator_version: Literal["evidence-graph-evaluation-v1"] = "evidence-graph-evaluation-v1"
    retrieval_depth: int = Field(default=10, ge=1)
    retain_instrumentation_events: bool = True


class EvidenceGraphTrace(ContractModel):
    question_id: Identifier
    status: Literal["completed", "failed"]
    question_analysis: QuestionAnalysis | None
    chunk_evidence: tuple[CandidateChunkEvidence, ...]
    evidence_units: tuple[EvidenceUnit, ...]
    scored_evidence_units: tuple[ScoredEvidenceUnit, ...]
    evidence_graph: EvidenceGraph | None
    error_type: str | None = None
    latency_ms: float = Field(ge=0)


class EvidenceGraphDiagnostics(ContractModel):
    scored_evidence_units: int = Field(ge=0)
    graphs: int = Field(ge=0)
    nodes: int = Field(ge=0)
    edges: int = Field(ge=0)
    sentence_nodes: int = Field(ge=0)
    seed_anchor_edges: int = Field(ge=0)
    edges_missing_evidence_provenance: int = Field(ge=0)
    sentence_nodes_missing_scored_evidence: int = Field(ge=0)
    node_type_counts: dict[str, int]
    edge_type_counts: dict[str, int]


class EvidenceGraphRunSummary(ContractModel):
    schema_version: Literal["evidence-graph-run-v1"] = "evidence-graph-run-v1"
    run_id: Identifier
    configuration: EvidenceGraphEvaluationConfig
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: tuple[Identifier, ...]
    completed_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    diagnostics: EvidenceGraphDiagnostics


def evaluate_evidence_graphs(
    inputs: tuple[InferenceRetrieval, ...],
    *,
    run_id: str,
    config: EvidenceGraphEvaluationConfig | None = None,
) -> tuple[
    EvidenceGraphRunSummary, tuple[EvidenceGraphTrace, ...], tuple[InstrumentationEvent, ...]
]:
    if not inputs:
        raise ValueError("Component 05 evaluation requires inference inputs")
    selected = config or EvidenceGraphEvaluationConfig()
    sink = (
        InMemoryInstrumentationSink()
        if selected.retain_instrumentation_events
        else NoOpInstrumentationSink()
    )
    questions = RuleBasedQuestionAnalyzer(instrumentation_sink=sink)
    chunks = RuleBasedV2CandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    extractor = RuleBasedV2EvidenceUnitExtractor(instrumentation_sink=sink)
    scorer = RuleBasedEvidenceScorerV1(instrumentation_sink=sink)
    builder = EvidenceGraphBuilderV1(instrumentation_sink=sink)
    traces: list[EvidenceGraphTrace] = []
    for item in inputs:
        started = perf_counter()
        analysis = None
        candidates = []
        units = []
        scored = ()
        try:
            context = TraceContext.start(run_id=run_id, question_id=item.question.question_id)
            analysis = questions.analyze(item.question.text, trace_context=context)
            ranked = item.chunks[: selected.retrieval_depth]
            candidates.extend(chunks.analyze_batch(ranked, analysis, trace_context=context))
            for candidate, chunk in zip(candidates, ranked, strict=True):
                units.extend(
                    extractor.extract_from_chunk(candidate, chunk, analysis, trace_context=context)
                )
            scored = scorer.score_many(units, analysis, trace_context=context)
            graph = builder.build(analysis, scored, trace_context=context)
        except (
            QuestionAnalysisError,
            ChunkAnalysisError,
            EvidenceUnitExtractionError,
            EvidenceScoringError,
            EvidenceGraphBuildError,
            ValueError,
        ) as error:
            traces.append(
                EvidenceGraphTrace(
                    question_id=item.question.question_id,
                    status="failed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=scored,
                    evidence_graph=None,
                    error_type=type(error).__name__,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        else:
            traces.append(
                EvidenceGraphTrace(
                    question_id=item.question.question_id,
                    status="completed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=scored,
                    evidence_graph=graph,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
    graphs = tuple(t.evidence_graph for t in traces if t.evidence_graph is not None)
    nodes = Counter(n.node_type.value for g in graphs for n in g.nodes)
    edges = Counter(e.edge_type.value for g in graphs for e in g.edges)
    summary = EvidenceGraphRunSummary(
        run_id=run_id,
        configuration=selected,
        configuration_fingerprint=selected.fingerprint(),
        question_ids=tuple(i.question.question_id for i in inputs),
        completed_questions=sum(t.status == "completed" for t in traces),
        failed_questions=sum(t.status == "failed" for t in traces),
        diagnostics=EvidenceGraphDiagnostics(
            scored_evidence_units=sum(len(t.scored_evidence_units) for t in traces),
            graphs=len(graphs),
            nodes=sum(len(g.nodes) for g in graphs),
            edges=sum(len(g.edges) for g in graphs),
            sentence_nodes=sum(
                n.node_type is GraphNodeType.SENTENCE for g in graphs for n in g.nodes
            ),
            seed_anchor_edges=edges["seed_entity_to_sentence"],
            edges_missing_evidence_provenance=sum(
                e.evidence_unit_id is None for g in graphs for e in g.edges
            ),
            sentence_nodes_missing_scored_evidence=sum(
                n.scored_evidence is None
                for g in graphs
                for n in g.nodes
                if n.node_type is GraphNodeType.SENTENCE
            ),
            node_type_counts=dict(sorted(nodes.items())),
            edge_type_counts=dict(sorted(edges.items())),
        ),
    )
    return (
        summary,
        tuple(traces),
        sink.events if isinstance(sink, InMemoryInstrumentationSink) else (),
    )
