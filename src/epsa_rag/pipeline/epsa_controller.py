"""Deterministic orchestration of EPSA Components 01 through 09."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Literal

from epsa_rag.core.models import RankedParagraphChunk
from epsa_rag.epsa.chunk_analysis import RuleBasedV2CandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.context_pruning import ContextPrunerProtocol, ResearchContextPrunerV1
from epsa_rag.epsa.evidence_graph import EvidenceGraphBuilderV1
from epsa_rag.epsa.evidence_graph.protocols import EvidenceGraphBuilderProtocol
from epsa_rag.epsa.evidence_path_search import (
    EvidencePathSearcherProtocol,
    EvidencePathSearcherV1,
)
from epsa_rag.epsa.evidence_scoring import RuleBasedEvidenceScorerV1
from epsa_rag.epsa.evidence_scoring.protocols import EvidenceScorerProtocol
from epsa_rag.epsa.evidence_units import RuleBasedV2EvidenceUnitExtractor
from epsa_rag.epsa.evidence_units.protocols import EvidenceUnitExtractorProtocol
from epsa_rag.epsa.next_hop_query import (
    NextHopQueryGeneratorProtocol,
    RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1,
)
from epsa_rag.epsa.question_analysis import RuleBasedQuestionAnalyzer
from epsa_rag.epsa.question_analysis.protocols import QuestionAnalyzerProtocol
from epsa_rag.epsa.sufficiency_decision import (
    RuleBasedSufficiencyEngineV1,
    SufficiencyDecisionEngineProtocol,
)
from epsa_rag.instrumentation import InstrumentationSink, TraceContext
from epsa_rag.pipeline.config import EPSAControllerConfig
from epsa_rag.pipeline.epsa_models import EPSAPassResult
from epsa_rag.pipeline.protocols import ContextualChunkAnalyzerProtocol


class EPSAController:
    """Run the accepted deterministic component chain over one candidate ranking."""

    def __init__(
        self,
        *,
        question_analyzer: QuestionAnalyzerProtocol,
        chunk_analyzer: ContextualChunkAnalyzerProtocol,
        evidence_unit_extractor: EvidenceUnitExtractorProtocol,
        evidence_scorer: EvidenceScorerProtocol,
        graph_builder: EvidenceGraphBuilderProtocol,
        path_searcher: EvidencePathSearcherProtocol,
        sufficiency_engine: SufficiencyDecisionEngineProtocol,
        context_pruner: ContextPrunerProtocol,
        next_hop_generator: NextHopQueryGeneratorProtocol,
        config: EPSAControllerConfig | None = None,
    ) -> None:
        self.question_analyzer = question_analyzer
        self.chunk_analyzer = chunk_analyzer
        self.evidence_unit_extractor = evidence_unit_extractor
        self.evidence_scorer = evidence_scorer
        self.graph_builder = graph_builder
        self.path_searcher = path_searcher
        self.sufficiency_engine = sufficiency_engine
        self.context_pruner = context_pruner
        self.next_hop_generator = next_hop_generator
        self.config = config or EPSAControllerConfig()

    @classmethod
    def research_v1(
        cls,
        *,
        config: EPSAControllerConfig | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> EPSAController:
        """Construct the exact component versions accepted by current evaluations."""

        return cls(
            question_analyzer=RuleBasedQuestionAnalyzer(
                instrumentation_sink=instrumentation_sink
            ),
            chunk_analyzer=RuleBasedV2CandidateChunkEvidenceAnalyzer(
                instrumentation_sink=instrumentation_sink
            ),
            evidence_unit_extractor=RuleBasedV2EvidenceUnitExtractor(
                instrumentation_sink=instrumentation_sink
            ),
            evidence_scorer=RuleBasedEvidenceScorerV1(
                instrumentation_sink=instrumentation_sink
            ),
            graph_builder=EvidenceGraphBuilderV1(
                instrumentation_sink=instrumentation_sink
            ),
            path_searcher=EvidencePathSearcherV1(
                instrumentation_sink=instrumentation_sink
            ),
            sufficiency_engine=RuleBasedSufficiencyEngineV1(
                instrumentation_sink=instrumentation_sink
            ),
            context_pruner=ResearchContextPrunerV1(
                instrumentation_sink=instrumentation_sink
            ),
            next_hop_generator=RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1(
                instrumentation_sink=instrumentation_sink
            ),
            config=config,
        )

    def run_pass(
        self,
        *,
        question: str,
        ranked_chunks: Sequence[RankedParagraphChunk],
        pass_number: Literal[1, 2],
        generate_next_hop: bool,
        trace_context: TraceContext | None = None,
    ) -> EPSAPassResult:
        """Execute Components 01-08 and optionally Component 09 once."""

        started = perf_counter()
        chunks = tuple(ranked_chunks)
        analysis = self.question_analyzer.analyze(question, trace_context=trace_context)
        candidates = self.chunk_analyzer.analyze_batch(
            chunks, analysis, trace_context=trace_context
        )
        evidence_units = tuple(
            unit
            for candidate, chunk in zip(candidates, chunks, strict=True)
            for unit in self.evidence_unit_extractor.extract_from_chunk(
                candidate,
                chunk,
                analysis,
                trace_context=trace_context,
            )
        )
        scored = self.evidence_scorer.score_many(
            evidence_units, analysis, trace_context=trace_context
        )
        graph = self.graph_builder.build(analysis, scored, trace_context=trace_context)
        paths = tuple(
            self.path_searcher.search_paths(
                graph,
                analysis,
                self.config.max_paths,
                trace_context=trace_context,
            )
        )
        decision = self.sufficiency_engine.decide(
            analysis, graph, paths, trace_context=trace_context
        )
        pruned = self.context_pruner.prune(decision, scored, trace_context=trace_context)
        next_query = (
            self.next_hop_generator.generate(
                analysis,
                decision,
                graph,
                paths,
                trace_context=trace_context,
            )
            if generate_next_hop
            else None
        )
        return EPSAPassResult(
            configuration_fingerprint=self.config.fingerprint(),
            pass_number=pass_number,
            input_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
            question_analysis=analysis,
            chunk_evidence=candidates,
            evidence_units=evidence_units,
            scored_evidence_units=scored,
            evidence_graph=graph,
            candidate_paths=paths,
            sufficiency_decision=decision,
            pruned_context=pruned,
            next_hop_query=next_query,
            latency_ms=(perf_counter() - started) * 1000,
        )
