"""Deterministic reconstructed Component 09 next-hop query generation."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

from pydantic import ValidationError

from epsa_rag.core.exceptions import NextHopQueryGenerationError
from epsa_rag.epsa.evidence_graph.models import EvidenceGraph
from epsa_rag.epsa.evidence_path_search.models import EvidencePath
from epsa_rag.epsa.next_hop_query.config import ReconstructedNextHopQueryGeneratorConfig
from epsa_rag.epsa.next_hop_query.models import (
    NextHopQuery,
    NextHopQueryMetadata,
    NextHopQuerySource,
    NextHopQueryType,
    QueryReasonCode,
)
from epsa_rag.epsa.question_analysis.models import EntityMention, QuestionAnalysis, QuestionType
from epsa_rag.epsa.sufficiency_decision.models import SufficiencyDecision
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext


class RuleBasedNextHopQueryGeneratorReconstructedV1:
    """Propose a conservative retrieval query without claiming historical parity.

    The original Component 09 source was not recovered. This generator is a
    deterministic compatibility reconstruction named ``research_v1_reconstructed``.
    It never executes retrieval or changes the question used by later EPSA runs.
    """

    def __init__(
        self,
        *,
        config: ReconstructedNextHopQueryGeneratorConfig | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> None:
        self._config = config or ReconstructedNextHopQueryGeneratorConfig()
        self._sink = instrumentation_sink

    @property
    def config(self) -> ReconstructedNextHopQueryGeneratorConfig:
        """Return the immutable reconstructed-policy configuration."""

        return self._config

    def generate(
        self,
        question_analysis: QuestionAnalysis,
        sufficiency_decision: SufficiencyDecision,
        evidence_graph: EvidenceGraph,
        evidence_paths: Sequence[EvidencePath],
        *,
        trace_context: TraceContext | None = None,
    ) -> NextHopQuery:
        """Return a deterministic next-hop proposal or an explicit no-query result."""

        started = perf_counter()
        try:
            paths = self._validate_inputs(
                question_analysis,
                sufficiency_decision,
                evidence_graph,
                evidence_paths,
            )
            result = self._generate(
                question_analysis,
                sufficiency_decision,
                evidence_graph,
                paths,
            )
        except NextHopQueryGenerationError as error:
            self._emit_failed(error, trace_context, started)
            raise
        except (ValidationError, ValueError, TypeError) as error:
            wrapped = NextHopQueryGenerationError(str(error))
            self._emit_failed(wrapped, trace_context, started)
            raise wrapped from error
        except Exception as error:
            wrapped = NextHopQueryGenerationError("next-hop query generation failed")
            self._emit_failed(wrapped, trace_context, started)
            raise wrapped from error
        self._emit_completed(result, trace_context, started)
        return result

    @staticmethod
    def _validate_inputs(
        analysis: object,
        decision: object,
        graph: object,
        paths: object,
    ) -> tuple[EvidencePath, ...]:
        if not isinstance(analysis, QuestionAnalysis):
            raise NextHopQueryGenerationError("question analysis must use Component 01 contract")
        if not isinstance(decision, SufficiencyDecision):
            raise NextHopQueryGenerationError("sufficiency decision must use Component 07 contract")
        if not isinstance(graph, EvidenceGraph):
            raise NextHopQueryGenerationError("evidence graph must use Component 05 contract")
        if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
            raise NextHopQueryGenerationError("evidence paths must be a sequence")
        typed_paths = tuple(paths)
        if graph.question_type is not analysis.question_type:
            raise NextHopQueryGenerationError(
                "evidence graph question type must match Component 01 input"
            )
        if graph.metadata.expected_answer_type is not analysis.expected_answer_type:
            raise NextHopQueryGenerationError(
                "evidence graph answer type must match Component 01 input"
            )
        expected_relations = tuple(
            dict.fromkeys(hint.relation for hint in analysis.required_relation_hints)
        )
        if graph.metadata.required_relation_hints != expected_relations:
            raise NextHopQueryGenerationError(
                "evidence graph relation hints must match Component 01 input"
            )
        if decision.question_type is not analysis.question_type:
            raise NextHopQueryGenerationError(
                "sufficiency decision question type must match Component 01 input"
            )
        if decision.answer_type is not analysis.expected_answer_type:
            raise NextHopQueryGenerationError(
                "sufficiency decision answer type must match Component 01 input"
            )
        if decision.metadata.source_graph != graph.metadata:
            raise NextHopQueryGenerationError(
                "sufficiency decision graph provenance must match Component 05 input"
            )
        path_ids: list[str] = []
        for path in typed_paths:
            if not isinstance(path, EvidencePath):
                raise NextHopQueryGenerationError("evidence paths must use Component 06 contract")
            if path.question_type is not analysis.question_type:
                raise NextHopQueryGenerationError(
                    "candidate path question type must match Component 01 input"
                )
            if path.metadata.source_graph != graph.metadata:
                raise NextHopQueryGenerationError(
                    "candidate path graph provenance must match Component 05 input"
                )
            path_ids.append(path.path_id)
        if len(path_ids) != len(set(path_ids)):
            raise NextHopQueryGenerationError("candidate path IDs must not be duplicated")
        if set(path_ids) != set(decision.metadata.candidate_path_ids):
            raise NextHopQueryGenerationError(
                "candidate paths must match Component 07 decision provenance"
            )
        return typed_paths

    def _generate(
        self,
        analysis: QuestionAnalysis,
        decision: SufficiencyDecision,
        graph: EvidenceGraph,
        paths: tuple[EvidencePath, ...],
    ) -> NextHopQuery:
        if decision.sufficient:
            return self._no_query(
                analysis,
                decision,
                graph,
                paths,
                QueryReasonCode.SUFFICIENT_DECISION,
                "Current evidence is sufficient; no next-hop retrieval is proposed.",
            )
        if not paths:
            return self._no_query(
                analysis,
                decision,
                graph,
                paths,
                QueryReasonCode.NO_CANDIDATE_PATHS,
                "No candidate evidence path provides a grounded next-hop target.",
            )
        ranked = tuple(sorted(paths, key=lambda path: (-path.score, path.path_id)))
        bridge = self._bridge_query(analysis, ranked)
        if bridge is not None:
            target, relation, path = bridge
            return self._query(
                analysis,
                decision,
                graph,
                paths,
                target=target,
                relation=relation,
                path=path,
                query_type=NextHopQueryType.BRIDGE_ENTITY_RELATION,
                source=NextHopQuerySource.PATH_BRIDGE_ENTITY,
                reason_code=QueryReasonCode.BRIDGE_ENTITY_RELATION,
                reason="A reconstructed bridge-entity relation query targets the partial path.",
                confidence=self._config.bridge_query_confidence,
            )
        comparison = self._comparison_query(analysis, ranked)
        if comparison is not None:
            target, relation, path = comparison
            return self._query(
                analysis,
                decision,
                graph,
                paths,
                target=target,
                relation=relation,
                path=path,
                query_type=NextHopQueryType.COMPARISON_TARGET_RELATION,
                source=NextHopQuerySource.PATH_COMPARISON_TARGET,
                reason_code=QueryReasonCode.COMPARISON_TARGET_RELATION,
                reason=(
                    "A reconstructed comparison-target relation query targets unresolved evidence."
                ),
                confidence=self._config.comparison_query_confidence,
            )
        seed = self._seed_query(analysis, ranked)
        if seed is not None:
            target, relation, path = seed
            return self._query(
                analysis,
                decision,
                graph,
                paths,
                target=target,
                relation=relation,
                path=path,
                query_type=NextHopQueryType.SEED_ENTITY_RELATION,
                source=NextHopQuerySource.QUESTION_SEED,
                reason_code=QueryReasonCode.SEED_ENTITY_RELATION,
                reason="A reconstructed seed-entity relation query is the safe fallback.",
                confidence=self._config.seed_query_confidence,
            )
        reason = (
            QueryReasonCode.NO_RELATION_HINT
            if not analysis.required_relation_hints
            else QueryReasonCode.NO_GROUNDED_TARGET
        )
        message = (
            "No relation hint supports a deterministic next-hop query."
            if reason is QueryReasonCode.NO_RELATION_HINT
            else "No grounded target supports a deterministic next-hop query."
        )
        return self._no_query(analysis, decision, graph, paths, reason, message)

    def _bridge_query(
        self, analysis: QuestionAnalysis, paths: tuple[EvidencePath, ...]
    ) -> tuple[str, str, EvidencePath] | None:
        for path in paths:
            target = path.metadata.bridge_entity
            if path.metadata.path_kind != "bridge_candidate" or not _usable_text(target):
                continue
            relation = _relation_for_path(analysis, path)
            if relation is not None:
                return target, relation, path
        return None

    def _comparison_query(
        self, analysis: QuestionAnalysis, paths: tuple[EvidencePath, ...]
    ) -> tuple[str, str, EvidencePath] | None:
        if analysis.question_type is not QuestionType.COMPARISON:
            return None
        if len(analysis.comparison_targets) != 2:
            return None
        partial_paths = tuple(
            path
            for path in paths
            if path.metadata.path_kind == "comparison_target_partial"
            and _usable_text(path.metadata.comparison_target)
        )
        if not partial_paths:
            return None
        observed = {_normalise(path.metadata.comparison_target or "") for path in partial_paths}
        target_mention = next(
            (
                target
                for target in analysis.comparison_targets
                if _normalise(target.text) not in observed
            ),
            analysis.comparison_targets[0],
        )
        relation = _relation_for_path(analysis, partial_paths[0])
        if relation is None:
            return None
        return target_mention.text, relation, partial_paths[0]

    @staticmethod
    def _seed_query(
        analysis: QuestionAnalysis, paths: tuple[EvidencePath, ...]
    ) -> tuple[str, str, EvidencePath] | None:
        if not analysis.seed_entities or not analysis.required_relation_hints:
            return None
        relation = _relation_for_path(analysis, paths[0])
        if relation is None:
            return None
        target = _first_usable_entity(analysis.seed_entities)
        if target is None:
            return None
        return target.text, relation, paths[0]

    def _query(
        self,
        analysis: QuestionAnalysis,
        decision: SufficiencyDecision,
        graph: EvidenceGraph,
        paths: tuple[EvidencePath, ...],
        *,
        target: str,
        relation: str,
        path: EvidencePath,
        query_type: NextHopQueryType,
        source: NextHopQuerySource,
        reason_code: QueryReasonCode,
        reason: str,
        confidence: float,
    ) -> NextHopQuery:
        target_text = " ".join(target.split())
        relation_text = " ".join(relation.split())
        return NextHopQuery(
            query=f"{target_text}{self._config.query_separator}{relation_text}",
            query_type=query_type,
            source=source,
            target_entity=target_text,
            missing_relation=relation_text,
            expected_answer_type=analysis.expected_answer_type,
            reason=reason,
            confidence=confidence,
            metadata=self._metadata(
                analysis,
                decision,
                graph,
                paths,
                selected_path_id=path.path_id,
                reason_code=reason_code,
            ),
        )

    def _no_query(
        self,
        analysis: QuestionAnalysis,
        decision: SufficiencyDecision,
        graph: EvidenceGraph,
        paths: tuple[EvidencePath, ...],
        reason_code: QueryReasonCode,
        reason: str,
    ) -> NextHopQuery:
        return NextHopQuery(
            query=None,
            query_type=NextHopQueryType.NO_QUERY,
            source=NextHopQuerySource.NO_QUERY,
            reason=reason,
            confidence=0.0,
            metadata=self._metadata(
                analysis,
                decision,
                graph,
                paths,
                selected_path_id=None,
                reason_code=reason_code,
            ),
        )

    def _metadata(
        self,
        analysis: QuestionAnalysis,
        decision: SufficiencyDecision,
        graph: EvidenceGraph,
        paths: tuple[EvidencePath, ...],
        *,
        selected_path_id: str | None,
        reason_code: QueryReasonCode,
    ) -> NextHopQueryMetadata:
        return NextHopQueryMetadata(
            configuration_fingerprint=self._config.fingerprint(),
            source_question_analysis=analysis.metadata,
            source_sufficiency_decision=decision.metadata,
            source_graph=graph.metadata,
            candidate_path_ids=tuple(path.path_id for path in paths),
            selected_path_id=selected_path_id,
            reason_code=reason_code,
        )

    def _emit_completed(
        self, result: NextHopQuery, trace: TraceContext | None, started: float
    ) -> None:
        if self._sink is not None and trace is not None:
            self._sink.emit(
                InstrumentationEvent(
                    context=trace,
                    event_type="epsa.next_hop_query.completed",
                    source="epsa.next_hop_query",
                    source_version="research_v1_reconstructed",
                    payload={
                        "latency_ms": round((perf_counter() - started) * 1000, 6),
                        "query_available": result.query is not None,
                        "query_type": result.query_type.value,
                        "source": result.source.value,
                        "reason_code": result.metadata.reason_code.value,
                        "confidence": result.confidence,
                        "selected_path_present": result.metadata.selected_path_id is not None,
                    },
                )
            )

    def _emit_failed(
        self, error: NextHopQueryGenerationError, trace: TraceContext | None, started: float
    ) -> None:
        if self._sink is not None and trace is not None:
            self._sink.emit(
                InstrumentationEvent(
                    context=trace,
                    event_type="epsa.next_hop_query.failed",
                    source="epsa.next_hop_query",
                    source_version="research_v1_reconstructed",
                    payload={
                        "latency_ms": round((perf_counter() - started) * 1000, 6),
                        "error_type": type(error).__name__,
                    },
                )
            )


def _relation_for_path(analysis: QuestionAnalysis, path: EvidencePath) -> str | None:
    """Choose a stable question relation, preferring one absent from the partial path."""

    if not analysis.required_relation_hints:
        return None
    observed = {_normalise(relation) for relation in path.relation_chain}
    for hint in analysis.required_relation_hints:
        if _normalise(hint.relation) not in observed:
            return hint.relation
    return analysis.required_relation_hints[0].relation


def _first_usable_entity(entities: tuple[EntityMention, ...]) -> EntityMention | None:
    return next((entity for entity in entities if _usable_text(entity.text)), None)


def _usable_text(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _normalise(value: str) -> str:
    return " ".join(value.split()).casefold()
