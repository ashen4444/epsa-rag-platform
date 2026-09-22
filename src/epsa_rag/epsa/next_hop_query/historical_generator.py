"""Recovered historical Component 09 policy adapted to current immutable contracts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from time import perf_counter

from pydantic import ValidationError

from epsa_rag.core.exceptions import NextHopQueryGenerationError
from epsa_rag.epsa.evidence_graph.models import EvidenceGraph
from epsa_rag.epsa.evidence_path_search.models import EvidencePath
from epsa_rag.epsa.next_hop_query.config import HistoricalAdaptedNextHopQueryGeneratorConfig
from epsa_rag.epsa.next_hop_query.generator import RuleBasedNextHopQueryGeneratorReconstructedV1
from epsa_rag.epsa.next_hop_query.models import (
    NextHopQuery,
    NextHopQueryMetadata,
    NextHopQuerySource,
    NextHopQueryType,
    QueryReasonCode,
)
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis, QuestionType
from epsa_rag.epsa.sufficiency_decision.models import SufficiencyDecision
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext


class RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1:
    """Implement recovered ``next_query_generator.py`` rules without retrieval execution.

    The original policy used prior mutable schemas. This adapter preserves its query
    construction, branch order, and confidence formula while recording only current
    Component 01/05/07 provenance.
    """

    def __init__(
        self,
        *,
        config: HistoricalAdaptedNextHopQueryGeneratorConfig | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> None:
        self._config = config or HistoricalAdaptedNextHopQueryGeneratorConfig()
        self._sink = instrumentation_sink

    @property
    def config(self) -> HistoricalAdaptedNextHopQueryGeneratorConfig:
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
        started = perf_counter()
        try:
            paths = RuleBasedNextHopQueryGeneratorReconstructedV1._validate_inputs(
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
                source=NextHopQuerySource.SUFFICIENCY_DECISION,
                reason="Evidence is already sufficient; no next-hop query is needed.",
            )
        best_path = decision.best_path or _best_fallback_path(paths)
        relations = tuple(hint.relation for hint in analysis.required_relation_hints)
        missing_relation = _choose_missing_relation(
            decision.missing_evidence, relations, best_path.relation_chain if best_path else ()
        )
        expected = analysis.expected_answer_type.value
        if analysis.question_type is QuestionType.COMPARISON:
            targets = tuple(target.text for target in analysis.comparison_targets) or tuple(
                entity.text for entity in analysis.seed_entities[:2]
            )
            return self._special_query(
                analysis,
                decision,
                graph,
                paths,
                best_path,
                entities=targets,
                relation=missing_relation
                or _first_relation_from_text(decision.missing_evidence or ""),
                expected_answer_type=expected,
                include_answer_type=True,
                query_type=NextHopQueryType.COMPARISON_TARGET_COMPLETION,
                reason_code=QueryReasonCode.HISTORICAL_COMPARISON_TARGET_COMPLETION,
                unavailable=(
                    "Comparison evidence is incomplete, but no comparison target or relation "
                    "was available."
                ),
            )
        if analysis.question_type is QuestionType.YES_NO:
            return self._special_query(
                analysis,
                decision,
                graph,
                paths,
                best_path,
                entities=tuple(entity.text for entity in analysis.seed_entities),
                relation=missing_relation,
                expected_answer_type="BOOLEAN",
                include_answer_type=False,
                query_type=NextHopQueryType.YES_NO_RELATION_CHECK,
                reason_code=QueryReasonCode.HISTORICAL_YES_NO_RELATION_CHECK,
                unavailable=(
                    "Yes/no evidence is incomplete, but no entity or relation signal was available."
                ),
            )
        target = _choose_target_entity(
            best_path,
            tuple(entity.text for entity in analysis.seed_entities),
            decision.answer_candidate,
            analysis.question_type,
        )
        query = _build_query(
            entities=(target,)
            if target
            else tuple(entity.text for entity in analysis.seed_entities[:1]),
            relation=missing_relation,
            expected_answer_type=expected,
            config=self._config,
        )
        if query is None:
            return self._no_query(
                analysis,
                decision,
                graph,
                paths,
                source=NextHopQuerySource.QUESTION_ANALYSIS_AND_SUFFICIENCY_DECISION,
                reason=(
                    "No seed, bridge, relation, or expected-answer-type signal was available "
                    "for deterministic query generation."
                ),
            )
        if analysis.question_type is QuestionType.BRIDGE:
            query_type, reason_code = (
                NextHopQueryType.BRIDGE_COMPLETION,
                QueryReasonCode.HISTORICAL_BRIDGE_COMPLETION,
            )
        elif missing_relation:
            query_type, reason_code = (
                NextHopQueryType.RELATION_COMPLETION,
                QueryReasonCode.HISTORICAL_RELATION_COMPLETION,
            )
        elif expected not in {"", "UNKNOWN", "ENTITY"}:
            query_type, reason_code = (
                NextHopQueryType.ANSWER_TYPE_COMPLETION,
                QueryReasonCode.HISTORICAL_ANSWER_TYPE_COMPLETION,
            )
        else:
            query_type, reason_code = (
                NextHopQueryType.FACTOID_COMPLETION,
                QueryReasonCode.HISTORICAL_FACTOID_COMPLETION,
            )
        return self._result(
            analysis,
            decision,
            graph,
            paths,
            best_path,
            query=query,
            query_type=query_type,
            target=target,
            relation=missing_relation,
            expected_answer_type=expected,
            reason_code=reason_code,
            confidence=_confidence(
                bool(target or analysis.seed_entities),
                bool(missing_relation),
                expected not in {"", "UNKNOWN"},
                best_path is not None,
            ),
        )

    def _special_query(
        self,
        analysis: QuestionAnalysis,
        decision: SufficiencyDecision,
        graph: EvidenceGraph,
        paths: tuple[EvidencePath, ...],
        best_path: EvidencePath | None,
        *,
        entities: tuple[str, ...],
        relation: str | None,
        expected_answer_type: str,
        include_answer_type: bool,
        query_type: NextHopQueryType,
        reason_code: QueryReasonCode,
        unavailable: str,
    ) -> NextHopQuery:
        query = _build_query(
            entities=entities,
            relation=relation,
            expected_answer_type=expected_answer_type,
            include_answer_type=include_answer_type,
            config=self._config,
        )
        if query is None:
            return self._no_query(
                analysis,
                decision,
                graph,
                paths,
                source=NextHopQuerySource.QUESTION_ANALYSIS_AND_SUFFICIENCY_DECISION,
                reason=unavailable,
            )
        target = " | ".join(entities) if entities else None
        return self._result(
            analysis,
            decision,
            graph,
            paths,
            best_path,
            query=query,
            query_type=query_type,
            target=target,
            relation=relation,
            expected_answer_type=expected_answer_type,
            reason_code=reason_code,
            confidence=_confidence(
                bool(entities),
                bool(relation),
                expected_answer_type not in {"", "UNKNOWN"},
                decision.best_path is not None,
            ),
        )

    def _result(
        self,
        analysis: QuestionAnalysis,
        decision: SufficiencyDecision,
        graph: EvidenceGraph,
        paths: tuple[EvidencePath, ...],
        best_path: EvidencePath | None,
        *,
        query: str,
        query_type: NextHopQueryType,
        target: str | None,
        relation: str | None,
        expected_answer_type: str,
        reason_code: QueryReasonCode,
        confidence: float,
    ) -> NextHopQuery:
        return NextHopQuery(
            query=query,
            query_type=query_type,
            source=NextHopQuerySource.QUESTION_ANALYSIS_AND_SUFFICIENCY_DECISION,
            target_entity=target,
            missing_relation=relation,
            expected_answer_type=analysis.expected_answer_type,
            reason=_reason_text(decision),
            confidence=confidence,
            metadata=self._metadata(analysis, decision, graph, paths, best_path, reason_code),
        )

    def _no_query(
        self,
        analysis: QuestionAnalysis,
        decision: SufficiencyDecision,
        graph: EvidenceGraph,
        paths: tuple[EvidencePath, ...],
        *,
        source: NextHopQuerySource,
        reason: str,
    ) -> NextHopQuery:
        return NextHopQuery(
            query=None,
            query_type=NextHopQueryType.NO_QUERY,
            source=source,
            reason=reason,
            confidence=0.0,
            metadata=self._metadata(
                analysis, decision, graph, paths, None, QueryReasonCode.HISTORICAL_NO_QUERY
            ),
        )

    def _metadata(
        self,
        analysis: QuestionAnalysis,
        decision: SufficiencyDecision,
        graph: EvidenceGraph,
        paths: tuple[EvidencePath, ...],
        best_path: EvidencePath | None,
        reason_code: QueryReasonCode,
    ) -> NextHopQueryMetadata:
        return NextHopQueryMetadata(
            schema_version=self._config.schema_version,
            generator="RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1",
            version=self._config.mode,
            historical_rules_status="recovered_historical_policy_adapted_to_current_contracts",
            configuration_fingerprint=self._config.fingerprint(),
            source_question_analysis=analysis.metadata,
            source_sufficiency_decision=decision.metadata,
            source_graph=graph.metadata,
            candidate_path_ids=tuple(path.path_id for path in paths),
            selected_path_id=best_path.path_id if best_path is not None else None,
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
                    source_version=self._config.mode,
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
                    source_version=self._config.mode,
                    payload={
                        "latency_ms": round((perf_counter() - started) * 1000, 6),
                        "error_type": type(error).__name__,
                    },
                )
            )


def _norm(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _first_relation_from_text(text: str) -> str | None:
    normalized = _norm(text)
    match = re.search(r"required relation ([a-z0-9_ -]+?)(?:\.|$)", normalized)
    if match:
        return match.group(1).strip()
    for relation in HistoricalAdaptedNextHopQueryGeneratorConfig().relation_query_terms:
        if re.search(rf"\b{re.escape(_norm(relation))}\b", normalized):
            return relation
    return None


def _choose_missing_relation(
    missing_evidence: str | None, required: tuple[str, ...], path_relations: tuple[str, ...]
) -> str | None:
    from_missing = _first_relation_from_text(missing_evidence or "")
    if from_missing:
        return from_missing
    observed = {_norm(relation) for relation in path_relations}
    for relation in required:
        normalized = _norm(relation)
        if not any(
            normalized == value or normalized in value or value in normalized for value in observed
        ):
            return relation
    return required[-1] if required else None


def _best_fallback_path(paths: Sequence[EvidencePath]) -> EvidencePath | None:
    return min(
        paths,
        key=lambda path: (
            -path.score,
            len(path.evidence_unit_ids),
            path.answer_candidate or "",
            path.path_id,
        ),
        default=None,
    )


def _choose_target_entity(
    path: EvidencePath | None,
    seeds: tuple[str, ...],
    answer_candidate: str | None,
    question_type: QuestionType,
) -> str | None:
    if path is None:
        return seeds[0] if seeds else None
    bridge = path.metadata.bridge_entity
    seed_norms = {_norm(seed) for seed in seeds}
    answer_norm = _norm(answer_candidate or path.answer_candidate or "")
    if bridge and _norm(bridge) not in seed_norms:
        return bridge
    entities = path.entity_chain
    if question_type is QuestionType.BRIDGE:
        for entity in reversed(entities):
            if _norm(entity) not in seed_norms and _norm(entity) != answer_norm:
                return entity
    for entity in reversed(entities):
        if _norm(entity) != answer_norm:
            return entity
    return seeds[0] if seeds else None


def _build_query(
    *,
    entities: tuple[str, ...],
    relation: str | None,
    expected_answer_type: str,
    config: HistoricalAdaptedNextHopQueryGeneratorConfig,
    include_answer_type: bool = True,
) -> str | None:
    parts: list[str] = []
    for value in (
        *entities,
        *(config.relation_query_terms.get(_norm(relation or ""), relation or "").split()),
    ):
        if value and _norm(value) not in {_norm(part) for part in parts}:
            parts.append(value.strip())
    answer = config.answer_type_keywords.get(expected_answer_type, "")
    relation_terms = {_norm(part) for part in parts[len(entities) :]}
    covered = {
        "LOCATION": {"birthplace", "capital", "located", "location"},
        "DATE": {"date"},
        "NUMBER": {"number"},
        "PERSON": {"person"},
    }
    if include_answer_type and answer and answer != "evidence" and answer not in parts:
        if not relation_terms.intersection(covered.get(expected_answer_type, set())):
            parts.append(answer)
    query = " ".join(parts).strip()
    return query or None


def _confidence(
    has_entity: bool, has_relation: bool, has_answer_type: bool, has_path: bool
) -> float:
    return round(
        min(
            0.20
            + 0.30 * has_entity
            + 0.25 * has_relation
            + 0.10 * has_answer_type
            + 0.10 * has_path,
            0.90,
        ),
        6,
    )


def _reason_text(decision: SufficiencyDecision) -> str:
    if decision.missing_evidence:
        return f"Generated from missing evidence: {decision.missing_evidence}"
    return f"Generated from insufficient decision: {decision.decision_reason.value}"
