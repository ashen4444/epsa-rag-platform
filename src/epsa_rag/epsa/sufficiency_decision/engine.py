"""Deterministic, inference-safe Component 07 research-v1 decision engine."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from time import perf_counter

from pydantic import JsonValue, ValidationError

from epsa_rag.core.exceptions import SufficiencyDecisionError
from epsa_rag.epsa.evidence_graph.models import EvidenceGraph, GraphEdgeType, GraphNodeType
from epsa_rag.epsa.evidence_path_search.models import EvidencePath
from epsa_rag.epsa.question_analysis.models import AnswerType, QuestionAnalysis, QuestionType
from epsa_rag.epsa.sufficiency_decision.config import SufficiencyDecisionV1Config
from epsa_rag.epsa.sufficiency_decision.models import (
    DecisionReasonCode,
    DecisionTraceEntry,
    GuardCode,
    SufficiencyDecision,
    SufficiencyDecisionMetadata,
)
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext


class RuleBasedSufficiencyEngineV1:
    """Apply frozen research-v1 sufficiency guards without generating an answer or query."""

    def __init__(
        self,
        *,
        config: SufficiencyDecisionV1Config | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> None:
        self._config = config or SufficiencyDecisionV1Config()
        self._sink = instrumentation_sink

    @property
    def config(self) -> SufficiencyDecisionV1Config:
        return self._config

    def decide(
        self,
        question_analysis: QuestionAnalysis,
        evidence_graph: EvidenceGraph,
        evidence_paths: Sequence[EvidencePath],
        *,
        trace_context: TraceContext | None = None,
    ) -> SufficiencyDecision:
        """Return a typed, deterministic sufficiency decision from candidate paths."""

        started = perf_counter()
        try:
            self._validate_inputs(question_analysis, evidence_graph, evidence_paths)
            ranked = tuple(sorted(evidence_paths, key=_path_sort_key))
            if not ranked:
                decision = self._no_paths(question_analysis, evidence_graph)
            elif question_analysis.question_type is QuestionType.BRIDGE:
                decision = self._bridge(question_analysis, evidence_graph, ranked)
            elif question_analysis.question_type is QuestionType.COMPARISON:
                decision = self._comparison(question_analysis, evidence_graph, ranked)
            elif question_analysis.question_type is QuestionType.YES_NO:
                decision = self._yes_no(question_analysis, evidence_graph, ranked)
            else:
                decision = self._factoid(question_analysis, evidence_graph, ranked)
        except SufficiencyDecisionError as error:
            self._emit_failed(error, trace_context, started)
            raise
        except (ValidationError, ValueError, TypeError) as error:
            wrapped = SufficiencyDecisionError(str(error))
            self._emit_failed(wrapped, trace_context, started)
            raise wrapped from error
        except Exception as error:
            wrapped = SufficiencyDecisionError("sufficiency decision failed")
            self._emit_failed(wrapped, trace_context, started)
            raise wrapped from error
        self._emit_completed(decision, trace_context, started)
        return decision

    @staticmethod
    def _validate_inputs(analysis: object, graph: object, paths: object) -> None:
        if not isinstance(analysis, QuestionAnalysis):
            raise SufficiencyDecisionError("question analysis must use Component 01 contract")
        if not isinstance(graph, EvidenceGraph):
            raise SufficiencyDecisionError("evidence graph must use Component 05 contract")
        if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
            raise SufficiencyDecisionError("evidence paths must be a sequence")
        if graph.question_type is not analysis.question_type:
            raise SufficiencyDecisionError(
                "evidence graph question type must match Component 01 input"
            )
        if graph.metadata.expected_answer_type is not analysis.expected_answer_type:
            raise SufficiencyDecisionError(
                "evidence graph answer type must match Component 01 input"
            )
        expected_relations = _dedupe(hint.relation for hint in analysis.required_relation_hints)
        if graph.metadata.required_relation_hints != expected_relations:
            raise SufficiencyDecisionError(
                "evidence graph relation hints must match Component 01 input"
            )
        for path in paths:
            if not isinstance(path, EvidencePath):
                raise SufficiencyDecisionError("evidence paths must use Component 06 contract")
            if path.question_type is not analysis.question_type:
                raise SufficiencyDecisionError(
                    "candidate path question type must match Component 01 input"
                )
            if path.metadata.source_graph != graph.metadata:
                raise SufficiencyDecisionError(
                    "candidate path graph provenance must match Component 05 input"
                )

    def _bridge(
        self, analysis: QuestionAnalysis, graph: EvidenceGraph, paths: tuple[EvidencePath, ...]
    ) -> SufficiencyDecision:
        latest: tuple[list[DecisionTraceEntry], str] = (
            [],
            "No complete bridge evidence path found.",
        )
        for path in paths:
            trace = _path_trace(path)
            selected = _dedupe(path.evidence_unit_ids)
            if not _seed_connected(path, graph, analysis):
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.SEED_CONNECTION,
                            "Path does not connect a question seed.",
                        ),
                    ],
                    "Bridge path does not connect to a question seed entity.",
                )
                continue
            trace.append(_pass(GuardCode.SEED_CONNECTION, "Path connects a question seed."))
            if len(selected) < 2:
                bridge = _bridge_entity(path, analysis)
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.EVIDENCE_UNITS,
                            "Bridge path needs two evidence units.",
                            {"count": len(selected)},
                            {"minimum": 2},
                        ),
                    ],
                    f"Bridge path is incomplete after bridge entity {bridge}."
                    if bridge
                    else "Bridge path has fewer than two useful evidence units.",
                )
                continue
            trace.append(
                _pass(
                    GuardCode.EVIDENCE_UNITS,
                    "Bridge path has multiple evidence units.",
                    {"count": len(selected)},
                    {"minimum": 2},
                )
            )
            bridge = _bridge_entity(path, analysis)
            if bridge is None:
                latest = (
                    [*trace, _fail(GuardCode.BRIDGE_ENTITY, "No non-seed bridge entity exists.")],
                    "No non-seed bridge entity found in the candidate path.",
                )
                continue
            trace.append(
                _pass(
                    GuardCode.BRIDGE_ENTITY,
                    "Non-seed bridge entity found.",
                    {"bridge_entity": bridge},
                )
            )
            if not _specific_bridge(bridge):
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.BRIDGE_SPECIFICITY,
                            "Bridge entity is too generic.",
                            {"bridge_entity": bridge},
                        ),
                    ],
                    f"Bridge entity {bridge} is too generic to support a complete bridge path.",
                )
                continue
            trace.append(
                _pass(
                    GuardCode.BRIDGE_SPECIFICITY,
                    "Bridge entity is specific.",
                    {"bridge_entity": bridge},
                )
            )
            if not _bridge_grounded_answer_side(path, graph, bridge):
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.BRIDGE_ANSWER_SIDE_GROUNDING,
                            "Bridge is absent from answer-side evidence.",
                        ),
                    ],
                    f"Bridge entity {bridge} is not strongly grounded in the answer-side evidence.",
                )
                continue
            trace.append(
                _pass(
                    GuardCode.BRIDGE_ANSWER_SIDE_GROUNDING,
                    "Bridge is grounded in answer-side evidence.",
                )
            )
            failure = _answer_failure(path, graph, analysis.expected_answer_type)
            if failure is not None:
                latest = (
                    [*trace, failure],
                    "Answer candidate does not satisfy the expected answer type "
                    f"{analysis.expected_answer_type.value}.",
                )
                continue
            trace.extend(_answer_success_trace())
            matched, required = _relation_coverage(path, analysis)
            if required and matched < min(len(required), 2):
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.RELATION_COVERAGE,
                            "Bridge relation coverage is incomplete.",
                            {"matched": matched},
                            {"required": min(len(required), 2)},
                        ),
                    ],
                    "Required relation evidence is incomplete.",
                )
                continue
            trace.append(
                _pass(
                    GuardCode.RELATION_COVERAGE,
                    "Required bridge relations are covered.",
                    {"matched": matched},
                    {"required": min(len(required), 2)},
                )
            )
            anchors = _strong_anchor_coverage(path, analysis)
            if not anchors[0]:
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.QUESTION_ANCHOR_COVERAGE,
                            "Selected evidence omits every strong question anchor.",
                            anchors[1],
                        ),
                    ],
                    "Selected evidence does not mention any strong question anchor.",
                )
                continue
            trace.append(
                _pass(
                    GuardCode.QUESTION_ANCHOR_COVERAGE,
                    "Strong question anchor coverage is satisfied.",
                    anchors[1],
                )
            )
            role = _role_coverage(path, graph, analysis, bridge)
            if not role[0]:
                latest = (
                    [*trace, _fail(GuardCode.ROLE_COVERAGE, "Bridge path roles are not grounded.")],
                    "Bridge path lacks role-relevant seed, bridge, or answer evidence.",
                )
                continue
            trace.append(_pass(GuardCode.ROLE_COVERAGE, "Bridge path roles are grounded."))
            coverage = _coverage(path, graph)
            if not coverage[0]:
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.INDEPENDENT_EVIDENCE_COVERAGE,
                            "Bridge evidence lacks independent coverage.",
                            coverage[1],
                            {"units": 2, "sources": 2},
                        ),
                    ],
                    "Selected evidence does not meet minimum independent bridge coverage.",
                )
                continue
            trace.append(
                _pass(
                    GuardCode.INDEPENDENT_EVIDENCE_COVERAGE,
                    "Bridge evidence has independent coverage.",
                    coverage[1],
                )
            )
            return self._result(
                True,
                analysis,
                graph,
                paths,
                path,
                selected,
                None,
                DecisionReasonCode.SUFFICIENT_BRIDGE,
                tuple(trace),
                self._confidence(path, self._config.bridge_confidence_base),
                False,
            )
        return self._result(
            False,
            analysis,
            graph,
            paths,
            paths[0],
            _dedupe(paths[0].evidence_unit_ids),
            latest[1],
            DecisionReasonCode.BRIDGE_RULES_UNSATISFIED,
            tuple(latest[0]),
            self._confidence(paths[0], self._config.insufficient_confidence_base),
            False,
        )

    def _factoid(
        self, analysis: QuestionAnalysis, graph: EvidenceGraph, paths: tuple[EvidencePath, ...]
    ) -> SufficiencyDecision:
        latest: tuple[list[DecisionTraceEntry], str] = (
            [],
            "No complete factoid evidence path found.",
        )
        for path in paths:
            trace = _path_trace(path)
            selected = _dedupe(path.evidence_unit_ids)
            if not selected:
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.EVIDENCE_UNITS,
                            "Factoid path has no evidence units.",
                        ),
                    ],
                    "No evidence unit supports the answer candidate.",
                )
                continue
            trace.append(_pass(GuardCode.EVIDENCE_UNITS, "Factoid path has evidence."))
            if not _seed_connected(path, graph, analysis):
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.SEED_CONNECTION,
                            "Path does not connect a question seed.",
                        ),
                    ],
                    "Candidate path does not connect a seed entity to an answer candidate.",
                )
                continue
            trace.append(_pass(GuardCode.SEED_CONNECTION, "Path connects a question seed."))
            failure = _answer_failure(path, graph, analysis.expected_answer_type)
            if failure is not None:
                latest = (
                    [*trace, failure],
                    "No typed, specific answer candidate found in the candidate path.",
                )
                continue
            trace.extend(_answer_success_trace())
            if not _role_coverage(path, graph, analysis, None)[0]:
                latest = (
                    [*trace, _fail(GuardCode.ROLE_COVERAGE, "Factoid roles are not grounded.")],
                    "Factoid path lacks role-relevant seed and answer evidence.",
                )
                continue
            trace.append(_pass(GuardCode.ROLE_COVERAGE, "Factoid roles are grounded."))
            matched, required = _relation_coverage(path, analysis)
            if required and matched < len(required):
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.RELATION_COVERAGE,
                            "Factoid relation coverage is incomplete.",
                            {"matched": matched},
                            {"required": len(required)},
                        ),
                    ],
                    "Required relation evidence is incomplete.",
                )
                continue
            trace.append(
                _pass(GuardCode.RELATION_COVERAGE, "Required factoid relations are covered.")
            )
            anchors = _strong_anchor_coverage(path, analysis)
            if not anchors[0]:
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.QUESTION_ANCHOR_COVERAGE,
                            "Selected evidence omits every strong question anchor.",
                            anchors[1],
                        ),
                    ],
                    "Selected evidence does not mention any strong question anchor.",
                )
                continue
            trace.append(
                _pass(
                    GuardCode.QUESTION_ANCHOR_COVERAGE,
                    "Strong question anchor coverage is satisfied.",
                    anchors[1],
                )
            )
            multi = _multi_fact_question(analysis)
            generic = analysis.expected_answer_type in {AnswerType.ENTITY, AnswerType.UNKNOWN}
            if generic and not required and len(selected) < 2:
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.GENERIC_FACTOID_COVERAGE,
                            "Generic factoid needs more than one evidence unit.",
                        ),
                    ],
                    "Generic factoid answer type has only one evidence unit and no explicit "
                    "relation evidence.",
                )
                continue
            if multi:
                coverage = _coverage(path, graph)
                if not coverage[0]:
                    latest = (
                        [
                            *trace,
                            _fail(
                                GuardCode.MULTI_FACT_COVERAGE,
                                "Multi-fact question lacks independent coverage.",
                                coverage[1],
                            ),
                        ],
                        "Question appears to require multiple facts, but evidence coverage is "
                        "incomplete.",
                    )
                    continue
                trace.append(
                    _pass(
                        GuardCode.MULTI_FACT_COVERAGE,
                        "Multi-fact evidence has independent coverage.",
                        coverage[1],
                    )
                )
            return self._result(
                True,
                analysis,
                graph,
                paths,
                path,
                selected,
                None,
                DecisionReasonCode.SUFFICIENT_FACTOID,
                tuple(trace),
                self._confidence(path, self._config.factoid_confidence_base),
                False,
            )
        return self._result(
            False,
            analysis,
            graph,
            paths,
            paths[0],
            _dedupe(paths[0].evidence_unit_ids),
            latest[1],
            DecisionReasonCode.FACTOID_RULES_UNSATISFIED,
            tuple(latest[0]),
            self._confidence(paths[0], self._config.insufficient_confidence_base),
            False,
        )

    def _comparison(
        self, analysis: QuestionAnalysis, graph: EvidenceGraph, paths: tuple[EvidencePath, ...]
    ) -> SufficiencyDecision:
        selected = _dedupe(evidence_id for path in paths for evidence_id in path.evidence_unit_ids)
        trace = (
            _fail(
                GuardCode.COMPARISON_RESOLUTION,
                "research_v1 does not resolve comparison values.",
                {"paths": len(paths)},
            ),
        )
        return self._result(
            False,
            analysis,
            graph,
            paths,
            paths[0],
            selected,
            "Comparison target evidence is incomplete or requires later specialized "
            "comparison resolution.",
            DecisionReasonCode.COMPARISON_UNSUPPORTED_RESEARCH_V1,
            trace,
            self._confidence(paths[0], self._config.insufficient_confidence_base),
            False,
        )

    def _yes_no(
        self, analysis: QuestionAnalysis, graph: EvidenceGraph, paths: tuple[EvidencePath, ...]
    ) -> SufficiencyDecision:
        latest: tuple[list[DecisionTraceEntry], str] = (
            [],
            "No connected yes/no evidence path found.",
        )
        for path in paths:
            trace = _path_trace(path)
            selected = _dedupe(path.evidence_unit_ids)
            if not selected:
                latest = (
                    [*trace, _fail(GuardCode.EVIDENCE_UNITS, "Yes/no path has no evidence units.")],
                    "No evidence unit supports the yes/no claim.",
                )
                continue
            if not _seed_connected(path, graph, analysis):
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.SEED_CONNECTION,
                            "Path does not connect question entities.",
                        ),
                    ],
                    "Evidence path does not connect the relevant question entities.",
                )
                continue
            matched, required = _relation_coverage(path, analysis)
            if required and matched < len(required):
                latest = (
                    [
                        *trace,
                        _fail(
                            GuardCode.RELATION_COVERAGE,
                            "Yes/no relation coverage is incomplete.",
                        ),
                    ],
                    "Required yes/no relation evidence is incomplete.",
                )
                continue
            trace.extend(
                (
                    _pass(GuardCode.EVIDENCE_UNITS, "Yes/no path has evidence."),
                    _pass(GuardCode.SEED_CONNECTION, "Question entities are connected."),
                    _pass(GuardCode.RELATION_COVERAGE, "Required yes/no relations are covered."),
                    _fail(
                        GuardCode.YES_NO_POLARITY,
                        "Final polarity is intentionally outside Component 07.",
                    ),
                )
            )
            return self._result(
                True,
                analysis,
                graph,
                paths,
                path,
                selected,
                None,
                DecisionReasonCode.SUFFICIENT_YES_NO_EVIDENCE,
                tuple(trace),
                self._confidence(path, self._config.yes_no_confidence_base),
                True,
            )
        return self._result(
            False,
            analysis,
            graph,
            paths,
            paths[0],
            _dedupe(paths[0].evidence_unit_ids),
            latest[1],
            DecisionReasonCode.YES_NO_RULES_UNSATISFIED,
            tuple(latest[0]),
            self._confidence(paths[0], self._config.insufficient_confidence_base),
            True,
        )

    def _no_paths(self, analysis: QuestionAnalysis, graph: EvidenceGraph) -> SufficiencyDecision:
        return self._result(
            False,
            analysis,
            graph,
            (),
            None,
            (),
            "No candidate evidence path found.",
            DecisionReasonCode.NO_CANDIDATE_PATHS,
            (_fail(GuardCode.PATH_AVAILABLE, "No candidate evidence path was available."),),
            0.0,
            analysis.question_type is QuestionType.YES_NO,
        )

    def _result(
        self,
        sufficient: bool,
        analysis: QuestionAnalysis,
        graph: EvidenceGraph,
        paths: tuple[EvidencePath, ...],
        best: EvidencePath | None,
        selected: tuple[str, ...],
        missing: str | None,
        reason: DecisionReasonCode,
        trace: tuple[DecisionTraceEntry, ...],
        confidence: float,
        no_polarity: bool,
    ) -> SufficiencyDecision:
        selected_chunks = _selected_chunk_ids(graph, selected)
        return SufficiencyDecision(
            sufficient=sufficient,
            confidence=confidence,
            question_type=analysis.question_type,
            best_path=best,
            candidate_paths_considered=paths,
            selected_evidence_unit_ids=selected,
            selected_chunk_ids=selected_chunks,
            answer_candidate=None
            if analysis.question_type is QuestionType.YES_NO or best is None
            else best.answer_candidate,
            answer_type=AnswerType.BOOLEAN
            if analysis.question_type is QuestionType.YES_NO
            else (best.answer_type if best is not None else analysis.expected_answer_type),
            missing_evidence=missing,
            decision_reason=reason,
            rule_trace=trace,
            metadata=SufficiencyDecisionMetadata(
                configuration_fingerprint=self._config.fingerprint(),
                source_graph=graph.metadata,
                candidate_path_ids=tuple(path.path_id for path in paths),
                does_not_generate_yes_no_polarity=no_polarity,
            ),
        )

    def _confidence(self, path: EvidencePath, base: float) -> float:
        return round(
            min(
                1.0,
                base
                + min(path.score, self._config.path_score_contribution_cap)
                * self._config.path_score_contribution_weight,
            ),
            6,
        )

    def _emit_completed(
        self, decision: SufficiencyDecision, context: TraceContext | None, started: float
    ) -> None:
        if self._sink is not None and context is not None:
            self._sink.emit(
                InstrumentationEvent(
                    context=context,
                    event_type="epsa.sufficiency_decision.completed",
                    source="epsa.sufficiency_decision",
                    source_version="research_v1",
                    payload={
                        "latency_ms": round((perf_counter() - started) * 1000, 6),
                        "sufficient": decision.sufficient,
                        "reason_code": decision.decision_reason.value,
                        "candidate_paths": len(decision.candidate_paths_considered),
                        "graph_version": decision.metadata.source_graph.version,
                        "confidence_kind": "uncalibrated_heuristic",
                    },
                )
            )

    def _emit_failed(
        self, error: SufficiencyDecisionError, context: TraceContext | None, started: float
    ) -> None:
        if self._sink is not None and context is not None:
            self._sink.emit(
                InstrumentationEvent(
                    context=context,
                    event_type="epsa.sufficiency_decision.failed",
                    source="epsa.sufficiency_decision",
                    source_version="research_v1",
                    payload={
                        "latency_ms": round((perf_counter() - started) * 1000, 6),
                        "error_type": type(error).__name__,
                    },
                )
            )


def _path_sort_key(path: EvidencePath) -> tuple[float, int, str, str]:
    return (-path.score, len(path.evidence_unit_ids), path.answer_candidate or "", path.path_id)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _path_trace(path: EvidencePath) -> list[DecisionTraceEntry]:
    return [
        _pass(
            GuardCode.PATH_AVAILABLE,
            "Candidate path is being evaluated.",
            {"path_id": path.path_id, "path_score": path.score},
        )
    ]


def _pass(
    code: GuardCode,
    message: str,
    observed: Mapping[str, JsonValue] | None = None,
    required: Mapping[str, JsonValue] | None = None,
) -> DecisionTraceEntry:
    return DecisionTraceEntry(
        rule_code=code,
        passed=True,
        message=message,
        observed=dict(observed or {}),
        required=dict(required or {}),
    )


def _fail(
    code: GuardCode,
    message: str,
    observed: Mapping[str, JsonValue] | None = None,
    required: Mapping[str, JsonValue] | None = None,
) -> DecisionTraceEntry:
    return DecisionTraceEntry(
        rule_code=code,
        passed=False,
        message=message,
        observed=dict(observed or {}),
        required=dict(required or {}),
    )


def _norm(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def _seed_connected(path: EvidencePath, graph: EvidenceGraph, analysis: QuestionAnalysis) -> bool:
    if not analysis.seed_entities:
        return True
    if set(path.node_ids).intersection(graph.seed_entity_node_ids):
        return True
    seeds = {_norm(seed.text) for seed in analysis.seed_entities}
    return bool(seeds.intersection(_norm(entity) for entity in path.entity_chain))


def _bridge_entity(path: EvidencePath, analysis: QuestionAnalysis) -> str | None:
    if path.metadata.bridge_entity:
        return path.metadata.bridge_entity
    excluded = {_norm(seed.text) for seed in analysis.seed_entities}
    if path.answer_candidate:
        excluded.add(_norm(path.answer_candidate))
    return next((entity for entity in path.entity_chain if _norm(entity) not in excluded), None)


def _specific_bridge(value: str) -> bool:
    generic = {
        "a",
        "an",
        "the",
        "company",
        "hotel",
        "family",
        "group",
        "business",
        "corporation",
        "organization",
        "organisation",
        "person",
        "city",
        "country",
        "state",
        "location",
        "place",
        "head office",
        "office",
        "headquarters",
        "member",
        "song",
        "album",
        "film",
        "movie",
        "series",
        "character",
        "school",
        "team",
        "band",
    }
    blocked = {
        "african",
        "american",
        "arab",
        "argentine",
        "asian",
        "australian",
        "austrian",
        "belgian",
        "brazilian",
        "british",
        "canadian",
        "chinese",
        "danish",
        "dutch",
        "english",
        "european",
        "finnish",
        "french",
        "german",
        "greek",
        "indian",
        "irish",
        "italian",
        "japanese",
        "korean",
        "mexican",
        "norwegian",
        "polish",
        "russian",
        "scottish",
        "spanish",
        "swedish",
        "swiss",
        "turkish",
        "welsh",
        "us",
        "uk",
    }
    cleaned = value.strip(" ,.;:()[]{}")
    normalized = _norm(cleaned)
    if not cleaned or normalized in generic | blocked | _MONTHS:
        return False
    return (
        (cleaned.isupper() and len(cleaned) >= 2)
        or len(cleaned.split()) >= 2
        or (cleaned[:1].isupper() and len(cleaned) >= 3)
    )


def _bridge_grounded_answer_side(path: EvidencePath, graph: EvidenceGraph, bridge: str) -> bool:
    if not path.scored_evidence_units:
        return False
    answer_evidence = path.scored_evidence_units[-1].evidence_unit
    bridge_norm = _norm(bridge)
    if bridge_norm in _norm(answer_evidence.resolved_text) or bridge_norm in {
        _norm(value) for value in answer_evidence.entities
    }:
        return True
    sentence_ids = {
        node.node_id
        for node in graph.nodes
        if node.node_type is GraphNodeType.SENTENCE
        and node.scored_evidence
        and node.scored_evidence.evidence_unit.evidence_unit_id == answer_evidence.evidence_unit_id
    }
    return any(
        edge.edge_type is GraphEdgeType.SENTENCE_MENTIONS_ENTITY
        and edge.source_id in sentence_ids
        and _norm(graph.node_by_id(edge.target_id).label) == bridge_norm
        for edge in graph.edges
    )


def _answer_failure(
    path: EvidencePath, graph: EvidenceGraph, expected: AnswerType
) -> DecisionTraceEntry | None:
    if not _specific_answer(path.answer_candidate):
        return _fail(
            GuardCode.ANSWER_CANDIDATE, "Answer candidate is missing, generic, or incomplete."
        )
    if not _answer_surface_matches(path.answer_candidate or "", expected):
        return _fail(
            GuardCode.ANSWER_SURFACE_TYPE,
            "Answer candidate surface is incompatible with expected type.",
            {"candidate": path.answer_candidate, "expected": expected.value},
        )
    if not _graph_path_answer_type_matches(path, graph, expected):
        return _fail(
            GuardCode.GRAPH_PATH_ANSWER_TYPE,
            "Path and graph do not support expected answer type.",
            {"expected": expected.value},
        )
    return None


def _answer_success_trace() -> tuple[DecisionTraceEntry, ...]:
    return (
        _pass(GuardCode.ANSWER_CANDIDATE, "Specific answer candidate found."),
        _pass(GuardCode.ANSWER_SURFACE_TYPE, "Answer candidate surface is compatible."),
        _pass(GuardCode.GRAPH_PATH_ANSWER_TYPE, "Path and graph support the expected answer type."),
    )


def _specific_answer(value: str | None) -> bool:
    if value is None:
        return False
    normalized = _norm(value.strip(" ,.;:()[]{}"))
    if (
        not normalized
        or normalized
        in {
            "person",
            "location",
            "date",
            "number",
            "boolean",
            "entity",
            "organization",
            "organisation",
            "title or work",
            "unknown",
        }
        | _MONTHS
    ):
        return False
    return not bool(
        re.fullmatch(
            r"(new|old|north|south|east|west|middle|central|united|republic|kingdom|states)",
            normalized,
        )
    ) and normalized.split()[-1] not in {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "after",
        "before",
        "from",
        "to",
        "for",
    }


def _answer_surface_matches(candidate: str, expected: AnswerType) -> bool:
    if expected in {AnswerType.UNKNOWN, AnswerType.ENTITY, AnswerType.BOOLEAN}:
        return True
    text = candidate.strip()
    if expected is AnswerType.NUMBER:
        return bool(re.search(r"\d", text))
    if expected is AnswerType.DATE:
        return bool(re.search(r"\b\d{4}\b", text)) or any(month in _norm(text) for month in _MONTHS)
    if expected is AnswerType.LOCATION:
        return bool(re.fullmatch(r"[A-Z][A-Za-z]*(?:[ .'-][A-Z][A-Za-z]*)*", text))
    if expected is AnswerType.PERSON:
        return len(re.findall(r"[A-Z][A-Za-z'-]*", text)) >= 2
    if expected is AnswerType.ORGANIZATION:
        return (
            bool(re.search(r"\b(Inc|Ltd|University|Company|Corporation|Association|Club)\b", text))
            or len(re.findall(r"[A-Z][A-Za-z'-]*", text)) >= 2
        )
    return len(text) >= 2


def _graph_path_answer_type_matches(
    path: EvidencePath, graph: EvidenceGraph, expected: AnswerType
) -> bool:
    if expected in {AnswerType.UNKNOWN, AnswerType.ENTITY}:
        return True
    if path.answer_type not in {expected, AnswerType.UNKNOWN, AnswerType.ENTITY}:
        return False
    evidence_ids = set(path.evidence_unit_ids)
    sentence_ids = {
        node.node_id
        for node in graph.nodes
        if node.node_type is GraphNodeType.SENTENCE
        and node.scored_evidence
        and node.scored_evidence.evidence_unit.evidence_unit_id in evidence_ids
    }
    return any(
        edge.edge_type is GraphEdgeType.SENTENCE_HAS_ANSWER_TYPE
        and edge.source_id in sentence_ids
        and graph.node_by_id(edge.target_id).label == expected.value
        for edge in graph.edges
    )


def _relation_coverage(
    path: EvidencePath, analysis: QuestionAnalysis
) -> tuple[int, tuple[str, ...]]:
    required = _dedupe(hint.relation for hint in analysis.required_relation_hints)
    normalized_path = tuple(_norm(value) for value in path.relation_chain)
    matched = sum(
        any(_norm(relation) in item or item in _norm(relation) for item in normalized_path)
        for relation in required
    )
    return matched, required


def _role_coverage(
    path: EvidencePath, graph: EvidenceGraph, analysis: QuestionAnalysis, bridge: str | None
) -> tuple[bool, dict[str, int]]:
    texts = tuple(unit.evidence_unit.resolved_text for unit in path.scored_evidence_units)
    seeds = tuple(_norm(seed.text) for seed in analysis.seed_entities)
    seed_side = bool(texts and any(seed in _norm(texts[0]) for seed in seeds))
    answer_side = bool(
        texts and path.answer_candidate and _norm(path.answer_candidate) in _norm(texts[-1])
    )
    bridge_side = bridge is None or (
        len(texts) >= 2 and _norm(bridge) in _norm(texts[0]) and _norm(bridge) in _norm(texts[-1])
    )
    return seed_side and answer_side and bridge_side, {
        "seed_side": int(seed_side),
        "answer_side": int(answer_side),
        "bridge_side": int(bridge_side),
    }


def _coverage(path: EvidencePath, graph: EvidenceGraph) -> tuple[bool, dict[str, int]]:
    selected = _dedupe(path.evidence_unit_ids)
    known = {
        node.scored_evidence.evidence_unit.evidence_unit_id
        for node in graph.nodes
        if node.node_type is GraphNodeType.SENTENCE and node.scored_evidence
    }
    units = tuple(unit.evidence_unit for unit in path.scored_evidence_units)
    titles = {unit.doc_title for unit in units if unit.doc_title}
    chunks = {unit.chunk_id for unit in units}
    non_empty = sum(bool(unit.resolved_text.strip()) for unit in units)
    observed = {
        "selected": len(selected),
        "covered": sum(value in known for value in selected),
        "sources": len(titles) if titles else len(chunks),
        "non_empty": non_empty,
    }
    return observed["covered"] == len(selected) and observed["covered"] >= 2 and observed[
        "sources"
    ] >= 2 and observed["non_empty"] >= 2, observed


def _multi_fact_question(analysis: QuestionAnalysis) -> bool:
    text = _norm(analysis.normalized_question)
    nested_markers = (
        " that ",
        " who ",
        " whose ",
        " which ",
        " in which ",
        " of the ",
        " named after ",
        "head office",
        "headquarters",
    )
    bridge_like_markers = (
        "director of",
        "writer of",
        "written by",
        "author of",
        "composer of",
        "producer of",
        "founder of",
        "wife of",
        "husband of",
        "father of",
        "mother of",
        "member of",
        "part of",
    )
    return any(marker.strip() in text for marker in nested_markers) and any(
        marker in text for marker in bridge_like_markers
    )


def _strong_anchor_coverage(
    path: EvidencePath, analysis: QuestionAnalysis
) -> tuple[bool, dict[str, JsonValue]]:
    """Apply the narrow historical guard for quoted or digit-bearing question targets."""

    anchors = [
        (left or right).strip()
        for left, right in re.findall(r'"([^"]+)"|\'([^\']+)\'', analysis.raw_question)
        if left or right
    ]
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.-]*", analysis.raw_question)
    for offset, token in enumerate(tokens):
        if not any(character.isdigit() for character in token):
            continue
        left = max(0, offset - 2)
        right = min(len(tokens), offset + 5)
        candidate = tokens[left:right]
        if len(candidate) >= 2 and any(part[:1].isupper() for part in candidate):
            anchors.append(" ".join(candidate))
    anchors = list(_dedupe(anchors))
    selected_text = " ".join(
        f"{unit.evidence_unit.doc_title} {unit.evidence_unit.resolved_text}"
        for unit in path.scored_evidence_units
    )
    normalized = _anchor_normalize(selected_text)
    matched = tuple(
        anchor for anchor in anchors if f" {_anchor_normalize(anchor)} " in f" {normalized} "
    )
    observed: dict[str, JsonValue] = {
        "required": bool(anchors),
        "anchor_count": len(anchors),
        "matched_count": len(matched),
    }
    return not anchors or bool(matched), observed


def _anchor_normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _selected_chunk_ids(graph: EvidenceGraph, evidence_ids: tuple[str, ...]) -> tuple[str, ...]:
    chunk_by_evidence_id = {
        node.scored_evidence.evidence_unit.evidence_unit_id: (
            node.scored_evidence.evidence_unit.chunk_id
        )
        for node in graph.nodes
        if node.node_type is GraphNodeType.SENTENCE and node.scored_evidence
    }
    return _dedupe(
        chunk_by_evidence_id[evidence_id]
        for evidence_id in evidence_ids
        if evidence_id in chunk_by_evidence_id
    )


_MONTHS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
)
