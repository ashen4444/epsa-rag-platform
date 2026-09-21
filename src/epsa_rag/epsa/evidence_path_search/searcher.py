"""Deterministic, inference-safe EPSA Component 06 research-v1 searcher."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

from pydantic import ValidationError

from epsa_rag.core.exceptions import EvidencePathSearchError
from epsa_rag.epsa.evidence_graph.models import (
    EvidenceGraph,
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
)
from epsa_rag.epsa.evidence_path_search.config import EvidencePathSearcherV1Config
from epsa_rag.epsa.evidence_path_search.models import (
    EvidencePath,
    EvidencePathMetadata,
    PathKind,
    PathScoreBreakdown,
)
from epsa_rag.epsa.evidence_scoring.models import ScoredEvidenceUnit
from epsa_rag.epsa.question_analysis.models import AnswerType, QuestionAnalysis, QuestionType
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext


@dataclass(frozen=True)
class _GraphIndex:
    """Sorted local graph projections used by template searches."""

    nodes: dict[str, GraphNode]
    outgoing_by_type: dict[tuple[str, GraphEdgeType], tuple[GraphEdge, ...]]
    sentence_to_entities: dict[str, tuple[str, ...]]
    entity_to_sentences: dict[str, tuple[str, ...]]
    sentence_to_relations: dict[str, tuple[str, ...]]
    sentence_to_answer_types: dict[str, tuple[AnswerType, ...]]
    seed_to_sentences: dict[str, tuple[GraphEdge, ...]]
    possible_answer_targets: dict[str, frozenset[str]]
    evidence_by_sentence: dict[str, ScoredEvidenceUnit]
    evidence_unit_by_sentence: dict[str, str]
    evidence_score_by_sentence: dict[str, float]
    retrieval_rank_by_sentence: dict[str, int | None]


class EvidencePathSearcherV1:
    """Generate and rank documented ``research_v1`` candidate paths only."""

    def __init__(
        self,
        *,
        config: EvidencePathSearcherV1Config | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> None:
        self._config = config or EvidencePathSearcherV1Config()
        self._sink = instrumentation_sink

    @property
    def config(self) -> EvidencePathSearcherV1Config:
        return self._config

    def search_paths(
        self,
        evidence_graph: EvidenceGraph,
        question_analysis: QuestionAnalysis,
        max_paths: int = 10,
        *,
        trace_context: TraceContext | None = None,
    ) -> list[EvidencePath]:
        """Return ranked candidates without deciding sufficiency or answer correctness."""

        started = perf_counter()
        try:
            self._validate_inputs(evidence_graph, question_analysis, max_paths)
            if max_paths <= 0:
                paths: list[EvidencePath] = []
            else:
                index = self._build_index(evidence_graph)
                paths = self._search_by_question_type(evidence_graph, question_analysis, index)
                paths = self._deduplicate(paths)
                paths.sort(
                    key=lambda path: (
                        -path.score,
                        len(path.evidence_unit_ids),
                        path.answer_candidate or "",
                        path.path_id,
                    )
                )
                paths = paths[:max_paths]
        except EvidencePathSearchError as error:
            self._emit_failed(evidence_graph, error, trace_context, started)
            raise
        except (ValidationError, ValueError, TypeError) as error:
            wrapped = EvidencePathSearchError(str(error))
            self._emit_failed(evidence_graph, wrapped, trace_context, started)
            raise wrapped from error
        except Exception as error:
            wrapped = EvidencePathSearchError("evidence path search failed")
            self._emit_failed(evidence_graph, wrapped, trace_context, started)
            raise wrapped from error
        self._emit_completed(evidence_graph, paths, trace_context, started)
        return paths

    def _validate_inputs(
        self,
        evidence_graph: object,
        question_analysis: object,
        max_paths: object,
    ) -> None:
        if not isinstance(evidence_graph, EvidenceGraph):
            raise EvidencePathSearchError("evidence graph must use Component 05 contract")
        if not isinstance(question_analysis, QuestionAnalysis):
            raise EvidencePathSearchError("question analysis must use Component 01 contract")
        if isinstance(max_paths, bool) or not isinstance(max_paths, int):
            raise EvidencePathSearchError("max_paths must be an integer")
        if evidence_graph.question_type is not question_analysis.question_type:
            raise EvidencePathSearchError(
                "evidence graph question type must match Component 01 input"
            )
        if (
            evidence_graph.metadata.expected_answer_type
            is not question_analysis.expected_answer_type
        ):
            raise EvidencePathSearchError(
                "evidence graph answer type must match Component 01 input"
            )
        graph_hints = tuple(evidence_graph.metadata.required_relation_hints)
        analysis_hints = tuple(hint.relation for hint in question_analysis.required_relation_hints)
        if graph_hints != _dedupe_preserve_order(analysis_hints):
            raise EvidencePathSearchError(
                "evidence graph relation hints must match Component 01 input"
            )

    def _search_by_question_type(
        self,
        graph: EvidenceGraph,
        analysis: QuestionAnalysis,
        index: _GraphIndex,
    ) -> list[EvidencePath]:
        if analysis.question_type is QuestionType.BRIDGE:
            return self._search_bridge_paths(graph, analysis, index)
        if analysis.question_type is QuestionType.COMPARISON:
            return self._search_comparison_partial_paths(graph, analysis, index)
        if analysis.question_type is QuestionType.YES_NO:
            return self._search_yes_no_paths(graph, analysis, index)
        return self._search_factoid_paths(graph, analysis, index)

    def _search_bridge_paths(
        self, graph: EvidenceGraph, analysis: QuestionAnalysis, index: _GraphIndex
    ) -> list[EvidencePath]:
        paths: list[EvidencePath] = []
        seed_ids = graph.seed_entity_node_ids
        excluded_seeds = set(seed_ids)
        for seed_id in seed_ids:
            for seed_edge in index.seed_to_sentences.get(seed_id, ()):
                first_sentence_id = seed_edge.target_id
                for bridge_id in index.sentence_to_entities.get(first_sentence_id, ()):
                    if bridge_id in excluded_seeds:
                        continue
                    for second_sentence_id in index.entity_to_sentences.get(bridge_id, ()):
                        if second_sentence_id == first_sentence_id:
                            continue
                        answer_ids = self._rank_answer_candidates(
                            index,
                            second_sentence_id,
                            excluded_seeds | {bridge_id},
                            analysis.expected_answer_type,
                        )
                        for answer_id in answer_ids:
                            edge_ids = self._bridge_edge_ids(
                                index,
                                seed_edge,
                                first_sentence_id,
                                bridge_id,
                                second_sentence_id,
                                answer_id,
                            )
                            paths.append(
                                self._make_path(
                                    graph=graph,
                                    index=index,
                                    question_type=QuestionType.BRIDGE,
                                    prefix="bridge",
                                    node_ids=(
                                        seed_id,
                                        first_sentence_id,
                                        bridge_id,
                                        second_sentence_id,
                                        answer_id,
                                    ),
                                    edge_ids=edge_ids,
                                    sentence_ids=(first_sentence_id, second_sentence_id),
                                    entity_ids=(seed_id, bridge_id, answer_id),
                                    answer_id=answer_id,
                                    expected_answer_type=analysis.expected_answer_type,
                                    required_relations=tuple(
                                        hint.relation for hint in analysis.required_relation_hints
                                    ),
                                    bridge_id=bridge_id,
                                    path_kind="bridge_candidate",
                                )
                            )
        return paths

    def _search_factoid_paths(
        self, graph: EvidenceGraph, analysis: QuestionAnalysis, index: _GraphIndex
    ) -> list[EvidencePath]:
        paths: list[EvidencePath] = []
        excluded_seeds = set(graph.seed_entity_node_ids)
        for seed_id in graph.seed_entity_node_ids:
            for seed_edge in index.seed_to_sentences.get(seed_id, ()):
                sentence_id = seed_edge.target_id
                for answer_id in self._rank_answer_candidates(
                    index, sentence_id, excluded_seeds, analysis.expected_answer_type
                ):
                    mention = self._first_edge(
                        index, sentence_id, answer_id, GraphEdgeType.SENTENCE_MENTIONS_ENTITY
                    )
                    if mention is None:
                        continue
                    paths.append(
                        self._make_path(
                            graph=graph,
                            index=index,
                            question_type=QuestionType.FACTOID,
                            prefix="factoid",
                            node_ids=(seed_id, sentence_id, answer_id),
                            edge_ids=(seed_edge.edge_id, mention.edge_id),
                            sentence_ids=(sentence_id,),
                            entity_ids=(seed_id, answer_id),
                            answer_id=answer_id,
                            expected_answer_type=analysis.expected_answer_type,
                            required_relations=tuple(
                                hint.relation for hint in analysis.required_relation_hints
                            ),
                            path_kind="factoid_candidate",
                        )
                    )
        return paths

    def _search_comparison_partial_paths(
        self, graph: EvidenceGraph, analysis: QuestionAnalysis, index: _GraphIndex
    ) -> list[EvidencePath]:
        target_ids = tuple(
            node.node_id
            for target in analysis.comparison_targets
            for node in graph.nodes
            if node.node_type is GraphNodeType.ENTITY and _same_label(node.label, target.text)
        )
        if not target_ids:
            target_ids = graph.seed_entity_node_ids
        paths: list[EvidencePath] = []
        for target_id in _dedupe_preserve_order(target_ids):
            for sentence_id in index.entity_to_sentences.get(target_id, ()):
                answer_ids = self._rank_answer_candidates(
                    index, sentence_id, {target_id}, analysis.expected_answer_type
                )
                if not answer_ids:
                    continue
                answer_id = answer_ids[0]
                target_mention = self._first_edge(
                    index, sentence_id, target_id, GraphEdgeType.SENTENCE_MENTIONS_ENTITY
                )
                answer_mention = self._first_edge(
                    index, sentence_id, answer_id, GraphEdgeType.SENTENCE_MENTIONS_ENTITY
                )
                if target_mention is None or answer_mention is None:
                    continue
                paths.append(
                    self._make_path(
                        graph=graph,
                        index=index,
                        question_type=QuestionType.COMPARISON,
                        prefix="comparison_partial",
                        node_ids=(target_id, sentence_id, answer_id),
                        edge_ids=(target_mention.edge_id, answer_mention.edge_id),
                        sentence_ids=(sentence_id,),
                        entity_ids=(target_id, answer_id),
                        answer_id=answer_id,
                        expected_answer_type=analysis.expected_answer_type,
                        required_relations=tuple(
                            hint.relation for hint in analysis.required_relation_hints
                        ),
                        path_kind="comparison_target_partial",
                        comparison_target=index.nodes[target_id].label,
                    )
                )
        return paths

    def _search_yes_no_paths(
        self, graph: EvidenceGraph, analysis: QuestionAnalysis, index: _GraphIndex
    ) -> list[EvidencePath]:
        seed_ids = graph.seed_entity_node_ids
        if len(seed_ids) < 2:
            return self._search_factoid_paths(graph, analysis, index)
        paths: list[EvidencePath] = []
        for left_offset, left_id in enumerate(seed_ids):
            for right_id in seed_ids[left_offset + 1 :]:
                shared = sorted(
                    set(index.entity_to_sentences.get(left_id, ()))
                    & set(index.entity_to_sentences.get(right_id, ()))
                )
                for sentence_id in shared:
                    left = self._first_edge(
                        index, sentence_id, left_id, GraphEdgeType.SENTENCE_MENTIONS_ENTITY
                    )
                    right = self._first_edge(
                        index, sentence_id, right_id, GraphEdgeType.SENTENCE_MENTIONS_ENTITY
                    )
                    if left is None or right is None:
                        continue
                    paths.append(
                        self._make_yes_no_path(
                            graph,
                            index,
                            left_id,
                            sentence_id,
                            right_id,
                            (left.edge_id, right.edge_id),
                        )
                    )
        return paths

    def _make_path(
        self,
        *,
        graph: EvidenceGraph,
        index: _GraphIndex,
        question_type: QuestionType,
        prefix: str,
        node_ids: tuple[str, ...],
        edge_ids: tuple[str, ...],
        sentence_ids: tuple[str, ...],
        entity_ids: tuple[str, ...],
        answer_id: str,
        expected_answer_type: AnswerType,
        required_relations: tuple[str, ...],
        path_kind: PathKind,
        bridge_id: str | None = None,
        comparison_target: str | None = None,
    ) -> EvidencePath:
        evidence_ids, scored_units = self._provenance(index, sentence_ids)
        relations = self._relations(index, sentence_ids)
        breakdown, score = self._score_path(
            index=index,
            sentence_ids=sentence_ids,
            answer_sentence_id=sentence_ids[-1],
            expected_answer_type=expected_answer_type,
            required_relations=required_relations,
            is_bridge=bridge_id is not None,
            bridge_id=bridge_id,
            answer_id=answer_id,
        )
        metadata = EvidencePathMetadata(
            configuration_fingerprint=self._config.fingerprint(),
            path_kind=path_kind,
            source_graph=graph.metadata,
            bridge_entity=index.nodes[bridge_id].label if bridge_id else None,
            comparison_target=comparison_target,
            does_not_compare_values_yet=path_kind == "comparison_target_partial",
        )
        return EvidencePath(
            path_id=_stable_path_id(prefix, node_ids, edge_ids),
            question_type=question_type,
            node_ids=node_ids,
            edge_ids=_dedupe_preserve_order(edge_ids),
            evidence_unit_ids=evidence_ids,
            entity_chain=tuple(index.nodes[node_id].label for node_id in entity_ids),
            relation_chain=relations,
            answer_candidate=index.nodes[answer_id].label,
            answer_type=expected_answer_type,
            score=score,
            score_breakdown=breakdown,
            scored_evidence_units=scored_units,
            metadata=metadata,
        )

    def _make_yes_no_path(
        self,
        graph: EvidenceGraph,
        index: _GraphIndex,
        left_id: str,
        sentence_id: str,
        right_id: str,
        edge_ids: tuple[str, ...],
    ) -> EvidencePath:
        evidence_ids, scored_units = self._provenance(index, (sentence_id,))
        sentence_score = index.evidence_score_by_sentence[sentence_id]
        breakdown = PathScoreBreakdown(
            average_evidence_score=round(sentence_score, 6),
            expected_answer_type_match=0.0,
            relation_match_score=0.0,
            path_length_score=0.0,
            bridge_entity_quality=0.0,
            answer_specificity_score=0.0,
            retrieval_quality_score=0.0,
        )
        return EvidencePath(
            path_id=_stable_path_id("yes_no", (left_id, sentence_id, right_id), edge_ids),
            question_type=QuestionType.YES_NO,
            node_ids=(left_id, sentence_id, right_id),
            edge_ids=_dedupe_preserve_order(edge_ids),
            evidence_unit_ids=evidence_ids,
            entity_chain=(index.nodes[left_id].label, index.nodes[right_id].label),
            relation_chain=self._relations(index, (sentence_id,)),
            answer_candidate=None,
            answer_type=AnswerType.BOOLEAN,
            score=round(sentence_score + self._config.yes_no_connection_bonus, 6),
            score_breakdown=breakdown,
            scored_evidence_units=scored_units,
            metadata=EvidencePathMetadata(
                configuration_fingerprint=self._config.fingerprint(),
                path_kind="yes_no_evidence_connection",
                source_graph=graph.metadata,
                does_not_decide_yes_no=True,
            ),
        )

    def _rank_answer_candidates(
        self,
        index: _GraphIndex,
        sentence_id: str,
        excluded_ids: set[str],
        expected_answer_type: AnswerType,
    ) -> tuple[str, ...]:
        candidates = [
            node_id
            for node_id in index.sentence_to_entities.get(sentence_id, ())
            if node_id not in excluded_ids
        ]
        expected_type_match = expected_answer_type in index.sentence_to_answer_types.get(
            sentence_id, ()
        )

        def rank_key(node_id: str) -> tuple[float, str, str]:
            score = 0.0
            if node_id in index.possible_answer_targets.get(sentence_id, frozenset()):
                score += self._config.possible_answer_candidate_bonus
            if expected_type_match:
                score += self._config.sentence_answer_type_bonus
            label = index.nodes[node_id].label
            if _looks_specific(label):
                score += self._config.specific_candidate_bonus
            if _same_label(label, expected_answer_type.value):
                score -= self._config.generic_answer_type_penalty
            return (-score, label.casefold(), node_id)

        return tuple(sorted(candidates, key=rank_key))

    def _score_path(
        self,
        *,
        index: _GraphIndex,
        sentence_ids: tuple[str, ...],
        answer_sentence_id: str,
        expected_answer_type: AnswerType,
        required_relations: tuple[str, ...],
        is_bridge: bool,
        bridge_id: str | None,
        answer_id: str,
    ) -> tuple[PathScoreBreakdown, float]:
        average = sum(index.evidence_score_by_sentence[item] for item in sentence_ids) / len(
            sentence_ids
        )
        answer_type_match = float(
            expected_answer_type in index.sentence_to_answer_types.get(answer_sentence_id, ())
        )
        relations = self._relations(index, sentence_ids)
        matched = sum(_matches_any(relation, required_relations) for relation in relations)
        relation_score = min(1.0, matched / max(1, len(required_relations)))
        path_length = 1.0 if not is_bridge or len(set(sentence_ids)) == 2 else 0.4
        bridge_quality = 0.0
        if bridge_id is not None:
            bridge_quality = (
                self._config.specific_bridge_bonus
                if _looks_specific(index.nodes[bridge_id].label)
                else self._config.generic_bridge_bonus
            )
        answer_specificity = (
            self._config.answer_specificity_bonus
            if _looks_specific(index.nodes[answer_id].label)
            else 0.0
        )
        ranks = tuple(
            rank
            for sentence_id in sentence_ids
            if (rank := index.retrieval_rank_by_sentence[sentence_id]) is not None
        )
        retrieval_quality = 1.0 / min(ranks) if ranks else 0.0
        breakdown = PathScoreBreakdown(
            average_evidence_score=round(average, 6),
            expected_answer_type_match=round(answer_type_match, 6),
            relation_match_score=round(relation_score, 6),
            path_length_score=round(path_length, 6),
            bridge_entity_quality=round(bridge_quality, 6),
            answer_specificity_score=round(answer_specificity, 6),
            retrieval_quality_score=round(retrieval_quality, 6),
        )
        score = round(
            average
            + self._config.expected_answer_type_weight * answer_type_match
            + self._config.relation_match_weight * relation_score
            + self._config.path_length_weight * path_length
            + bridge_quality
            + answer_specificity
            + self._config.retrieval_quality_weight * retrieval_quality,
            6,
        )
        return breakdown, score

    def _build_index(self, graph: EvidenceGraph) -> _GraphIndex:
        nodes = {node.node_id: node for node in graph.nodes}
        outgoing: dict[tuple[str, GraphEdgeType], list[GraphEdge]] = defaultdict(list)
        sentence_entities: dict[str, list[str]] = defaultdict(list)
        entity_sentences: dict[str, list[str]] = defaultdict(list)
        sentence_relations: dict[str, list[str]] = defaultdict(list)
        sentence_answer_types: dict[str, list[AnswerType]] = defaultdict(list)
        seed_sentences: dict[str, list[GraphEdge]] = defaultdict(list)
        possible_answers: dict[str, set[str]] = defaultdict(set)
        evidence_by_sentence: dict[str, ScoredEvidenceUnit] = {}
        evidence_ids: dict[str, str] = {}
        evidence_scores: dict[str, float] = {}
        retrieval_ranks: dict[str, int | None] = {}

        for node in graph.nodes:
            if node.node_type is GraphNodeType.SENTENCE and node.scored_evidence is not None:
                evidence_by_sentence[node.node_id] = node.scored_evidence
                evidence_ids[node.node_id] = node.scored_evidence.evidence_unit.evidence_unit_id
                evidence_scores[node.node_id] = node.scored_evidence.final_score
                retrieval_ranks[node.node_id] = node.scored_evidence.evidence_unit.retrieval_rank
        for edge in graph.edges:
            outgoing[(edge.source_id, edge.edge_type)].append(edge)
            if edge.edge_type is GraphEdgeType.SENTENCE_MENTIONS_ENTITY:
                sentence_entities[edge.source_id].append(edge.target_id)
                entity_sentences[edge.target_id].append(edge.source_id)
            elif edge.edge_type is GraphEdgeType.SENTENCE_HAS_RELATION and edge.relation:
                sentence_relations[edge.source_id].append(edge.relation)
            elif edge.edge_type is GraphEdgeType.SENTENCE_HAS_ANSWER_TYPE:
                try:
                    sentence_answer_types[edge.source_id].append(
                        AnswerType(nodes[edge.target_id].label)
                    )
                except (KeyError, ValueError):
                    continue
            elif edge.edge_type is GraphEdgeType.SEED_ENTITY_TO_SENTENCE:
                seed_sentences[edge.source_id].append(edge)
            elif edge.edge_type is GraphEdgeType.POSSIBLE_ANSWER_CANDIDATE:
                possible_answers[edge.source_id].add(edge.target_id)

        return _GraphIndex(
            nodes=nodes,
            outgoing_by_type={key: tuple(value) for key, value in outgoing.items()},
            sentence_to_entities={
                key: _dedupe_preserve_order(value) for key, value in sentence_entities.items()
            },
            entity_to_sentences={
                key: tuple(sorted(set(value))) for key, value in entity_sentences.items()
            },
            sentence_to_relations={
                key: _dedupe_preserve_order(value) for key, value in sentence_relations.items()
            },
            sentence_to_answer_types={
                key: tuple(dict.fromkeys(value)) for key, value in sentence_answer_types.items()
            },
            seed_to_sentences={key: tuple(value) for key, value in seed_sentences.items()},
            possible_answer_targets={
                key: frozenset(value) for key, value in possible_answers.items()
            },
            evidence_by_sentence=evidence_by_sentence,
            evidence_unit_by_sentence=evidence_ids,
            evidence_score_by_sentence=evidence_scores,
            retrieval_rank_by_sentence=retrieval_ranks,
        )

    @staticmethod
    def _first_edge(
        index: _GraphIndex, source_id: str, target_id: str, edge_type: GraphEdgeType
    ) -> GraphEdge | None:
        return next(
            (
                edge
                for edge in index.outgoing_by_type.get((source_id, edge_type), ())
                if edge.target_id == target_id
            ),
            None,
        )

    def _bridge_edge_ids(
        self,
        index: _GraphIndex,
        seed_edge: GraphEdge,
        first_sentence_id: str,
        bridge_id: str,
        second_sentence_id: str,
        answer_id: str,
    ) -> tuple[str, ...]:
        edges = (
            seed_edge,
            self._first_edge(
                index, first_sentence_id, bridge_id, GraphEdgeType.SENTENCE_MENTIONS_ENTITY
            ),
            self._first_edge(
                index, second_sentence_id, bridge_id, GraphEdgeType.SENTENCE_MENTIONS_ENTITY
            ),
            self._first_edge(
                index, second_sentence_id, answer_id, GraphEdgeType.SENTENCE_MENTIONS_ENTITY
            ),
        )
        return _dedupe_preserve_order(edge.edge_id for edge in edges if edge is not None)

    @staticmethod
    def _relations(index: _GraphIndex, sentence_ids: tuple[str, ...]) -> tuple[str, ...]:
        return _dedupe_preserve_order(
            relation
            for sentence_id in sentence_ids
            for relation in index.sentence_to_relations.get(sentence_id, ())
        )

    @staticmethod
    def _provenance(
        index: _GraphIndex, sentence_ids: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[ScoredEvidenceUnit, ...]]:
        ordered = _dedupe_preserve_order(sentence_ids)
        return (
            tuple(index.evidence_unit_by_sentence[sentence_id] for sentence_id in ordered),
            tuple(index.evidence_by_sentence[sentence_id] for sentence_id in ordered),
        )

    @staticmethod
    def _deduplicate(paths: Iterable[EvidencePath]) -> list[EvidencePath]:
        best: dict[tuple[object, ...], EvidencePath] = {}
        for path in paths:
            signature = (
                path.question_type,
                path.evidence_unit_ids,
                path.entity_chain,
                path.answer_candidate,
            )
            current = best.get(signature)
            if current is None or (path.score, _inverted_path_id(path.path_id)) > (
                current.score,
                _inverted_path_id(current.path_id),
            ):
                best[signature] = path
        return list(best.values())

    def _emit_completed(
        self,
        graph: EvidenceGraph,
        paths: list[EvidencePath],
        context: TraceContext | None,
        started: float,
    ) -> None:
        if self._sink is not None and context is not None:
            self._sink.emit(
                InstrumentationEvent(
                    context=context,
                    event_type="epsa.evidence_path_search.completed",
                    source="epsa.evidence_path_search",
                    source_version="research_v1",
                    payload={
                        "latency_ms": round((perf_counter() - started) * 1000, 6),
                        "candidate_paths": len(paths),
                        "graph_schema_version": graph.metadata.schema_version,
                        "graph_version": graph.metadata.version,
                        "makes_sufficiency_decision": False,
                    },
                )
            )

    def _emit_failed(
        self,
        graph: object,
        error: EvidencePathSearchError,
        context: TraceContext | None,
        started: float,
    ) -> None:
        if self._sink is not None and context is not None:
            graph_version = graph.metadata.version if isinstance(graph, EvidenceGraph) else None
            self._sink.emit(
                InstrumentationEvent(
                    context=context,
                    event_type="epsa.evidence_path_search.failed",
                    source="epsa.evidence_path_search",
                    source_version="research_v1",
                    payload={
                        "latency_ms": round((perf_counter() - started) * 1000, 6),
                        "error_type": type(error).__name__,
                        "graph_version": graph_version,
                    },
                )
            )


def _stable_path_id(prefix: str, node_ids: tuple[str, ...], edge_ids: tuple[str, ...]) -> str:
    """Reproduce research-v1 SHA-1 IDs from ordered path topology."""

    digest = hashlib.sha1("|".join((prefix, *node_ids, *edge_ids)).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}::{digest}"


def _same_label(left: str, right: str) -> bool:
    return _normalize(left) == _normalize(right)


def _matches_any(value: str, candidates: tuple[str, ...]) -> bool:
    normalized = _normalize(value)
    return any(
        normalized == _normalize(candidate)
        or normalized in _normalize(candidate)
        or _normalize(candidate) in normalized
        for candidate in candidates
    )


def _normalize(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def _looks_specific(label: str) -> bool:
    generic = {
        "person",
        "location",
        "date",
        "number",
        "boolean",
        "entity",
        "organization",
        "title_or_work",
        "unknown",
    }
    return _normalize(label) not in generic and any(character.isalnum() for character in label)


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _inverted_path_id(value: str) -> tuple[int, ...]:
    """Prefer lexical-smaller IDs when scores tie, without affecting higher scores."""

    return tuple(-ord(character) for character in value)
