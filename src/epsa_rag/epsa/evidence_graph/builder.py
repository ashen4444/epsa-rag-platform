"""Deterministic, inference-safe implementation of EPSA Component 05 research-v1."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from epsa_rag.core.exceptions import EvidenceGraphBuildError
from epsa_rag.epsa.evidence_graph.config import EvidenceGraphBuilderV1Config
from epsa_rag.epsa.evidence_graph.models import (
    EvidenceGraph,
    EvidenceGraphMetadata,
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
)
from epsa_rag.epsa.evidence_scoring.models import ScoredEvidenceUnit
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext


class EvidenceGraphBuilderV1:
    """Build the documented ``research_v1`` graph from inference-visible inputs only."""

    def __init__(
        self,
        *,
        config: EvidenceGraphBuilderV1Config | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> None:
        self._config = config or EvidenceGraphBuilderV1Config()
        self._sink = instrumentation_sink

    @property
    def config(self) -> EvidenceGraphBuilderV1Config:
        return self._config

    def build(
        self,
        question_analysis: QuestionAnalysis,
        scored_evidence_units: Sequence[ScoredEvidenceUnit],
        *,
        trace_context: TraceContext | None = None,
    ) -> EvidenceGraph:
        """Build a deterministic pre-sufficiency graph without using gold labels."""

        started = perf_counter()
        try:
            self._validate_inputs(question_analysis, scored_evidence_units)
            graph = self._build_graph(question_analysis, scored_evidence_units)
        except EvidenceGraphBuildError as error:
            self._emit_failed(scored_evidence_units, error, trace_context, started)
            raise
        except (ValidationError, ValueError, TypeError) as error:
            wrapped = EvidenceGraphBuildError(str(error))
            self._emit_failed(scored_evidence_units, wrapped, trace_context, started)
            raise wrapped from error
        except Exception as error:
            wrapped = EvidenceGraphBuildError("evidence graph construction failed")
            self._emit_failed(scored_evidence_units, wrapped, trace_context, started)
            raise wrapped from error
        self._emit_completed(graph, trace_context, started)
        return graph

    def _validate_inputs(
        self,
        question_analysis: object,
        scored_evidence_units: object,
    ) -> None:
        if not isinstance(question_analysis, QuestionAnalysis):
            raise EvidenceGraphBuildError("question analysis must use Component 01 contract")
        if isinstance(scored_evidence_units, (str, bytes)) or not isinstance(
            scored_evidence_units, Sequence
        ):
            raise EvidenceGraphBuildError("scored evidence units must be a Component 04 sequence")
        if any(not isinstance(unit, ScoredEvidenceUnit) for unit in scored_evidence_units):
            raise EvidenceGraphBuildError("scored evidence units must use Component 04 contract")

    def _build_graph(
        self,
        question_analysis: QuestionAnalysis,
        scored_evidence_units: Sequence[ScoredEvidenceUnit],
    ) -> EvidenceGraph:
        node_map: dict[str, GraphNode] = {}
        edge_map: dict[str, GraphEdge] = {}
        seed_entities = _dedupe_preserve_order(
            entity.text for entity in question_analysis.seed_entities
        )
        relation_hints = _dedupe_preserve_order(
            hint.relation for hint in question_analysis.required_relation_hints
        )
        seed_norms = {_normalized_key(seed) for seed in seed_entities}
        seed_entity_node_ids: list[str] = []
        evidence_unit_node_ids: list[str] = []
        entity_node_ids: list[str] = []

        for seed_entity in seed_entities:
            node_id = stable_node_id(GraphNodeType.ENTITY, seed_entity)
            seed_entity_node_ids.append(node_id)
            _add_or_update_node(
                node_map,
                GraphNode(
                    node_id=node_id,
                    node_type=GraphNodeType.ENTITY,
                    label=seed_entity,
                    metadata={"is_question_seed": True, "original_label": seed_entity},
                ),
            )

        for scored_unit in scored_evidence_units:
            unit = scored_unit.evidence_unit
            evidence_score = scored_unit.final_score
            chunk_node_id = stable_node_id(GraphNodeType.CHUNK, unit.chunk_id)
            title_node_id = stable_node_id(GraphNodeType.TITLE, unit.doc_title)
            sentence_node_id = stable_sentence_node_id(unit.evidence_unit_id)
            evidence_unit_node_ids.append(sentence_node_id)

            _add_or_update_node(
                node_map,
                GraphNode(
                    node_id=chunk_node_id,
                    node_type=GraphNodeType.CHUNK,
                    label=unit.chunk_id,
                    metadata={
                        "chunk_id": unit.chunk_id,
                        "doc_title": unit.doc_title,
                        "paragraph_index": unit.paragraph_index,
                    },
                ),
            )
            _add_or_update_node(
                node_map,
                GraphNode(
                    node_id=title_node_id,
                    node_type=GraphNodeType.TITLE,
                    label=unit.doc_title,
                    metadata={
                        "doc_title": unit.doc_title,
                        "normalized_title": _normalized_key(unit.doc_title),
                    },
                ),
            )
            _add_or_update_node(
                node_map,
                GraphNode(
                    node_id=sentence_node_id,
                    node_type=GraphNodeType.SENTENCE,
                    label=unit.sentence_text,
                    metadata={
                        "evidence_unit_id": unit.evidence_unit_id,
                        "chunk_id": unit.chunk_id,
                        "doc_title": unit.doc_title,
                        "paragraph_index": unit.paragraph_index,
                        "sentence_id": unit.sentence_id,
                        "sentence_text": unit.sentence_text,
                        "resolved_text": unit.resolved_text,
                    },
                    scored_evidence=scored_unit,
                ),
            )
            _add_edge(
                edge_map,
                source_id=chunk_node_id,
                target_id=sentence_node_id,
                edge_type=GraphEdgeType.CHUNK_TO_SENTENCE,
                weight=evidence_score,
                evidence_unit_id=unit.evidence_unit_id,
                metadata={"chunk_id": unit.chunk_id},
            )
            _add_edge(
                edge_map,
                source_id=title_node_id,
                target_id=sentence_node_id,
                edge_type=GraphEdgeType.TITLE_TO_SENTENCE,
                weight=evidence_score,
                evidence_unit_id=unit.evidence_unit_id,
                metadata={"doc_title": unit.doc_title},
            )

            title_entity_node_id: str | None = None
            if unit.doc_title:
                title_entity_node_id = stable_node_id(GraphNodeType.ENTITY, unit.doc_title)
                entity_node_ids.append(title_entity_node_id)
                _add_or_update_node(
                    node_map,
                    GraphNode(
                        node_id=title_entity_node_id,
                        node_type=GraphNodeType.ENTITY,
                        label=unit.doc_title,
                        metadata={
                            "original_label": unit.doc_title,
                            "from_doc_title": True,
                            "is_question_seed": _normalized_key(unit.doc_title) in seed_norms,
                        },
                    ),
                )
                anchor_weight = max(evidence_score, self._config.minimum_anchor_weight)
                _add_edge(
                    edge_map,
                    source_id=title_node_id,
                    target_id=title_entity_node_id,
                    edge_type=GraphEdgeType.TITLE_TO_ENTITY,
                    weight=anchor_weight,
                    evidence_unit_id=unit.evidence_unit_id,
                    metadata={"doc_title": unit.doc_title},
                )
                _add_edge(
                    edge_map,
                    source_id=sentence_node_id,
                    target_id=title_entity_node_id,
                    edge_type=GraphEdgeType.SENTENCE_IN_DOCUMENT_ABOUT_ENTITY,
                    weight=anchor_weight,
                    evidence_unit_id=unit.evidence_unit_id,
                    metadata={"doc_title": unit.doc_title},
                )

            sentence_entity_node_ids: list[str] = []
            entities = _dedupe_preserve_order(unit.entities)
            for entity in entities:
                entity_node_id = stable_node_id(GraphNodeType.ENTITY, entity)
                sentence_entity_node_ids.append(entity_node_id)
                entity_node_ids.append(entity_node_id)
                _add_or_update_node(
                    node_map,
                    GraphNode(
                        node_id=entity_node_id,
                        node_type=GraphNodeType.ENTITY,
                        label=entity,
                        metadata={
                            "original_label": entity,
                            "is_question_seed": _normalized_key(entity) in seed_norms,
                        },
                    ),
                )
                _add_edge(
                    edge_map,
                    source_id=sentence_node_id,
                    target_id=entity_node_id,
                    edge_type=GraphEdgeType.SENTENCE_MENTIONS_ENTITY,
                    weight=evidence_score,
                    evidence_unit_id=unit.evidence_unit_id,
                    metadata={"entity": entity},
                )

            for relation_hint in _dedupe_preserve_order(unit.relation_hints):
                relation_node_id = stable_node_id(GraphNodeType.RELATION, relation_hint)
                matches_required = _matches_any(relation_hint, relation_hints)
                _add_or_update_node(
                    node_map,
                    GraphNode(
                        node_id=relation_node_id,
                        node_type=GraphNodeType.RELATION,
                        label=relation_hint,
                        metadata={
                            "relation": relation_hint,
                            "matches_required_relation": matches_required,
                        },
                    ),
                )
                _add_edge(
                    edge_map,
                    source_id=sentence_node_id,
                    target_id=relation_node_id,
                    edge_type=GraphEdgeType.SENTENCE_HAS_RELATION,
                    weight=evidence_score,
                    evidence_unit_id=unit.evidence_unit_id,
                    relation=relation_hint,
                    metadata={
                        "relation": relation_hint,
                        "matches_required_relation": matches_required,
                    },
                )

            answer_types = tuple(unit.answer_type_candidates)
            for answer_type in _dedupe_preserve_order(
                answer_type.value for answer_type in answer_types
            ):
                answer_type_node_id = stable_node_id(GraphNodeType.ANSWER_TYPE, answer_type)
                matches_expected = _same_label(
                    answer_type, question_analysis.expected_answer_type.value
                )
                _add_or_update_node(
                    node_map,
                    GraphNode(
                        node_id=answer_type_node_id,
                        node_type=GraphNodeType.ANSWER_TYPE,
                        label=answer_type,
                        metadata={
                            "answer_type": answer_type,
                            "matches_expected_answer_type": matches_expected,
                        },
                    ),
                )
                _add_edge(
                    edge_map,
                    source_id=sentence_node_id,
                    target_id=answer_type_node_id,
                    edge_type=GraphEdgeType.SENTENCE_HAS_ANSWER_TYPE,
                    weight=evidence_score,
                    evidence_unit_id=unit.evidence_unit_id,
                    metadata={
                        "answer_type": answer_type,
                        "matches_expected_answer_type": matches_expected,
                    },
                )

            for left_id, right_id in _unique_pairs(sentence_entity_node_ids):
                _add_edge(
                    edge_map,
                    source_id=left_id,
                    target_id=right_id,
                    edge_type=GraphEdgeType.ENTITY_COOCCURS_WITH_ENTITY,
                    weight=evidence_score,
                    evidence_unit_id=unit.evidence_unit_id,
                    metadata={"within_sentence_node_id": sentence_node_id},
                )

            for seed_node_id in self._matched_seed_node_ids(
                seed_entities=seed_entities,
                sentence_entities=entities,
                question_entity_overlap=unit.question_entity_overlap,
                doc_title=unit.doc_title,
                sentence_text=unit.resolved_text or unit.sentence_text,
            ):
                _add_edge(
                    edge_map,
                    source_id=seed_node_id,
                    target_id=sentence_node_id,
                    edge_type=GraphEdgeType.SEED_ENTITY_TO_SENTENCE,
                    weight=max(evidence_score, self._config.minimum_anchor_weight),
                    evidence_unit_id=unit.evidence_unit_id,
                    metadata={"reason": "seed_entity_match"},
                )
                for entity_node_id in sentence_entity_node_ids:
                    if entity_node_id != seed_node_id:
                        _add_edge(
                            edge_map,
                            source_id=seed_node_id,
                            target_id=entity_node_id,
                            edge_type=GraphEdgeType.POSSIBLE_BRIDGE,
                            weight=evidence_score,
                            evidence_unit_id=unit.evidence_unit_id,
                            metadata={
                                "via_sentence_node_id": sentence_node_id,
                                "reason": "non_seed_entity_in_seed_matched_sentence",
                            },
                        )

            if any(
                _same_label(answer_type.value, question_analysis.expected_answer_type.value)
                for answer_type in answer_types
            ):
                for entity_node_id in sentence_entity_node_ids:
                    entity_label = node_map[entity_node_id].label
                    if _normalized_key(entity_label) not in seed_norms:
                        _add_edge(
                            edge_map,
                            source_id=sentence_node_id,
                            target_id=entity_node_id,
                            edge_type=GraphEdgeType.POSSIBLE_ANSWER_CANDIDATE,
                            weight=evidence_score,
                            evidence_unit_id=unit.evidence_unit_id,
                            metadata={
                                "expected_answer_type": (
                                    question_analysis.expected_answer_type.value
                                ),
                                "reason": "entity_in_expected_answer_type_sentence",
                            },
                        )

        return EvidenceGraph(
            nodes=tuple(sorted(node_map.values(), key=lambda node: node.node_id)),
            edges=tuple(sorted(edge_map.values(), key=lambda edge: edge.edge_id)),
            question_type=question_analysis.question_type,
            seed_entity_node_ids=tuple(_dedupe_preserve_order(seed_entity_node_ids)),
            evidence_unit_node_ids=tuple(_dedupe_preserve_order(evidence_unit_node_ids)),
            entity_node_ids=tuple(_dedupe_preserve_order(entity_node_ids)),
            metadata=EvidenceGraphMetadata(
                configuration_fingerprint=self._config.fingerprint(),
                expected_answer_type=question_analysis.expected_answer_type,
                required_relation_hints=tuple(relation_hints),
                num_scored_evidence_units=len(scored_evidence_units),
            ),
        )

    def _matched_seed_node_ids(
        self,
        *,
        seed_entities: Sequence[str],
        sentence_entities: Sequence[str],
        question_entity_overlap: Sequence[str],
        doc_title: str,
        sentence_text: str,
    ) -> tuple[str, ...]:
        sentence_entity_norms = {_normalized_key(entity) for entity in sentence_entities}
        overlap_norms = {_normalized_key(entity) for entity in question_entity_overlap}
        title_norm = _normalized_key(doc_title)
        sentence_norm = f" {_normalized_key(sentence_text).replace('_', ' ')} "
        matches = []
        for seed in seed_entities:
            seed_norm = _normalized_key(seed)
            if (
                seed_norm in sentence_entity_norms
                or seed_norm in overlap_norms
                or seed_norm == title_norm
                or f" {seed_norm.replace('_', ' ')} " in sentence_norm
            ):
                matches.append(stable_node_id(GraphNodeType.ENTITY, seed))
        return tuple(_dedupe_preserve_order(matches))

    def _emit_completed(
        self, graph: EvidenceGraph, context: TraceContext | None, started: float
    ) -> None:
        if self._sink is not None and context is not None:
            self._sink.emit(
                InstrumentationEvent(
                    context=context,
                    event_type="epsa.evidence_graph.completed",
                    source="epsa.evidence_graph",
                    source_version=self._config.mode,
                    payload={
                        "evidence_graph": graph.model_dump(mode="json"),
                        "latency_ms": (perf_counter() - started) * 1000,
                    },
                )
            )

    def _emit_failed(
        self,
        units: object,
        error: EvidenceGraphBuildError,
        context: TraceContext | None,
        started: float,
    ) -> None:
        if self._sink is not None and context is not None:
            count = len(units) if isinstance(units, Sequence) and not isinstance(units, str) else 0
            self._sink.emit(
                InstrumentationEvent(
                    context=context,
                    event_type="epsa.evidence_graph.failed",
                    source="epsa.evidence_graph",
                    source_version=self._config.mode,
                    payload={
                        "scored_evidence_unit_count": count,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "latency_ms": (perf_counter() - started) * 1000,
                    },
                )
            )


def stable_node_id(node_type: GraphNodeType | str, label: str) -> str:
    """Return the frozen research-v1 surface-normalized graph node identifier."""

    type_text = node_type.value if isinstance(node_type, GraphNodeType) else node_type
    return f"{type_text}::{_normalized_key(label)}"


def stable_sentence_node_id(evidence_unit_id: str) -> str:
    """Return the sentence node ID that directly embeds the Component 03 identifier."""

    return f"sentence::{evidence_unit_id}"


def stable_edge_id(
    source_id: str,
    target_id: str,
    edge_type: GraphEdgeType | str,
    evidence_unit_id: str | None = None,
    relation: str | None = None,
) -> str:
    """Return the historical stable edge ID used by research-v1."""

    type_text = edge_type.value if isinstance(edge_type, GraphEdgeType) else edge_type
    return (
        f"edge::{type_text}::{source_id}::{target_id}::"
        f"evidence::{_normalized_key(evidence_unit_id or 'global')}::"
        f"relation::{_normalized_key(relation or 'none')}"
    )


def _add_or_update_node(node_map: dict[str, GraphNode], node: GraphNode) -> None:
    existing = node_map.get(node.node_id)
    if existing is None:
        node_map[node.node_id] = node
        return
    metadata = dict(existing.metadata)
    for key, value in node.metadata.items():
        if key not in metadata:
            metadata[key] = value
        elif isinstance(metadata[key], bool) and isinstance(value, bool):
            metadata[key] = metadata[key] or value
        elif key == "original_label" and metadata[key] != value:
            existing_labels = metadata.get("original_labels")
            labels = list(existing_labels) if isinstance(existing_labels, list) else [metadata[key]]
            if value not in labels:
                labels.append(value)
            metadata["original_labels"] = labels
    node_map[node.node_id] = existing.model_copy(update={"metadata": metadata})


def _add_edge(
    edge_map: dict[str, GraphEdge],
    *,
    source_id: str,
    target_id: str,
    edge_type: GraphEdgeType,
    weight: float,
    evidence_unit_id: str | None = None,
    relation: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    edge_id = stable_edge_id(source_id, target_id, edge_type, evidence_unit_id, relation)
    clean_weight = max(0.0, weight)
    existing = edge_map.get(edge_id)
    if existing is None:
        edge_map[edge_id] = GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=clean_weight,
            evidence_unit_id=evidence_unit_id,
            relation=relation,
            metadata=metadata or {},
        )
    elif clean_weight > existing.weight:
        merged = dict(existing.metadata)
        merged.update(metadata or {})
        edge_map[edge_id] = existing.model_copy(update={"weight": clean_weight, "metadata": merged})


def _normalized_key(value: object) -> str:
    text = str(value or "")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = _normalized_key(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _same_label(left: str, right: str) -> bool:
    return _normalized_key(left) == _normalized_key(right)


def _matches_any(value: str, candidates: Sequence[str]) -> bool:
    value_normalized = _normalized_key(value).replace("_", " ")
    return any(
        value_normalized == _normalized_key(candidate).replace("_", " ")
        or value_normalized in _normalized_key(candidate).replace("_", " ")
        or _normalized_key(candidate).replace("_", " ") in value_normalized
        for candidate in candidates
    )


def _unique_pairs(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    unique = _dedupe_preserve_order(values)
    pairs: set[tuple[str, str]] = set()
    for index, left in enumerate(unique):
        for right in unique[index + 1 :]:
            if left != right:
                pairs.add((left, right) if left < right else (right, left))
    return tuple(sorted(pairs))
