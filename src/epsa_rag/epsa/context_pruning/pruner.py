"""Deterministic research-v1 sentence context pruning for EPSA Component 08."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from time import perf_counter

from pydantic import ValidationError

from epsa_rag.core.exceptions import ContextPruningError
from epsa_rag.epsa.context_pruning.config import ResearchContextPrunerV1Config
from epsa_rag.epsa.context_pruning.models import (
    PrunedContext,
    PrunedContextMetadata,
    PruningDiagnostics,
    PruningStrategy,
)
from epsa_rag.epsa.evidence_scoring.models import ScoredEvidenceUnit
from epsa_rag.epsa.question_analysis.models import QuestionType
from epsa_rag.epsa.sufficiency_decision.models import SufficiencyDecision
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext


class ResearchContextPrunerV1:
    """Format Component 07-selected evidence, with historical bridge-neighbor expansion."""

    def __init__(
        self,
        *,
        config: ResearchContextPrunerV1Config | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> None:
        self._config = config or ResearchContextPrunerV1Config()
        self._sink = instrumentation_sink

    @property
    def config(self) -> ResearchContextPrunerV1Config:
        """Return the immutable research-v1 configuration."""

        return self._config

    def prune(
        self,
        sufficiency_decision: SufficiencyDecision,
        scored_evidence_units: Sequence[ScoredEvidenceUnit],
        *,
        trace_context: TraceContext | None = None,
    ) -> PrunedContext:
        """Return diagnostic context without replacing Component 07 selection authority."""

        started = perf_counter()
        try:
            self._validate_inputs(sufficiency_decision, scored_evidence_units)
            context = self._prune(sufficiency_decision, scored_evidence_units)
        except ContextPruningError as error:
            self._emit_failed(error, trace_context, started)
            raise
        except (ValidationError, ValueError, TypeError) as error:
            wrapped = ContextPruningError(str(error))
            self._emit_failed(wrapped, trace_context, started)
            raise wrapped from error
        except Exception as error:
            wrapped = ContextPruningError("context pruning failed")
            self._emit_failed(wrapped, trace_context, started)
            raise wrapped from error
        self._emit_completed(context, trace_context, started)
        return context

    @staticmethod
    def _validate_inputs(decision: object, evidence: object) -> None:
        if not isinstance(decision, SufficiencyDecision):
            raise ContextPruningError("sufficiency decision must use Component 07 contract")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise ContextPruningError("scored evidence units must be a sequence")
        if any(not isinstance(unit, ScoredEvidenceUnit) for unit in evidence):
            raise ContextPruningError("scored evidence units must use Component 04 contract")

    def _prune(
        self,
        decision: SufficiencyDecision,
        evidence: Sequence[ScoredEvidenceUnit],
    ) -> PrunedContext:
        all_units_by_id = {unit.evidence_unit.evidence_unit_id: unit for unit in evidence}
        all_ids = tuple(unit.evidence_unit.evidence_unit_id for unit in evidence)
        requested_ids = _dedupe_preserve_order(decision.selected_evidence_unit_ids)
        mandatory = [
            all_units_by_id[unit_id] for unit_id in requested_ids if unit_id in all_units_by_id
        ]
        expanded = self._expand_bridge_context_if_needed(decision, mandatory, evidence)
        selected = tuple(sorted(expanded, key=_unit_sort_key))
        selected_ids = tuple(unit.evidence_unit.evidence_unit_id for unit in selected)
        selected_id_set = set(selected_ids)
        missing_ids = tuple(unit_id for unit_id in requested_ids if unit_id not in all_units_by_id)
        mandatory_ids = {unit.evidence_unit.evidence_unit_id for unit in mandatory}
        strategy = _strategy_for(decision, selected_ids, requested_ids)
        text = "\n\n".join(_format_evidence_unit(unit) for unit in selected)
        return PrunedContext(
            selected_chunk_ids=_dedupe_preserve_order(
                unit.evidence_unit.chunk_id for unit in selected
            ),
            selected_evidence_unit_ids=selected_ids,
            selected_evidence_units=selected,
            selected_sentences=tuple(
                unit.evidence_unit.resolved_text or unit.evidence_unit.sentence_text
                for unit in selected
            ),
            selected_context_text=text,
            estimated_context_tokens=_estimate_tokens(text),
            pruning_strategy=strategy,
            removed_evidence_unit_ids=tuple(
                unit_id for unit_id in all_ids if unit_id not in selected_id_set
            ),
            diagnostics=PruningDiagnostics(
                requested_evidence_unit_ids=requested_ids,
                missing_requested_evidence_unit_ids=missing_ids,
                input_evidence_unit_count=len(evidence),
                mandatory_evidence_unit_count=len(mandatory_ids),
                neighbor_evidence_unit_count=sum(
                    unit_id not in mandatory_ids for unit_id in selected_ids
                ),
            ),
            metadata=PrunedContextMetadata(
                configuration_fingerprint=self._config.fingerprint(),
                source_sufficiency_decision=decision.metadata,
            ),
        )

    def _expand_bridge_context_if_needed(
        self,
        decision: SufficiencyDecision,
        selected: Sequence[ScoredEvidenceUnit],
        evidence: Sequence[ScoredEvidenceUnit],
    ) -> list[ScoredEvidenceUnit]:
        """Preserve the historical input-order expansion and exact six-unit stop condition."""

        if not decision.sufficient or decision.question_type is not QuestionType.BRIDGE:
            return list(selected)
        if not selected:
            return list(selected)
        expanded = {unit.evidence_unit.evidence_unit_id: unit for unit in selected}
        selected_chunk_ids = {unit.evidence_unit.chunk_id for unit in selected}
        selected_sentence_keys = {
            (unit.evidence_unit.chunk_id, unit.evidence_unit.sentence_id) for unit in selected
        }
        for candidate in evidence:
            unit = candidate.evidence_unit
            if unit.chunk_id not in selected_chunk_ids:
                continue
            candidate_key = (unit.chunk_id, unit.sentence_id)
            if any(
                candidate_key[0] == selected_key[0]
                and abs(candidate_key[1] - selected_key[1]) <= self._config.neighbor_distance
                for selected_key in selected_sentence_keys
            ):
                expanded[unit.evidence_unit_id] = candidate
            if len(expanded) >= self._config.max_expanded_units:
                break
        return list(expanded.values())

    def _emit_completed(
        self, context: PrunedContext, trace: TraceContext | None, started: float
    ) -> None:
        if self._sink is not None and trace is not None:
            self._sink.emit(
                InstrumentationEvent(
                    context=trace,
                    event_type="epsa.context_pruning.completed",
                    source="epsa.context_pruning",
                    source_version="research_v1",
                    payload={
                        "latency_ms": round((perf_counter() - started) * 1000, 6),
                        "strategy": context.pruning_strategy.value,
                        "selected_evidence_units": len(context.selected_evidence_unit_ids),
                        "missing_requested_evidence_units": len(
                            context.diagnostics.missing_requested_evidence_unit_ids
                        ),
                        "estimated_context_tokens": context.estimated_context_tokens,
                    },
                )
            )

    def _emit_failed(
        self, error: ContextPruningError, trace: TraceContext | None, started: float
    ) -> None:
        if self._sink is not None and trace is not None:
            self._sink.emit(
                InstrumentationEvent(
                    context=trace,
                    event_type="epsa.context_pruning.failed",
                    source="epsa.context_pruning",
                    source_version="research_v1",
                    payload={
                        "latency_ms": round((perf_counter() - started) * 1000, 6),
                        "error_type": type(error).__name__,
                    },
                )
            )


def _strategy_for(
    decision: SufficiencyDecision,
    selected_ids: tuple[str, ...],
    requested_ids: tuple[str, ...],
) -> PruningStrategy:
    if not selected_ids:
        return PruningStrategy.EMPTY_EVIDENCE
    if (
        decision.sufficient
        and decision.question_type is QuestionType.BRIDGE
        and len(selected_ids) > len(requested_ids)
    ):
        return PruningStrategy.SUFFICIENT_BRIDGE_NEIGHBOR_SENTENCE
    if decision.sufficient:
        return PruningStrategy.SUFFICIENT_PATH_SENTENCE
    return PruningStrategy.PARTIAL_EVIDENCE_SENTENCE


def _format_evidence_unit(scored_unit: ScoredEvidenceUnit) -> str:
    unit = scored_unit.evidence_unit
    text = unit.resolved_text or unit.sentence_text
    header = f"[Title: {unit.doc_title} | Chunk: {unit.chunk_id} | Sentence: {unit.sentence_id}]"
    return f"{header}\n{text}"


def _unit_sort_key(scored_unit: ScoredEvidenceUnit) -> tuple[int, str, int, str]:
    unit = scored_unit.evidence_unit
    rank = unit.retrieval_rank if unit.retrieval_rank is not None else 10**9
    return (rank, unit.chunk_id, unit.sentence_id, unit.evidence_unit_id)


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
