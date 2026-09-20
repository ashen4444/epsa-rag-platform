"""Deterministic paragraph-level EPSA Component 02."""

from __future__ import annotations

from time import perf_counter

from epsa_rag.core.exceptions import ChunkAnalysisError
from epsa_rag.epsa.chunk_analysis.chunk_adapter import adapt_retrieved_chunk
from epsa_rag.epsa.chunk_analysis.config import ChunkAnalyzerConfig
from epsa_rag.epsa.chunk_analysis.models import (
    BridgeCandidateDecision,
    CandidateChunkEvidence,
    ChunkAnalysisMetadata,
    ChunkEntityMention,
)
from epsa_rag.epsa.chunk_analysis.rules import (
    entity_matches,
    extract_answers,
    extract_entities,
    extract_relations,
    normalize_entity,
    token_overlap,
)
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext


class RuleBasedCandidateChunkEvidenceAnalyzer:
    """Reconstruct documented thesis rules without claiming a proven bridge or answer."""

    def __init__(
        self,
        *,
        config: ChunkAnalyzerConfig | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> None:
        self._config = config or ChunkAnalyzerConfig()
        self._instrumentation_sink = instrumentation_sink

    @property
    def config(self) -> ChunkAnalyzerConfig:
        return self._config

    def analyze(
        self,
        chunk: object,
        question_analysis: QuestionAnalysis | None = None,
        retrieval_rank: int | None = None,
        retrieval_score: float | None = None,
        *,
        trace_context: TraceContext | None = None,
    ) -> CandidateChunkEvidence:
        """Analyze one chunk; rank and score overrides take precedence over wrapper fields."""

        started = perf_counter()
        try:
            canonical = adapt_retrieved_chunk(
                chunk, retrieval_rank=retrieval_rank, retrieval_score=retrieval_score
            )
            body = canonical.paragraph_text or canonical.chunk_text
            analysis_text = "\n".join(part for part in (canonical.doc_title, body) if part)
            entities: list[ChunkEntityMention] = []
            seen: set[str] = set()
            title_norm = normalize_entity(canonical.doc_title)
            if title_norm:
                entities.append(
                    ChunkEntityMention(
                        text=canonical.doc_title.strip(),
                        normalized=title_norm,
                        source="doc_title",
                        confidence=self._config.title_entity_score,
                    )
                )
                seen.add(title_norm)
            for mention in extract_entities(body, self._config):
                if mention.normalized not in seen:
                    entities.append(mention)
                    seen.add(mention.normalized)
            question_entities = (
                ()
                if question_analysis is None
                else question_analysis.seed_entities + question_analysis.comparison_targets
            )
            question_norms = {normalize_entity(item.text) for item in question_entities}
            overlaps: list[str] = []
            overlap_seen: set[str] = set()
            for entity in entities:
                for question_entity in question_entities:
                    if entity_matches(entity.normalized, normalize_entity(question_entity.text)):
                        if entity.normalized not in overlap_seen:
                            overlaps.append(entity.text)
                            overlap_seen.add(entity.normalized)
            tokens, token_score = (
                ((), 0.0)
                if question_analysis is None
                else token_overlap(question_analysis.normalized_question, analysis_text)
            )
            title_match = False
            if title_norm and question_analysis is not None:
                title_match = (
                    title_norm in {normalize_entity(item) for item in overlaps}
                    or title_norm in normalize_entity(question_analysis.raw_question)
                    or any(entity_matches(title_norm, item) for item in question_norms)
                )
            bridges: list[ChunkEntityMention] = []
            decisions: list[BridgeCandidateDecision] = []
            for entity in entities:
                if entity.normalized == title_norm:
                    reason = "document_title"
                elif entity.normalized in question_norms:
                    reason = "question_entity"
                elif len(entity.normalized) < self._config.minimum_bridge_length:
                    reason = "too_short"
                else:
                    reason = "candidate"
                    bridges.append(entity)
                decisions.append(
                    BridgeCandidateDecision(
                        entity=entity.text,
                        accepted=reason == "candidate",
                        reason=reason,
                    )
                )
            evidence = CandidateChunkEvidence(
                chunk_id=canonical.chunk_id,
                doc_title=canonical.doc_title,
                paragraph_index=canonical.paragraph_index,
                retrieval_rank=canonical.retrieval_rank,
                retrieval_score=canonical.retrieval_score,
                entities=tuple(entities),
                relation_hints=extract_relations(analysis_text, self._config),
                answer_type_candidates=extract_answers(analysis_text, self._config),
                potential_bridge_entities=tuple(bridges),
                question_entity_overlap=tuple(overlaps),
                question_token_overlap=tokens,
                question_token_overlap_score=token_score,
                is_title_match=title_match,
                chunk_text=canonical.chunk_text,
                paragraph_text=canonical.paragraph_text,
                source_question_id=canonical.source_question_id,
                sentences=canonical.sentences,
                metadata=ChunkAnalysisMetadata(
                    configuration_fingerprint=self._config.fingerprint(),
                    bridge_decisions=tuple(decisions),
                ),
            )
        except ChunkAnalysisError as error:
            self._emit_failure(chunk, error, trace_context, started)
            raise
        except Exception as error:
            wrapped = ChunkAnalysisError("rule-based chunk analysis failed")
            self._emit_failure(chunk, wrapped, trace_context, started)
            raise wrapped from error
        self._emit_completed(evidence, trace_context, started)
        return evidence

    def _emit_completed(
        self, evidence: CandidateChunkEvidence, context: TraceContext | None, started: float
    ) -> None:
        if self._instrumentation_sink is None or context is None:
            return
        self._instrumentation_sink.emit(
            InstrumentationEvent(
                context=context,
                event_type="epsa.chunk_analysis.completed",
                source="epsa.chunk_analyzer",
                source_version=self._config.mode,
                payload={
                    "evidence": evidence.model_dump(mode="json"),
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            )
        )

    def _emit_failure(
        self, chunk: object, error: ChunkAnalysisError,
        context: TraceContext | None, started: float
    ) -> None:
        if self._instrumentation_sink is None or context is None:
            return
        chunk_id = (
            chunk.get("chunk_id") if isinstance(chunk, dict) else getattr(chunk, "chunk_id", None)
        )
        self._instrumentation_sink.emit(
            InstrumentationEvent(
                context=context,
                event_type="epsa.chunk_analysis.failed",
                source="epsa.chunk_analyzer",
                source_version=self._config.mode,
                payload={
                    "chunk_id": str(chunk_id) if chunk_id is not None else None,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            )
        )
