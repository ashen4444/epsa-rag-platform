"""Versioned, context-aware Candidate Chunk Evidence Analyzer."""

from __future__ import annotations

import re
from collections import defaultdict
from time import perf_counter

from epsa_rag.core.exceptions import ChunkAnalysisError
from epsa_rag.epsa.chunk_analysis.chunk_adapter import adapt_retrieved_chunk
from epsa_rag.epsa.chunk_analysis.config import RuleBasedV2ChunkAnalyzerConfig
from epsa_rag.epsa.chunk_analysis.models import (
    BridgeCandidateDecisionV2,
    CandidateChunkEvidence,
    CanonicalRetrievedChunk,
    ChunkAnalysisMetadataV2,
    ChunkEntityMention,
    ChunkEntityMentionV2,
    GroundedChunkRelationHint,
)
from epsa_rag.epsa.chunk_analysis.rules import token_overlap
from epsa_rag.epsa.chunk_analysis.v2_rules import (
    contextual_answers,
    contextual_entities,
    contextual_relations,
    is_specific_entity,
)
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.epsa.question_analysis.normalizer import normalize_entity
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext

_LinkIndex = dict[str, tuple[tuple[str, str], ...]]


class RuleBasedV2CandidateChunkEvidenceAnalyzer:
    """Generate relation-supported bridge candidates with retrieved-set connectivity.

    Connectivity means an exact normalized mention links to a different retrieved
    paragraph, preferably its title. This is a candidate signal, not path proof.
    """

    def __init__(
        self,
        *,
        config: RuleBasedV2ChunkAnalyzerConfig | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> None:
        self._v2_config = config or RuleBasedV2ChunkAnalyzerConfig()
        self._instrumentation_sink = instrumentation_sink

    @property
    def config(self) -> RuleBasedV2ChunkAnalyzerConfig:
        return self._v2_config

    def analyze(
        self,
        chunk: object,
        question_analysis: QuestionAnalysis | None = None,
        retrieval_rank: int | None = None,
        retrieval_score: float | None = None,
        *,
        trace_context: TraceContext | None = None,
        retrieved_chunks: tuple[object, ...] | None = None,
    ) -> CandidateChunkEvidence:
        """Analyze one chunk; absent retrieved context leaves bridges unconfirmed."""

        started = perf_counter()
        try:
            canonical = adapt_retrieved_chunk(
                chunk, retrieval_rank=retrieval_rank, retrieval_score=retrieval_score
            )
            context = (
                self._link_index(tuple(adapt_retrieved_chunk(item) for item in retrieved_chunks))
                if retrieved_chunks is not None else None
            )
            evidence = self._analyze_canonical(canonical, question_analysis, context)
        except ChunkAnalysisError as error:
            self._emit_failure(chunk, error, trace_context, started)
            raise
        except Exception as error:
            wrapped = ChunkAnalysisError("contextual chunk analysis failed")
            self._emit_failure(chunk, wrapped, trace_context, started)
            raise wrapped from error
        self._emit_completed(evidence, trace_context, started)
        return evidence

    def analyze_batch(
        self,
        chunks: tuple[object, ...],
        question_analysis: QuestionAnalysis | None,
        *,
        trace_context: TraceContext | None = None,
    ) -> tuple[CandidateChunkEvidence, ...]:
        """Share one inference-only retrieved-set index across the paragraph analyses."""

        adapted: list[CanonicalRetrievedChunk] = []
        for item in chunks:
            started = perf_counter()
            try:
                adapted.append(adapt_retrieved_chunk(item))
            except ChunkAnalysisError as error:
                self._emit_failure(item, error, trace_context, started)
                raise
            except Exception as error:
                wrapped = ChunkAnalysisError("invalid retrieved chunk in contextual batch")
                self._emit_failure(item, wrapped, trace_context, started)
                raise wrapped from error
        canonical = tuple(adapted)
        context = self._link_index(canonical)
        evidence: list[CandidateChunkEvidence] = []
        for chunk in canonical:
            started = perf_counter()
            try:
                analyzed = self._analyze_canonical(chunk, question_analysis, context)
            except ChunkAnalysisError as error:
                self._emit_failure(chunk, error, trace_context, started)
                raise
            except Exception as error:
                wrapped = ChunkAnalysisError("contextual chunk analysis failed")
                self._emit_failure(chunk, wrapped, trace_context, started)
                raise wrapped from error
            self._emit_completed(analyzed, trace_context, started)
            evidence.append(analyzed)
        return tuple(evidence)

    def _link_index(self, chunks: tuple[CanonicalRetrievedChunk, ...]) -> _LinkIndex:
        occurrences: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for chunk in chunks:
            title = normalize_entity(chunk.doc_title)
            names = [(title, "title")]
            title_words = re.findall(r"[A-Za-z]+", chunk.doc_title)
            if len(title_words) >= 3:
                alias = "".join(word[0] for word in title_words).casefold()
                if len(alias) >= self._v2_config.minimum_bridge_length:
                    names.append((alias, "title_alias"))
            if self._v2_config.allow_body_mention_links:
                names.extend(
                    (entity.normalized, "body_mention") for entity in contextual_entities(
                        chunk.paragraph_text, self._v2_config
                    ) if is_specific_entity(entity, self._v2_config)
                )
            for name, basis in names:
                if name and all(
                    identifier != chunk.chunk_id for identifier, _ in occurrences[name]
                ):
                    occurrences[name].append((chunk.chunk_id, basis))
        return {name: tuple(entries) for name, entries in occurrences.items()}

    def _analyze_canonical(
        self,
        chunk: CanonicalRetrievedChunk,
        question_analysis: QuestionAnalysis | None,
        context: _LinkIndex | None,
    ) -> CandidateChunkEvidence:
        if not chunk.paragraph_text.strip():
            raise ChunkAnalysisError("contextual analysis requires paragraph text")
        title_norm = normalize_entity(chunk.doc_title)
        body_mentions = contextual_entities(chunk.paragraph_text, self._v2_config)
        entities: list[ChunkEntityMentionV2] = []
        seen: set[str] = set()
        if title_norm:
            entities.append(
                ChunkEntityMentionV2(
                    text=chunk.doc_title.strip(),
                    normalized=title_norm,
                    source="doc_title",
                    span_scope="doc_title",
                    confidence=self._v2_config.title_entity_score,
                )
            )
            seen.add(title_norm)
        for mention in body_mentions:
            if mention.normalized not in seen:
                entities.append(mention)
                seen.add(mention.normalized)
        question_norms = {
            normalize_entity(item.text)
            for item in (
                () if question_analysis is None else
                question_analysis.seed_entities + question_analysis.comparison_targets
            )
        }
        overlaps = tuple(entity.text for entity in entities if entity.normalized in question_norms)
        title_match = bool(title_norm and title_norm in question_norms)
        analysis_text = "\n".join(part for part in (chunk.doc_title, chunk.paragraph_text) if part)
        tokens, token_score = (
            token_overlap(question_analysis.normalized_question, analysis_text)
            if question_analysis is not None else ((), 0.0)
        )
        relations = contextual_relations(
            chunk.doc_title, chunk.paragraph_text, chunk.sentences,
            body_mentions, self._v2_config,
        )
        answers = contextual_answers(
            chunk.doc_title, chunk.paragraph_text, tuple(entities), self._v2_config
        )
        required = (
            tuple(hint.relation for hint in question_analysis.required_relation_hints)
            if question_analysis is not None else ()
        )
        answer_type = (
            question_analysis.expected_answer_type if question_analysis is not None else None
        )
        answer_names = {
            normalize_entity(answer.text)
            for answer in answers if answer.answer_type == answer_type
        }
        is_seed_side = bool(overlaps or title_match)
        bridges: list[ChunkEntityMentionV2] = []
        decisions: list[BridgeCandidateDecisionV2] = []
        for entity in entities:
            linked = (
                tuple((identifier, basis) for identifier, basis in
                      context.get(entity.normalized, ())
                      if identifier != chunk.chunk_id
                      and (basis != "body_mention" or len(entity.normalized.split()) > 1)
                      and (basis != "title_alias" or entity.text.isupper()))
                if context is not None else ()
            )
            support = self._supporting_relation(
                entity, relations, required, chunk.paragraph_text,
                allow_sentence_window=any(basis != "body_mention" for _, basis in linked),
            )
            strongest_link_basis = next(
                (basis for basis in ("title", "title_alias", "body_mention")
                 if any(link_basis == basis for _, link_basis in linked)),
                None,
            )
            relation = support[0] if support is not None else None
            if entity.normalized == title_norm:
                reason = "document_title"
            elif entity.normalized in question_norms:
                reason = "question_entity"
            elif not is_specific_entity(entity, self._v2_config):
                reason = "non_specific"
            elif not is_seed_side:
                reason = "not_seed_side"
            elif context is None:
                reason = "no_context"
            elif not linked:
                reason = "no_cross_chunk_link"
            elif relation is None:
                reason = "no_required_relation"
            elif entity.normalized in answer_names and (
                not required or relation == required[-1]
            ):
                reason = "answer_role"
            else:
                reason = "candidate"
                bridges.append(entity)
            decisions.append(
                BridgeCandidateDecisionV2(
                    entity=entity.text,
                    accepted=reason == "candidate",
                    reason=reason,
                    relation=relation,
                    linked_chunk_ids=(
                        tuple(identifier for identifier, _ in linked)
                        if reason == "candidate" else ()
                    ),
                    relation_basis=support[1] if support is not None else None,
                    link_basis=strongest_link_basis if reason == "candidate" else None,
                    matches_question_relation=bool(support is not None and support[2]),
                )
            )
        return CandidateChunkEvidence(
            chunk_id=chunk.chunk_id,
            doc_title=chunk.doc_title,
            paragraph_index=chunk.paragraph_index,
            retrieval_rank=chunk.retrieval_rank,
            retrieval_score=chunk.retrieval_score,
            entities=tuple(entities),
            relation_hints=relations,
            answer_type_candidates=answers,
            potential_bridge_entities=tuple(bridges),
            question_entity_overlap=overlaps,
            question_token_overlap=tokens,
            question_token_overlap_score=token_score,
            is_title_match=title_match,
            chunk_text=chunk.chunk_text,
            paragraph_text=chunk.paragraph_text,
            source_question_id=chunk.source_question_id,
            sentences=chunk.sentences,
            metadata=ChunkAnalysisMetadataV2(
                configuration_fingerprint=self._v2_config.fingerprint(),
                bridge_decisions=tuple(decisions),
                fallbacks_used=("no_retrieved_context",) if context is None else (),
            ),
        )

    def _supporting_relation(
        self,
        entity: ChunkEntityMention,
        relations: tuple[GroundedChunkRelationHint, ...],
        required: tuple[str, ...],
        body: str,
        *,
        allow_sentence_window: bool,
    ) -> tuple[str, str, bool] | None:
        direct: list[GroundedChunkRelationHint] = []
        for hint in relations:
            if entity.normalized in {
                normalize_entity(item)
                for item in (hint.subject_entity, hint.object_entity) if item
            }:
                direct.append(hint)
        if direct:
            best = next((hint for hint in direct if hint.relation in required), direct[0])
            return best.relation, "entity_pair", best.relation in required
        if (
            not allow_sentence_window or len(entity.normalized.split()) < 2
            or entity.start_char is None
        ):
            return None
        for hint in relations:
            relation_start = hint.start_char
            relation_end = hint.end_char
            gap = (
                body[relation_end:entity.start_char]
                if relation_end <= entity.start_char else
                body[entity.end_char or entity.start_char:relation_start]
            )
            if (
                len(gap) <= min(self._v2_config.relation_window_chars, 32)
                and not re.search(r"[.;!?]", gap)
            ):
                return hint.relation, "sentence_window", hint.relation in required
        return None

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
                source_version=self._v2_config.mode,
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
                source_version=self._v2_config.mode,
                payload={
                    "chunk_id": str(chunk_id) if chunk_id is not None else None,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            )
        )
