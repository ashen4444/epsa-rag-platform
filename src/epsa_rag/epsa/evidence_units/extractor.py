"""Deterministic, observable sentence-level evidence extraction."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from pydantic import ValidationError

from epsa_rag.core.exceptions import ChunkAnalysisError, EvidenceUnitExtractionError
from epsa_rag.core.ids import make_evidence_unit_id
from epsa_rag.epsa.chunk_analysis.chunk_adapter import adapt_retrieved_chunk
from epsa_rag.epsa.chunk_analysis.models import CandidateChunkEvidence, CanonicalRetrievedChunk
from epsa_rag.epsa.evidence_units import historical_rules, production_rules
from epsa_rag.epsa.evidence_units.config import (
    EvidenceUnitExtractorConfig,
    EvidenceUnitExtractorV2Config,
)
from epsa_rag.epsa.evidence_units.models import (
    EvidenceUnit,
    EvidenceUnitMetadata,
    SentenceEntity,
    StructuredAnswerCandidate,
)
from epsa_rag.epsa.evidence_units.resolution import (
    ConservativeTitlePronounResolver,
    CoreferenceResolver,
    TitlePronounResolver,
)
from epsa_rag.epsa.evidence_units.sentences import normalize_sentences
from epsa_rag.epsa.question_analysis.models import AnswerType, QuestionAnalysis
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext


class RuleBasedEvidenceUnitExtractor:
    """Thesis-compatible inference features with a separate gold-label boundary."""

    def __init__(
        self,
        *,
        config: EvidenceUnitExtractorConfig | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
        resolver: CoreferenceResolver | None = None,
    ) -> None:
        self._config = config or EvidenceUnitExtractorConfig()
        self._sink = instrumentation_sink
        if self._config.mode == "rule_based_v1" and resolver is not None:
            raise ValueError("thesis-compatible mode uses its frozen title resolver")
        self._resolver = resolver or (
            ConservativeTitlePronounResolver()
            if self._config.mode == "rule_based_v2"
            else TitlePronounResolver()
        )
        if self._resolver.version != self._config.resolver_version:
            raise ValueError("configured resolver does not match extractor mode")

    @property
    def config(self) -> EvidenceUnitExtractorConfig:
        return self._config

    def extract_from_chunk(
        self,
        candidate_evidence: CandidateChunkEvidence,
        chunk: object,
        question_analysis: QuestionAnalysis,
        *,
        trace_context: TraceContext | None = None,
    ) -> tuple[EvidenceUnit, ...]:
        started = perf_counter()
        try:
            if not isinstance(candidate_evidence, CandidateChunkEvidence):
                raise EvidenceUnitExtractionError(
                    "candidate evidence must use Component 02 contract"
                )
            if not isinstance(question_analysis, QuestionAnalysis):
                raise EvidenceUnitExtractionError(
                    "question analysis must use Component 01 contract"
                )
            canonical = adapt_retrieved_chunk(chunk)
            self._validate_pair(candidate_evidence, canonical)
            units = self._extract(candidate_evidence, canonical, question_analysis)
        except (
            EvidenceUnitExtractionError,
            ChunkAnalysisError,
            ValidationError,
            ValueError,
        ) as error:
            wrapped = (
                error
                if isinstance(error, EvidenceUnitExtractionError)
                else EvidenceUnitExtractionError(str(error))
            )
            self._emit_failed(candidate_evidence, wrapped, trace_context, started)
            if wrapped is error:
                raise
            raise wrapped from error
        except Exception as error:
            wrapped = EvidenceUnitExtractionError("sentence-level extraction failed")
            self._emit_failed(candidate_evidence, wrapped, trace_context, started)
            raise wrapped from error
        self._emit_completed(units, canonical.chunk_id, trace_context, started)
        return units

    def extract_many(
        self,
        candidate_chunk_pairs: Iterable[tuple[CandidateChunkEvidence, object]],
        question_analysis: QuestionAnalysis,
        *,
        trace_context: TraceContext | None = None,
    ) -> tuple[EvidenceUnit, ...]:
        return tuple(
            unit
            for candidate, chunk in candidate_chunk_pairs
            for unit in self.extract_from_chunk(
                candidate, chunk, question_analysis, trace_context=trace_context
            )
        )

    @staticmethod
    def _validate_pair(candidate: CandidateChunkEvidence, chunk: CanonicalRetrievedChunk) -> None:
        if candidate.chunk_id != chunk.chunk_id:
            raise EvidenceUnitExtractionError("candidate and retrieved chunk IDs differ")
        if candidate.doc_title != chunk.doc_title:
            raise EvidenceUnitExtractionError("candidate and retrieved document titles differ")
        if candidate.paragraph_index != chunk.paragraph_index:
            raise EvidenceUnitExtractionError("candidate and retrieved paragraph indices differ")
        if candidate.paragraph_text != chunk.paragraph_text:
            raise EvidenceUnitExtractionError("candidate and retrieved paragraph text differ")
        if candidate.sentences != chunk.sentences:
            raise EvidenceUnitExtractionError("candidate and retrieved sentence metadata differ")
        if candidate.retrieval_rank != chunk.retrieval_rank:
            raise EvidenceUnitExtractionError("candidate and retrieved ranks differ")
        if candidate.retrieval_score != chunk.retrieval_score:
            raise EvidenceUnitExtractionError("candidate and retrieved scores differ")
        if candidate.source_question_id != chunk.source_question_id:
            raise EvidenceUnitExtractionError("candidate and retrieved source question IDs differ")

    def _extract(
        self,
        candidate: CandidateChunkEvidence,
        chunk: CanonicalRetrievedChunk,
        analysis: QuestionAnalysis,
    ) -> tuple[EvidenceUnit, ...]:
        historical = self._config.mode == "rule_based_v1"
        sentences = normalize_sentences(chunk, historical=historical)
        question_entities = tuple(item.text for item in analysis.seed_entities)
        context: list[str] = []
        units: list[EvidenceUnit] = []
        seen_ids: set[int] = set()
        for sentence in sentences:
            if sentence.sentence_id in seen_ids:
                raise EvidenceUnitExtractionError("duplicate sentence IDs within chunk")
            seen_ids.add(sentence.sentence_id)
            resolved = self._resolver.resolve(sentence.text, chunk.doc_title, tuple(context))
            context.append(sentence.text)
            if resolved.original_text != sentence.text:
                raise EvidenceUnitExtractionError("resolver changed original sentence text")
            entity_features: tuple[SentenceEntity, ...] = ()
            structured: tuple[StructuredAnswerCandidate, ...] = ()
            if historical:
                entity_values = historical_rules.entities(resolved.resolved_text, chunk.doc_title)
                relations = historical_rules.relation_hints(resolved.resolved_text)
                answer_types = tuple(
                    AnswerType(item)
                    for item in historical_rules.answer_types(
                        resolved.resolved_text, entity_values, relations
                    )
                )
                overlap, token_score = historical_rules.question_overlap(
                    analysis.normalized_question, question_entities, resolved.resolved_text
                )
            else:
                (entity_values, entity_features, relations, structured, overlap, token_score) = (
                    production_rules.features(sentence.text, chunk.doc_title, analysis)
                )
                answer_types = tuple(dict.fromkeys(item.answer_type for item in structured))
            fallbacks = tuple(
                item
                for item, enabled in (
                    ("sentence_metadata_missing", sentence.source != "metadata"),
                    ("title_pronoun_substitution", resolved.method == "title_pronoun"),
                    ("ambiguous_pronoun_unresolved", resolved.method == "ambiguous_context"),
                )
                if enabled
            )
            provider_versions: tuple[str, ...] = (
                f"segmenter:{self._config.segmenter_version}",
                f"resolver:{resolved.provider_version}",
                "historical-features-v1" if historical else "component02-v2-semantics",
            )
            if not historical:
                provider_versions += (
                    "sentence-overlap:literal-guard-v4",
                    "answer-candidates:sentence-span-v1",
                )
            metadata = EvidenceUnitMetadata(
                version=self._config.mode,
                configuration_fingerprint=self._config.fingerprint(),
                sentence_source=sentence.source,
                resolution=resolved,
                provider_versions=provider_versions,
                fallbacks_used=fallbacks,
            )
            units.append(
                EvidenceUnit(
                    evidence_unit_id=make_evidence_unit_id(chunk.chunk_id, sentence.sentence_id),
                    chunk_id=chunk.chunk_id,
                    doc_title=candidate.doc_title,
                    paragraph_index=candidate.paragraph_index,
                    sentence_id=sentence.sentence_id,
                    sentence_text=sentence.text,
                    resolved_text=resolved.resolved_text,
                    entities=entity_values,
                    entity_features=entity_features,
                    relation_hints=relations,
                    answer_type_candidates=answer_types,
                    structured_answer_candidates=structured,
                    question_entity_overlap=overlap,
                    question_token_overlap=token_score,
                    retrieval_rank=candidate.retrieval_rank,
                    retrieval_score=candidate.retrieval_score,
                    source_question_id=candidate.source_question_id,
                    start_char=sentence.start_char,
                    end_char=sentence.end_char,
                    metadata=metadata,
                )
            )
        return tuple(units)

    def _emit_completed(
        self,
        units: tuple[EvidenceUnit, ...],
        chunk_id: str,
        context: TraceContext | None,
        started: float,
    ) -> None:
        if self._sink is None or context is None:
            return
        self._sink.emit(
            InstrumentationEvent(
                context=context,
                event_type="epsa.evidence_units.completed",
                source="epsa.evidence_unit_extractor",
                source_version=self._config.mode,
                payload={
                    "chunk_id": chunk_id,
                    "units": [unit.model_dump(mode="json") for unit in units],
                    "unit_count": len(units),
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            )
        )

    def _emit_failed(
        self,
        candidate: object,
        error: EvidenceUnitExtractionError,
        context: TraceContext | None,
        started: float,
    ) -> None:
        if self._sink is None or context is None:
            return
        self._sink.emit(
            InstrumentationEvent(
                context=context,
                event_type="epsa.evidence_units.failed",
                source="epsa.evidence_unit_extractor",
                source_version=self._config.mode,
                payload={
                    "chunk_id": str(getattr(candidate, "chunk_id", "")),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            )
        )


class RuleBasedV2EvidenceUnitExtractor(RuleBasedEvidenceUnitExtractor):
    """Production successor with segmentation and conservative resolution."""

    def __init__(
        self,
        *,
        config: EvidenceUnitExtractorV2Config | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
        resolver: CoreferenceResolver | None = None,
    ) -> None:
        super().__init__(
            config=config or EvidenceUnitExtractorV2Config(),
            instrumentation_sink=instrumentation_sink,
            resolver=resolver,
        )
