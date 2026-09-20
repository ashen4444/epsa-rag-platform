"""Sentence identity, provenance, semantic features, and inference isolation."""

from __future__ import annotations

import pytest

from epsa_rag.core.exceptions import EvidenceUnitExtractionError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import QuestionInput
from epsa_rag.epsa.chunk_analysis import RuleBasedCandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.chunk_analysis.models import CanonicalRetrievedChunk
from epsa_rag.epsa.evidence_units import (
    EvidenceUnit,
    EvidenceUnitExtractorV2Config,
    RuleBasedEvidenceUnitExtractor,
    RuleBasedV2EvidenceUnitExtractor,
    historical_rules,
)
from epsa_rag.epsa.evidence_units.models import (
    ResolvedSentence,
    SentenceEntity,
    StructuredAnswerCandidate,
)
from epsa_rag.epsa.evidence_units.protocols import EvidenceUnitExtractorProtocol
from epsa_rag.epsa.evidence_units.resolution import (
    ConservativeTitlePronounResolver,
    TitlePronounResolver,
)
from epsa_rag.epsa.evidence_units.sentences import normalize_sentences, segment_sentences
from epsa_rag.epsa.question_analysis import RuleBasedQuestionAnalyzer
from epsa_rag.epsa.question_analysis.models import AnswerType
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.evaluation.components.evidence_units import evaluate_evidence_units
from epsa_rag.instrumentation import InMemoryInstrumentationSink, TraceContext


def _chunk(
    text: str,
    *,
    title: str = "Christopher Nolan",
    sentences: tuple[Sentence, ...] = (),
    chunk_id: str = "chunk-one",
) -> CanonicalRetrievedChunk:
    return CanonicalRetrievedChunk(
        chunk_id=chunk_id,
        doc_title=title,
        paragraph_index=2,
        paragraph_text=text,
        chunk_text=text,
        sentences=sentences,
        retrieval_rank=3,
        retrieval_score=0.75,
        source_question_id="question-one",
    )


def _run(
    chunk: CanonicalRetrievedChunk,
    question: str = "Where was the director of Inception born?",
    *,
    v2: bool = False,
) -> tuple[EvidenceUnit, ...]:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    extractor = RuleBasedV2EvidenceUnitExtractor() if v2 else RuleBasedEvidenceUnitExtractor()
    return extractor.extract_from_chunk(candidate, chunk, analysis)


def test_native_sentence_ids_and_provenance_are_stable() -> None:
    text = "Christopher Nolan directed Inception. He was born in London."
    chunk = _chunk(
        text,
        sentences=(
            Sentence(index=0, text="Christopher Nolan directed Inception."),
            Sentence(index=1, text=" He was born in London."),
        ),
    )
    units = _run(chunk)
    assert [unit.evidence_unit_id for unit in units] == ["chunk-one::s0", "chunk-one::s1"]
    assert units[1].sentence_text == "He was born in London."
    assert units[1].resolved_text == "Christopher Nolan was born in London."
    assert units[1].start_char == text.index("He was")
    assert units[1].retrieval_rank == 3
    assert units[1].retrieval_score == 0.75
    assert units[1].source_question_id == "question-one"
    assert units[1].metadata.sentence_source == "metadata"
    assert units[1].metadata.resolution.method == "title_pronoun"
    assert EvidenceUnit.model_validate_json(units[1].model_dump_json()) == units[1]


def test_v1_keeps_unsplit_fallback_and_historical_features() -> None:
    units = _run(_chunk("He directed Inception. He was born in London."))
    assert len(units) == 1
    unit = units[0]
    assert unit.evidence_unit_id == "chunk-one::s0"
    assert unit.metadata.sentence_source == "paragraph_fallback"
    assert {"directed", "born"}.issubset(unit.relation_hints)
    assert AnswerType.LOCATION in unit.answer_type_candidates
    assert "is_supporting_sentence" not in unit.model_dump()
    assert unit.structured_answer_candidates == ()


def test_v2_segments_without_metadata_and_preserves_source_spans() -> None:
    text = "Dr. Nolan directed Inception. He was born in London."
    units = _run(_chunk(text), v2=True)
    assert len(units) == 2
    assert [unit.sentence_id for unit in units] == [0, 1]
    assert all(unit.metadata.sentence_source == "segmented" for unit in units)
    assert [text[unit.start_char : unit.end_char] for unit in units] == [
        unit.sentence_text for unit in units
    ]
    assert units[1].metadata.resolution.method == "ambiguous_context"


def test_v2_does_not_force_ambiguous_pronoun_to_title() -> None:
    text = "Christopher Nolan worked with Leonardo DiCaprio. He later won an award."
    units = _run(
        _chunk(
            text,
            sentences=(
                Sentence(index=0, text="Christopher Nolan worked with Leonardo DiCaprio."),
                Sentence(index=1, text=" He later won an award."),
            ),
        ),
        v2=True,
    )
    assert units[1].sentence_text == " He later won an award."
    assert units[1].resolved_text == units[1].sentence_text
    assert units[1].metadata.resolution.method == "ambiguous_context"


def test_v2_exact_entity_overlap_and_lexical_token_equivalence() -> None:
    chunk = _chunk("New York was directed by Christopher Nolan.", title="New York")
    units = _run(chunk, question="Who was the director of York?", v2=True)
    assert len(units) == 1
    assert "York" not in units[0].question_entity_overlap
    assert units[0].question_token_overlap > 0.0
    assert "directed" in units[0].relation_hints


def test_v2_recovers_whole_literal_seed_in_long_entity_list() -> None:
    unit = _run(
        _chunk(
            "London Philharmonic Orchestra, London Symphony Orchestra, Philharmonia performed.",
            title="Avi Ostrowsky",
        ),
        question="Where did the London Philharmonic Orchestra perform?",
        v2=True,
    )[0]
    assert "London Philharmonic Orchestra" in unit.question_entity_overlap
    assert all(item.span_scope == "sentence_text" for item in unit.structured_answer_candidates)


def test_v2_literal_recovery_rejects_generic_words_and_partial_names() -> None:
    for sentence, question in (
        ("The casino opened yesterday.", "Which Casino opened yesterday?"),
        ("She attended high school locally.", "Which High School did she attend?"),
        ("New York City opened a new park.", "When did York open a park?"),
        ("New York City opened a new park.", "When did New York open a park?"),
        (
            "Royal London Philharmonic Orchestra performed yesterday.",
            "When did London Philharmonic Orchestra perform?",
        ),
    ):
        unit = _run(_chunk(sentence, title="Other Topic"), question=question, v2=True)[0]
        assert unit.question_entity_overlap == ()


def test_v2_literal_recovery_allows_person_honorific() -> None:
    unit = _run(
        _chunk("President Theodore Roosevelt spoke to the crowd.", title="Other Topic"),
        question="When did Theodore Roosevelt speak?",
        v2=True,
    )[0]
    assert "Theodore Roosevelt" in unit.question_entity_overlap


def test_v2_literal_recovery_tolerates_short_separator_variants() -> None:
    unit = _run(
        _chunk("The Norse�Gaels lived in the region.", title="Other Topic"),
        question="Where did the Norse-Gaels live?",
        v2=True,
    )[0]
    assert "Norse-Gaels" in unit.question_entity_overlap


def test_v2_literal_recovery_preserves_exact_punctuated_title() -> None:
    unit = _run(
        _chunk(
            "Evidently... John Cooper Clarke is a 2012 documentary.",
            title="Other Topic",
        ),
        question=(
            'What "Splitting Image" voice artist provided testimony in '
            "Evidently... John Cooper Clarke?"
        ),
        v2=True,
    )[0]
    assert "Evidently... John Cooper Clarke" in unit.question_entity_overlap


def test_v2_title_anchor_is_not_a_sentence_mention_or_answer_span() -> None:
    unit = _run(
        _chunk("It later changed names.", title="Operation Cold Comfort"),
        question="When did Operation Cold Comfort change names?",
        v2=True,
    )[0]
    assert unit.question_entity_overlap == ()
    assert unit.structured_answer_candidates == ()
    assert any(item.source == "doc_title" for item in unit.entity_features)


def test_v2_structured_year_candidates_are_source_grounded() -> None:
    unit = _run(_chunk("Inception was released in 2010.", title="Inception"), v2=True)[0]
    numeric = [
        candidate for candidate in unit.structured_answer_candidates if candidate.text == "2010"
    ]
    assert {item.answer_type for item in numeric} == {AnswerType.DATE, AnswerType.NUMBER}
    assert all(unit.sentence_text[item.start_char : item.end_char] == item.text for item in numeric)


def test_gold_fields_on_raw_input_are_not_emitted_or_used() -> None:
    raw = {
        "chunk_id": "chunk-gold",
        "doc_title": "Marie Curie",
        "paragraph_index": 0,
        "paragraph_text": "She discovered radium.",
        "sentences": [
            {"sentence_id": 7, "text": "She discovered radium.", "is_supporting_sentence": True}
        ],
        "supporting_sentence_ids": [7],
        "rank": 1,
        "score": 0.8,
    }
    analysis = RuleBasedQuestionAnalyzer().analyze("What did Marie Curie discover?")
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(raw, analysis)
    sink = InMemoryInstrumentationSink()
    units = RuleBasedEvidenceUnitExtractor(instrumentation_sink=sink).extract_from_chunk(
        candidate,
        raw,
        analysis,
        trace_context=TraceContext.start(run_id="test-run", question_id="question-one"),
    )
    assert units[0].sentence_id == 7
    assert units[0].evidence_unit_id == "chunk-gold::s7"
    assert "is_supporting_sentence" not in units[0].model_dump_json()
    assert "supporting_sentence_ids" not in sink.events[0].model_dump_json()


def test_mismatched_pair_fails_with_event() -> None:
    chunk = _chunk("He directed Inception.")
    analysis = RuleBasedQuestionAnalyzer().analyze("Who directed Inception?")
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    sink = InMemoryInstrumentationSink()
    with pytest.raises(EvidenceUnitExtractionError, match="IDs differ"):
        RuleBasedEvidenceUnitExtractor(instrumentation_sink=sink).extract_from_chunk(
            candidate,
            _chunk("He directed Inception.", chunk_id="other-chunk"),
            analysis,
            trace_context=TraceContext.start(run_id="test-run"),
        )
    assert sink.events[0].event_type == "epsa.evidence_units.failed"


def test_extract_many_flattens_in_retrieval_order() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze("Who directed Inception?")
    first = _chunk("Inception was directed by Christopher Nolan.")
    second = _chunk("He directed Inception.", chunk_id="chunk-two")
    analyzer = RuleBasedCandidateChunkEvidenceAnalyzer()
    pairs = [(analyzer.analyze(chunk, analysis), chunk) for chunk in (first, second)]
    units = RuleBasedEvidenceUnitExtractor().extract_many(pairs, analysis)
    assert [unit.chunk_id for unit in units] == ["chunk-one", "chunk-two"]


def test_v2_location_relation_requires_contextual_phrase() -> None:
    unit = _run(_chunk("Christopher Nolan appeared in Inception."), v2=True)[0]
    assert "located" not in unit.relation_hints


def test_inference_evaluator_runs_components_01_02_03() -> None:
    sentence = Sentence(index=0, text="Inception was directed by Christopher Nolan.")
    ranked = RankedParagraphChunk(
        chunk=ParagraphChunk(
            chunk_id="chunk-eval",
            title="Inception",
            paragraph_text=sentence.text,
            sentences=(sentence,),
        ),
        rank=1,
        score=0.8,
    )
    summary, traces, events = evaluate_evidence_units(
        (
            InferenceRetrieval(
                question=QuestionInput(question_id="question-eval", text="Who directed Inception?"),
                chunks=(ranked,),
                retriever_version="retriever-v1",
            ),
        ),
        run_id="unit-test-run",
        dataset_version="hotpotqa_1000_v1",
        dataset_manifest_sha256="0" * 64,
        corpus_version="hotpotqa_10000_v1",
        retrieval_run_id="retrieval-unit-test",
        git_commit_sha="0" * 40,
        git_dirty=True,
        source_sha256={},
        runtime={},
    )
    assert summary.completed_questions == 1
    assert summary.failed_questions == 0
    assert summary.diagnostics.evidence_units == 1
    assert traces[0].question_analysis is not None
    assert traces[0].chunk_evidence[0].chunk_id == "chunk-eval"
    assert traces[0].evidence_units[0].evidence_unit_id == "chunk-eval::s0"
    assert [event.event_type for event in events] == [
        "epsa.question_analysis.completed",
        "epsa.chunk_analysis.completed",
        "epsa.evidence_units.completed",
    ]
    assert "is_supporting_sentence" not in traces[0].model_dump_json()


def test_validation_rejects_candidate_provenance_mismatches() -> None:
    chunk = _chunk("He directed Inception.")
    analysis = RuleBasedQuestionAnalyzer().analyze("Who directed Inception?")
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    extractor = RuleBasedEvidenceUnitExtractor()
    for field, message, value in (
        ("doc_title", "titles differ", "Other"),
        ("paragraph_index", "indices differ", 99),
        ("paragraph_text", "text differ", "Other text."),
        ("sentences", "metadata differ", (Sentence(index=1, text="Other."),)),
        ("retrieval_rank", "ranks differ", 1),
        ("retrieval_score", "scores differ", 0.1),
        ("source_question_id", "source question IDs differ", "other-question"),
    ):
        changed = candidate.model_copy(update={field: value})
        with pytest.raises(EvidenceUnitExtractionError, match=message):
            extractor.extract_from_chunk(changed, chunk, analysis)


def test_duplicate_sentence_identity_is_rejected() -> None:
    text = "First. Second."
    chunk = _chunk(
        text,
        sentences=(
            Sentence(index=0, text="First."),
            Sentence(index=0, text=" Second."),
        ),
    )
    analysis = RuleBasedQuestionAnalyzer().analyze("Who directed Inception?")
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    with pytest.raises(EvidenceUnitExtractionError, match="duplicate sentence IDs"):
        RuleBasedEvidenceUnitExtractor().extract_from_chunk(candidate, chunk, analysis)


def test_resolvers_expose_no_substitution_and_conservative_ambiguity() -> None:
    historical = TitlePronounResolver()
    assert historical.resolve("A fact.", "Title", ()).method == "unchanged"
    assert historical.resolve("He directed it.", "", ()).resolved_text == "He directed it."
    production = ConservativeTitlePronounResolver()
    assert production.resolve("He won.", "Title", ("Someone Else spoke.",)).method == (
        "ambiguous_context"
    )
    assert production.resolve("He won.", "Title", ()).resolved_text == "Title won."
    assert production.resolve("It won.", "It", ()).method == "unchanged"


def test_v2_neutral_pronoun_can_use_explicit_recent_title_subject() -> None:
    production = ConservativeTitlePronounResolver()
    resolved = production.resolve(
        " It was later renamed Zombie.",
        "Operation Cold Comfort",
        ("During World War II, Operation Cold Comfort was a failed SAS raid.",),
    )
    assert resolved.resolved_text == "Operation Cold Comfort was later renamed Zombie."
    assert resolved.reason == "recent_title_subject"
    assert resolved.provider_version == "conservative-title-v2"
    assert production.resolve(
        " His father was a blacksmith.",
        "Erik Lundgren",
        ("Erik Lundgren was a racer who met Another Person.",),
    ).method == "ambiguous_context"


def test_segmenter_and_metadata_normalizer_cover_edge_cases() -> None:
    assert segment_sentences(" ") == ()
    assert [item.text for item in segment_sentences("Dr. A waited. Then B left.")] == [
        "Dr. A waited.",
        "Then B left.",
    ]
    ellipsis_sentences = segment_sentences("Evidently... John Cooper Clarke is a film.")
    assert [item.text for item in ellipsis_sentences] == [
        "Evidently... John Cooper Clarke is a film."
    ]
    chunk = _chunk(
        "Alpha. Beta.",
        sentences=(Sentence(index=0, text="Missing."),),
    )
    normalized = normalize_sentences(chunk, historical=False)
    assert normalized[0].start_char is None
    assert normalized[0].end_char is None


def test_contract_validators_protect_resolution_and_span_namespaces() -> None:
    with pytest.raises(ValueError, match="change flag"):
        ResolvedSentence(
            original_text="A",
            resolved_text="B",
            method="provider",
            changed=False,
            reason="test",
            provider_version="test-v1",
        )
    with pytest.raises(ValueError, match="title cannot"):
        SentenceEntity(
            text="Title", normalized="title", source="doc_title", start_char=0, end_char=5
        )
    with pytest.raises(ValueError, match="requires a nonempty"):
        StructuredAnswerCandidate(
            text="A",
            normalized="a",
            answer_type=AnswerType.ENTITY,
            source="test",
            span_scope="sentence_text",
            start_char=0,
            end_char=0,
            rule_score=0.5,
        )


def test_extractor_config_and_protocol_are_versioned() -> None:
    assert EvidenceUnitExtractorProtocol
    assert RuleBasedEvidenceUnitExtractor().config.mode == "rule_based_v1"
    with pytest.raises(ValueError, match="configured resolver"):
        RuleBasedV2EvidenceUnitExtractor(
            config=EvidenceUnitExtractorV2Config(resolver_version="other-v1")
        )
    with pytest.raises(ValueError, match="frozen title resolver"):
        RuleBasedEvidenceUnitExtractor(resolver=TitlePronounResolver())


def test_extractor_rejects_untyped_inputs_and_wraps_unexpected_resolver_failure() -> None:
    chunk = _chunk("He directed Inception.")
    analysis = RuleBasedQuestionAnalyzer().analyze("Who directed Inception?")
    candidate = RuleBasedCandidateChunkEvidenceAnalyzer().analyze(chunk, analysis)
    extractor = RuleBasedEvidenceUnitExtractor()
    with pytest.raises(EvidenceUnitExtractionError, match="Component 02 contract"):
        extractor.extract_from_chunk(object(), chunk, analysis)  # type: ignore[arg-type]
    with pytest.raises(EvidenceUnitExtractionError, match="Component 01 contract"):
        extractor.extract_from_chunk(candidate, chunk, object())  # type: ignore[arg-type]

    class BrokenResolver:
        version = "broken-v1"

        def resolve(self, sentence: str, document_title: str, context: tuple[str, ...]):
            del sentence, document_title, context
            raise RuntimeError("resolver unavailable")

    with pytest.raises(EvidenceUnitExtractionError, match="sentence-level extraction failed"):
        RuleBasedV2EvidenceUnitExtractor(
            config=EvidenceUnitExtractorV2Config(resolver_version="broken-v1"),
            resolver=BrokenResolver(),
        ).extract_from_chunk(candidate, chunk, analysis)


def test_evidence_unit_contract_rejects_identity_offsets_and_resolution_mismatches() -> None:
    unit = _run(_chunk("He directed Inception."))[0]
    payload = unit.model_dump(mode="json")
    payload["evidence_unit_id"] = "other::s0"
    with pytest.raises(ValueError, match="must derive"):
        EvidenceUnit.model_validate(payload)
    payload = unit.model_dump(mode="json")
    payload["start_char"] = 0
    payload["end_char"] = None
    with pytest.raises(ValueError, match="both be present"):
        EvidenceUnit.model_validate(payload)
    payload = unit.model_dump(mode="json")
    payload["end_char"] = payload["start_char"]
    with pytest.raises(ValueError, match="nonempty text"):
        EvidenceUnit.model_validate(payload)
    payload = unit.model_dump(mode="json")
    payload["metadata"]["resolution"]["original_text"] = "different"
    with pytest.raises(ValueError, match="original sentence"):
        EvidenceUnit.model_validate(payload)
    payload = unit.model_dump(mode="json")
    payload["retrieval_score"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        EvidenceUnit.model_validate(payload)


def test_historical_feature_rules_cover_all_answer_type_signals() -> None:
    values = historical_rules.entities(
        '"The Film" was released by Example University in 2010.', "Jane Doe"
    )
    relations = historical_rules.relation_hints("The Film was released.")
    answer_types = historical_rules.answer_types(
        '"The Film" was released by Example University in 2010.', values, relations
    )
    assert {"DATE", "NUMBER", "ORGANIZATION", "TITLE_OR_WORK", "ENTITY"} <= set(
        answer_types
    )
    assert "PERSON" in historical_rules.answer_types(
        "Jane Doe arrived.", ("Jane Doe",), ()
    )
    assert historical_rules.question_overlap("", (), "Anything") == ((), 0.0)


def test_segmenter_handles_non_period_endings_quotes_and_empty_chunk() -> None:
    assert [item.text for item in segment_sentences('Who left? "Bob did!"')] == [
        "Who left?",
        '"Bob did!"',
    ]
    empty = _chunk("", sentences=(), chunk_id="empty-chunk")
    assert normalize_sentences(empty, historical=False) == ()
