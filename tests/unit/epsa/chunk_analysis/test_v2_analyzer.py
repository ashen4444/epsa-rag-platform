"""Contextual Component 02 behavior, separate from historical v1 replay."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from epsa_rag.core.exceptions import ChunkAnalysisError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.epsa.chunk_analysis import (
    CandidateChunkEvidence,
    RuleBasedV2CandidateChunkEvidenceAnalyzer,
    RuleBasedV2ChunkAnalyzerConfig,
)
from epsa_rag.epsa.chunk_analysis import v2_analyzer as v2_module
from epsa_rag.epsa.chunk_analysis.models import (
    CanonicalRetrievedChunk,
    ChunkAnalysisMetadataV2,
    ChunkAnswerCandidateV2,
    ChunkEntityMentionV2,
    GroundedChunkRelationHint,
)
from epsa_rag.epsa.question_analysis import RuleBasedQuestionAnalyzer
from epsa_rag.epsa.question_analysis.models import AnswerType
from epsa_rag.instrumentation import InMemoryInstrumentationSink, TraceContext


def _ranked(identifier: str, title: str, paragraph: str, rank: int) -> RankedParagraphChunk:
    return RankedParagraphChunk(
        chunk=ParagraphChunk(
            chunk_id=identifier,
            title=title,
            paragraph_text=paragraph,
            sentences=(Sentence(index=0, text=paragraph),),
        ),
        rank=rank,
        score=1 / rank,
    )


def test_v2_bridge_requires_seed_side_relation_and_other_retrieved_chunk() -> None:
    question = RuleBasedQuestionAnalyzer().analyze(
        "Where was the director of Inception born?"
    )
    chunks = (
        _ranked(
            "chunk:inception", "Inception",
            "Inception was directed by Christopher Nolan.", 1,
        ),
        _ranked(
            "chunk:nolan", "Christopher Nolan",
            "Christopher Nolan directed Inception. Christopher Nolan was born in London.", 2,
        ),
        _ranked("chunk:london", "London", "London is a city in England.", 3),
    )
    sink = InMemoryInstrumentationSink()
    analyzer = RuleBasedV2CandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    evidence = analyzer.analyze_batch(
        chunks, question,
        trace_context=TraceContext.start(run_id="v2-test", question_id="q-1"),
    )

    first = evidence[0]
    assert first.is_title_match
    assert "Christopher Nolan" in {item.text for item in first.potential_bridge_entities}
    assert isinstance(first.metadata, ChunkAnalysisMetadataV2)
    accepted = [item for item in first.metadata.bridge_decisions if item.accepted]
    assert [(item.entity, item.relation, item.linked_chunk_ids) for item in accepted] == [
        ("Christopher Nolan", "directed", ("chunk:nolan",))
    ]
    assert "London" not in {item.text for item in evidence[1].potential_bridge_entities}
    assert "London" in {
        item.entity for item in evidence[1].metadata.bridge_decisions
        if item.reason == "answer_role"
    }
    assert [event.event_type for event in sink.events] == [
        "epsa.chunk_analysis.completed"
    ] * 3
    assert {event.source_version for event in sink.events} == {"rule_based_v2"}


def test_v2_rejects_partial_title_identity_and_bare_in_location_hint() -> None:
    analyzer = RuleBasedV2CandidateChunkEvidenceAnalyzer()
    york = analyzer.analyze(
        {"chunk_id": "chunk:new-york", "title": "New York",
         "paragraph_text": "New York was founded in 1624."},
        RuleBasedQuestionAnalyzer().analyze("Where was York founded?"),
    )
    inception = analyzer.analyze(
        {"chunk_id": "chunk:actor", "title": "Actor",
         "paragraph_text": "The actor appeared in Inception."},
        RuleBasedQuestionAnalyzer().analyze("Which actor appeared in Inception?"),
    )

    assert not york.is_title_match
    assert "New York" not in york.question_entity_overlap
    assert "located" not in {hint.relation for hint in inception.relation_hints}


def test_v2_single_chunk_does_not_claim_cross_chunk_connectivity() -> None:
    analyzer = RuleBasedV2CandidateChunkEvidenceAnalyzer()
    evidence = analyzer.analyze(
        {"chunk_id": "chunk:inception", "title": "Inception",
         "paragraph_text": "Inception was directed by Christopher Nolan."},
        RuleBasedQuestionAnalyzer().analyze("Where was the director of Inception born?"),
    )

    assert evidence.potential_bridge_entities == ()
    assert "no_retrieved_context" in evidence.metadata.fallbacks_used
    assert "no_context" in {item.reason for item in evidence.metadata.bridge_decisions}


def test_v2_quote_boundaries_and_relation_offsets_match_source_text() -> None:
    paragraph = (
        "The Republic's Army described a long campaign. "
        "Inception was directed by Christopher Nolan. "
        "He was born  in London and wrote 'Brief Notes'."
    )
    analyzer = RuleBasedV2CandidateChunkEvidenceAnalyzer()
    evidence = analyzer.analyze(
        {"chunk_id": "chunk:quotes", "title": "Inception", "paragraph_text": paragraph},
        RuleBasedQuestionAnalyzer().analyze("Where was the director of Inception born?"),
    )
    source = paragraph

    assert all(len(item.text) <= RuleBasedV2ChunkAnalyzerConfig().maximum_entity_length
               for item in evidence.entities)
    assert all(
        source[hint.start_char:hint.end_char].casefold() == hint.matched_text.casefold()
        for hint in evidence.relation_hints
    )
    assert any(hint.subject_entity == "Christopher Nolan" for hint in evidence.relation_hints)
    assert "located" not in {hint.relation for hint in evidence.relation_hints}
    assert CandidateChunkEvidence.model_validate_json(evidence.model_dump_json()) == evidence


def test_v2_separates_relation_and_connectivity_rejection_reasons() -> None:
    question = RuleBasedQuestionAnalyzer().analyze(
        "Where was the director of Inception born?"
    )
    analyzer = RuleBasedV2CandidateChunkEvidenceAnalyzer()
    inception = _ranked(
        "chunk:inception", "Inception",
        "Inception was directed by Christopher Nolan.", 1,
    )
    nolan = _ranked(
        "chunk:nolan", "Christopher Nolan",
        "Christopher Nolan was born in London.", 2,
    )
    without_link = analyzer.analyze(
        inception, question, retrieved_chunks=(inception,)
    )
    without_relation = analyzer.analyze(
        {"chunk_id": "chunk:feature", "title": "Inception",
         "paragraph_text": "Inception features Christopher Nolan."},
        question,
        retrieved_chunks=(
            {"chunk_id": "chunk:feature", "title": "Inception",
             "paragraph_text": "Inception features Christopher Nolan."},
            nolan,
        ),
    )
    unrelated = analyzer.analyze_batch(
        (
            _ranked("chunk:other", "Another Film",
                    "Another Film was directed by Christopher Nolan.", 1),
            nolan,
        ),
        question,
    )[0]

    assert without_link.potential_bridge_entities == ()
    assert "no_cross_chunk_link" in {
        item.reason for item in without_link.metadata.bridge_decisions
    }
    assert without_relation.potential_bridge_entities == ()
    assert "no_required_relation" in {
        item.reason for item in without_relation.metadata.bridge_decisions
    }
    assert unrelated.potential_bridge_entities == ()
    assert "not_seed_side" in {
        item.reason for item in unrelated.metadata.bridge_decisions
    }


def test_v2_failed_single_and_batch_inputs_emit_structured_failure_events() -> None:
    sink = InMemoryInstrumentationSink()
    analyzer = RuleBasedV2CandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    context = TraceContext.start(run_id="v2-failure", question_id="q-1")
    invalid = {"chunk_id": "chunk:empty", "title": "", "paragraph_text": ""}

    with pytest.raises(ChunkAnalysisError):
        analyzer.analyze(invalid, trace_context=context)
    with pytest.raises(ChunkAnalysisError):
        analyzer.analyze_batch((invalid,), None, trace_context=context)

    assert [item.event_type for item in sink.events] == [
        "epsa.chunk_analysis.failed", "epsa.chunk_analysis.failed"
    ]
    assert {item.source_version for item in sink.events} == {"rule_based_v2"}


def test_v2_keeps_date_and_number_as_distinct_deterministic_candidates() -> None:
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze(
        {"chunk_id": "chunk:year", "title": "Inception",
         "paragraph_text": "Inception was released in 2010 and directed by Christopher Nolan."},
        RuleBasedQuestionAnalyzer().analyze("When was Inception released?"),
    )
    pairs = {(item.answer_type.value, item.text) for item in evidence.answer_type_candidates}

    assert ("DATE", "2010") in pairs
    assert ("NUMBER", "2010") in pairs
    assert "Christopher Nolan" in {item.text for item in evidence.entities}


def test_v2_relation_pairs_do_not_use_intervening_nationality_or_previous_actor() -> None:
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze(
        {
            "chunk_id": "chunk:streak",
            "title": "Streak",
            "paragraph_text": (
                "Streak is a 2008 American film directed by Demi Moore, "
                "written by Kelly Fremon, and starring Brittany Snow."
            ),
        }
    )
    pairs = {
        (hint.relation, hint.subject_entity, hint.object_entity)
        for hint in evidence.relation_hints
    }

    assert ("directed", "Demi Moore", "Streak") in pairs
    assert ("written", "Kelly Fremon", "Streak") in pairs
    assert ("starring", "Brittany Snow", "Streak") in pairs
    assert all("American" not in pair for pair in pairs)


def test_v2_acronym_title_link_can_support_seed_side_candidate() -> None:
    question = RuleBasedQuestionAnalyzer().analyze(
        "Who founded the unit that conducted Operation Cold Comfort?"
    )
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze_batch(
        (
            _ranked("chunk:operation", "Operation Cold Comfort",
                    "Operation Cold Comfort was a raid by SAS.", 1),
            _ranked("chunk:unit", "Special Air Service",
                    "The Special Air Service was founded by David Stirling.", 2),
        ),
        question,
    )[0]
    accepted = [decision for decision in evidence.metadata.bridge_decisions if decision.accepted]

    assert [(decision.entity, decision.relation, decision.link_basis,
             decision.matches_question_relation) for decision in accepted] == [
        ("SAS", "associated", "title_alias", False)
    ]
    assert accepted[0].linked_chunk_ids == ("chunk:unit",)


def test_v2_acronym_noun_phrase_matches_development_question_wording() -> None:
    question = RuleBasedQuestionAnalyzer().analyze(
        "Operation Cold Comfort was a failed raid by a special forces unit founded in what year?"
    )
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze_batch(
        (
            _ranked("chunk:operation", "Operation Cold Comfort",
                    "During World War II, Operation Cold Comfort was a failed SAS raid.", 1),
            _ranked("chunk:unit", "Special Air Service",
                    "The Special Air Service was founded in 1941.", 2),
        ),
        question,
    )[0]

    assert [(item.entity, item.relation, item.link_basis) for item in
            evidence.metadata.bridge_decisions if item.accepted] == [
        ("SAS", "associated", "title_alias")
    ]


def test_v2_single_token_nationality_does_not_link_by_body_mention() -> None:
    question = RuleBasedQuestionAnalyzer().analyze(
        "Who directed the film Streak?"
    )
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze_batch(
        (
            _ranked("chunk:streak", "Streak",
                    "Streak is an American film directed by Demi Moore.", 1),
            _ranked("chunk:other", "Another Film",
                    "Another Film is an American film with Mexican actors.", 2),
        ),
        question,
    )[0]

    assert "American" not in {item.text for item in evidence.potential_bridge_entities}
    assert "no_cross_chunk_link" in {
        decision.reason for decision in evidence.metadata.bridge_decisions
        if decision.entity == "American"
    }


def test_v2_config_has_distinct_fingerprint_without_changing_v1_defaults() -> None:
    from epsa_rag.epsa.chunk_analysis import ChunkAnalyzerConfig

    v1 = ChunkAnalyzerConfig()
    v2 = RuleBasedV2ChunkAnalyzerConfig()
    assert v1.mode == "rule_based_v1"
    assert v2.mode == "rule_based_v2"
    assert v1.fingerprint() != v2.fingerprint()


def test_v2_unexpected_adapter_failures_are_wrapped_and_instrumented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_adapter(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("adapter bug")

    monkeypatch.setattr(v2_module, "adapt_retrieved_chunk", fail_adapter)
    sink = InMemoryInstrumentationSink()
    analyzer = RuleBasedV2CandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    context = TraceContext.start(run_id="v2-adapter-failure", question_id="q-1")
    raw = {"chunk_id": "chunk:bad", "title": "Bad", "paragraph_text": "Bad data."}

    with pytest.raises(ChunkAnalysisError, match="contextual chunk analysis failed"):
        analyzer.analyze(raw, trace_context=context)
    with pytest.raises(ChunkAnalysisError, match="invalid retrieved chunk in contextual batch"):
        analyzer.analyze_batch((raw,), None, trace_context=context)
    assert [event.payload["chunk_id"] for event in sink.events] == [
        "chunk:bad", "chunk:bad"
    ]


def test_v2_batch_analysis_failure_keeps_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = InMemoryInstrumentationSink()
    analyzer = RuleBasedV2CandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    context = TraceContext.start(run_id="v2-analysis-failure", question_id="q-1")
    empty = CanonicalRetrievedChunk(
        chunk_id="chunk:empty", doc_title="Empty", paragraph_text=" ", chunk_text="Empty"
    )
    with pytest.raises(ChunkAnalysisError, match="requires paragraph text"):
        analyzer.analyze_batch((empty,), None, trace_context=context)

    def fail_analysis(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("rule bug")

    monkeypatch.setattr(analyzer, "_analyze_canonical", fail_analysis)
    with pytest.raises(ChunkAnalysisError, match="contextual chunk analysis failed"):
        analyzer.analyze_batch((empty,), None, trace_context=context)
    assert [event.payload["error_type"] for event in sink.events] == [
        "ChunkAnalysisError", "ChunkAnalysisError"
    ]


def test_v2_config_can_disable_body_mention_connectivity() -> None:
    question = RuleBasedQuestionAnalyzer().analyze(
        "Where was the director of Inception born?"
    )
    analyzer = RuleBasedV2CandidateChunkEvidenceAnalyzer(
        config=RuleBasedV2ChunkAnalyzerConfig(allow_body_mention_links=False)
    )
    evidence = analyzer.analyze_batch(
        (
            _ranked("chunk:inception", "Inception",
                    "Inception was directed by Christopher Nolan.", 1),
            _ranked("chunk:other", "Another Film",
                    "Another Film was directed by Christopher Nolan.", 2),
        ),
        question,
    )[0]

    assert analyzer.config.allow_body_mention_links is False
    assert evidence.potential_bridge_entities == ()
    assert "no_cross_chunk_link" in {
        item.reason for item in evidence.metadata.bridge_decisions
    }


def test_v2_title_backed_sentence_window_is_marked_as_weaker_support() -> None:
    question = RuleBasedQuestionAnalyzer().analyze(
        "Where was the company that released Inception founded?"
    )
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze_batch(
        (
            _ranked("chunk:film", "Inception",
                    "Inception was released in July with Warner Bros.", 1),
            _ranked("chunk:other", "Other Film", "Other Film credits Warner Bros.", 2),
            _ranked("chunk:studio", "Warner Bros", "Warner Bros is a film studio.", 3),
        ),
        question,
    )[0]
    accepted = [item for item in evidence.metadata.bridge_decisions if item.accepted]

    assert [(item.entity, item.relation, item.relation_basis, item.link_basis) for
            item in accepted] == [
        ("Warner Bros", "released", "sentence_window", "title")
    ]
    assert accepted[0].matches_question_relation
    assert accepted[0].linked_chunk_ids == ("chunk:other", "chunk:studio")


def test_v2_grounded_starred_birth_and_location_pairs() -> None:
    paragraph = (
        "Streak starred Demi Moore. "
        "Demi Moore was born in New Mexico. "
        "Warner Bros is located in Burbank."
    )
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze(
        {"chunk_id": "chunk:relations", "title": "Streak", "paragraph_text": paragraph}
    )
    pairs = {
        (hint.relation, hint.subject_entity, hint.object_entity)
        for hint in evidence.relation_hints
    }
    source = paragraph

    assert ("starring", "Streak", "Demi Moore") in pairs
    assert ("born", "Demi Moore", "New Mexico") in pairs
    assert ("located", "Warner Bros", "Burbank") in pairs
    assert all(source[hint.start_char:hint.end_char] == hint.matched_text
               for hint in evidence.relation_hints)


def test_v2_answer_types_and_all_serialized_spans_use_explicit_scope() -> None:
    paragraph = (
        "Inception is a film directed by Christopher Nolan. "
        "Christopher Nolan was born in London in 1970. "
        "Warner Bros is a company. Cedar appears."
    )
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze(
        {"chunk_id": "chunk:typed", "title": "Inception", "paragraph_text": paragraph}
    )
    types = {(item.text, item.answer_type.value) for item in evidence.answer_type_candidates}

    assert ("Inception", "TITLE_OR_WORK") in types
    assert ("Christopher Nolan", "PERSON") in types
    assert ("London", "LOCATION") in types
    assert ("Warner Bros", "ORGANIZATION") in types
    assert ("1970", "DATE") in types
    assert ("1970", "NUMBER") in types
    assert ("Cedar", "ENTITY") in types
    for entity in evidence.entities:
        if entity.span_scope == "doc_title":
            assert entity.start_char is None
            assert entity.end_char is None
        else:
            assert paragraph[entity.start_char:entity.end_char] == entity.text
    for relation in evidence.relation_hints:
        assert relation.span_scope == "paragraph_text"
        assert paragraph[relation.start_char:relation.end_char] == relation.matched_text
        assert relation.grounding_type in {
            "entity_pair", "subject_only", "object_only", "lexical_hint"
        }
    for answer in evidence.answer_type_candidates:
        if answer.span_scope == "doc_title":
            assert answer.start_char is None
            assert answer.end_char is None
            assert answer.text == evidence.doc_title
        else:
            assert paragraph[answer.start_char:answer.end_char] == answer.text
    serialized = evidence.model_dump_json()
    assert '"span_scope":"paragraph_text"' in serialized
    assert '"grounding_type":' in serialized
    assert CandidateChunkEvidence.model_validate_json(serialized) == evidence


def test_v2_relation_grounding_distinguishes_pair_and_lexical_hint() -> None:
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze(
        {"chunk_id": "chunk:grounding", "title": "Inception",
         "paragraph_text": "Inception was directed by Christopher Nolan. The genre is unclear."}
    )
    kinds = {(item.relation, item.grounding_type) for item in evidence.relation_hints}

    assert ("directed", "entity_pair") in kinds
    assert ("genre", "lexical_hint") in kinds


@pytest.mark.parametrize(
    ("question", "paragraph", "canonical"),
    [
        ("Who was the director of Inception?",
         "Inception was directed by Christopher Nolan.", "directed"),
        ("Who was the writer of Inception?",
         "Inception was written by Christopher Nolan.", "written"),
        ("What was Christopher Nolan's birthplace?",
         "Christopher Nolan was born in London.", "born"),
    ],
)
def test_v2_relation_words_share_controlled_canonical_names(
    question: str, paragraph: str, canonical: str
) -> None:
    analyzed_question = RuleBasedQuestionAnalyzer().analyze(question)
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze(
        {"chunk_id": "chunk:relation", "title": "Inception", "paragraph_text": paragraph},
        analyzed_question,
    )

    assert canonical in {hint.relation for hint in analyzed_question.required_relation_hints}
    assert canonical in {hint.relation for hint in evidence.relation_hints}


def test_v2_span_models_reject_title_offsets_and_missing_body_offsets() -> None:
    with pytest.raises(ValidationError, match="title entity cannot claim"):
        ChunkEntityMentionV2(
            text="Inception", normalized="inception", source="doc_title",
            start_char=0, end_char=9, span_scope="doc_title", confidence=1.0,
        )
    with pytest.raises(ValidationError, match="body answer requires"):
        ChunkAnswerCandidateV2(
            answer_type=AnswerType.ENTITY, text="Unknown", source="chunk_entity_like_span",
            span_scope="paragraph_text", confidence=0.55,
        )
    with pytest.raises(ValidationError, match="grounding type must match"):
        GroundedChunkRelationHint(
            relation="directed", matched_text="directed", start_char=0, end_char=8,
            span_scope="paragraph_text", subject_entity="Christopher Nolan",
            object_entity="Inception", grounding_type="lexical_hint", confidence=0.8,
        )


def test_v2_generic_type_words_and_title_fragments_stay_untyped() -> None:
    evidence = RuleBasedV2CandidateChunkEvidenceAnalyzer().analyze(
        {"chunk_id": "chunk:book", "title": "The Ghost Map",
         "paragraph_text": (
             "The Company appeared in the story. "
             "The Ghost Map: The Story of London and the Modern World is a book."
         )}
    )
    classified = {(item.text, item.answer_type) for item in evidence.answer_type_candidates}

    assert ("Company", AnswerType.ORGANIZATION) not in classified
    assert ("Modern World", AnswerType.TITLE_OR_WORK) not in classified
