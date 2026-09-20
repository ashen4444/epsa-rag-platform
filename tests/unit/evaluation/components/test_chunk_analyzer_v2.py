"""Versioned Component 02 evaluation keeps the inference and export contracts."""

from __future__ import annotations

from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import QuestionInput
from epsa_rag.epsa.chunk_analysis import RuleBasedV2ChunkAnalyzerConfig
from epsa_rag.evaluation.components.chunk_analyzer import (
    ChunkAnalyzerEvaluationConfig,
    ChunkAnalyzerRunSummary,
    InferenceRetrieval,
    _v2_span_failures,
    evaluate_chunks,
)


def _ranked(identifier: str, title: str, text: str, rank: int) -> RankedParagraphChunk:
    return RankedParagraphChunk(
        chunk=ParagraphChunk(
            chunk_id=identifier,
            title=title,
            paragraph_text=text,
            sentences=(Sentence(index=0, text=text),),
        ),
        rank=rank,
        score=1 / rank,
    )


def test_v2_evaluator_uses_one_retrieved_set_and_records_distinct_mode() -> None:
    inputs = (
        InferenceRetrieval(
            question=QuestionInput(
                question_id="q-1", text="Where was the director of Inception born?"
            ),
            chunks=(
                _ranked(
                    "chunk:inception", "Inception",
                    "Inception was directed by Christopher Nolan.", 1,
                ),
                _ranked(
                    "chunk:nolan", "Christopher Nolan",
                    "Christopher Nolan was born in London.", 2,
                ),
            ),
            retriever_version="hybrid-retriever-v1",
        ),
    )
    config = ChunkAnalyzerEvaluationConfig(analyzer=RuleBasedV2ChunkAnalyzerConfig())
    summary, traces, events = evaluate_chunks(
        inputs,
        run_id="chunk-analyzer-v2-unit",
        dataset_version="hotpotqa_1000_v1",
        dataset_manifest_sha256="a" * 64,
        corpus_version="hotpotqa_10000_v1",
        retrieval_run_id="retrieval-unit",
        git_commit_sha="b" * 40,
        git_dirty=True,
        source_sha256={"src/example.py": "c" * 64},
        runtime={"python": "3.12.10"},
        config=config,
    )

    assert summary.configuration.analyzer.mode == "rule_based_v2"
    assert ChunkAnalyzerRunSummary.model_validate_json(summary.model_dump_json()) == summary
    assert summary.completed_questions == 1
    assert summary.diagnostics.analyzed_chunks == 2
    assert summary.diagnostics.span_integrity_failures == 0
    assert summary.diagnostics.analyzer_fingerprint_mismatches == 0
    assert summary.diagnostics.bridge_questions == 1
    assert summary.diagnostics.bridge_questions_with_candidates == 1
    assert summary.diagnostics.relation_grounding_type_counts["entity_pair"] >= 1
    assert set(summary.diagnostics.answer_candidate_type_counts) == {
        "PERSON", "LOCATION", "ORGANIZATION", "TITLE_OR_WORK", "DATE", "NUMBER", "ENTITY"
    }
    assert "Christopher Nolan" in {
        item.text for item in traces[0].evidence[0].potential_bridge_entities
    }
    assert [event.event_type for event in events] == [
        "epsa.question_analysis.completed",
        "epsa.chunk_analysis.completed",
        "epsa.chunk_analysis.completed",
    ]
    assert all(event.source_version == "rule_based_v2" for event in events[1:])
    assert "gold" not in traces[0].model_dump_json().lower()
    damaged = traces[0].evidence[0].model_copy(update={"paragraph_text": "Different text."})
    assert _v2_span_failures(damaged) > 0
