"""Tests for permanent paired system evaluation contracts and execution."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from epsa_rag.core.exceptions import FrozenArtifactError, SourceValidationError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, RetrievalQuery, Sentence
from epsa_rag.data.models import (
    BenchmarkExample,
    EvaluationLabels,
    QuestionInput,
    SupportingFactLabel,
)
from epsa_rag.evaluation.system.metrics import (
    answer_exact_match,
    answer_token_f1,
    normalize_answer,
    percentile,
)
from epsa_rag.evaluation.system.models import (
    SystemCondition,
    SystemEvaluationConfig,
    SystemQuestionTrace,
    SystemRunMetadata,
    SystemVariant,
)
from epsa_rag.evaluation.system.runner import evaluate_systems
from epsa_rag.evaluation.system.storage import (
    ResumableSystemRunStore,
    load_system_export,
)
from epsa_rag.evaluation.system.token_accounting import TikTokenCounter
from epsa_rag.pipeline import (
    AnswerGenerationRequest,
    FinalAnswer,
    FinalAnswerConfig,
    FixedBaselineTrace,
    LLMTokenUsage,
    merge_retrieval_hops,
    render_full_paragraph_context,
)
from epsa_rag.retrieval.config import HybridRetrieverConfig, RRFConfig
from epsa_rag.retrieval.models import RetrievalResult


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        inference=QuestionInput(question_id="q1", text="Where was the director born?"),
        evaluation=EvaluationLabels(
            answer="London",
            question_type="bridge",
            difficulty="hard",
            supporting_facts=(
                SupportingFactLabel(
                    chunk_id="c1",
                    title="Director",
                    sentence_index=0,
                    evidence_unit_id="c1::s0",
                ),
            ),
        ),
    )


def _fixed_trace() -> FixedBaselineTrace:
    text = "The director was born in London."
    ranked = RankedParagraphChunk(
        chunk=ParagraphChunk(
            chunk_id="c1",
            title="Director",
            paragraph_text=text,
            sentences=(Sentence(index=0, text=text),),
        ),
        rank=1,
        score=1.0,
    )
    retrieval = RetrievalResult(
        query=RetrievalQuery(text="Where was the director born?", question_id="q1"),
        retriever_version="hybrid-retriever-v2",
        results=(ranked,),
    )
    merged = merge_retrieval_hops(retrieval)
    request = AnswerGenerationRequest(
        question_id="q1",
        question=retrieval.query.text,
        context=render_full_paragraph_context(merged.results),
    )
    config = FinalAnswerConfig()
    answer = FinalAnswer(
        answer="London",
        response_id="response-1",
        model=config.model,
        generator_version=config.generator_version,
        prompt_version=config.prompt_version,
        configuration_fingerprint=config.fingerprint(),
        usage=LLMTokenUsage(input_tokens=12, output_tokens=1, total_tokens=13),
        latency_ms=1,
    )
    return FixedBaselineTrace(
        question_id="q1",
        top_k=1,
        hop1=retrieval,
        merged_retrieval=merged,
        final_answer_request=request,
        final_answer=answer,
        latency_ms=2,
    )


def _config() -> SystemEvaluationConfig:
    return SystemEvaluationConfig(
        systems=(SystemVariant.FIXED,),
        top_ks=(1,),
        retriever=HybridRetrieverConfig(fusion=RRFConfig(result_k=1)),
    )


def _metadata() -> SystemRunMetadata:
    config = _config()
    return SystemRunMetadata(
        run_id="system-test",
        git_commit_sha="a" * 40,
        git_dirty=False,
        source_sha256={},
        runtime={"python": "3.12"},
        dataset_version="dataset-v1",
        corpus_version="corpus-v1",
        dataset_manifest_sha256="1" * 64,
        dataset_file_sha256="2" * 64,
        corpus_manifest_sha256="3" * 64,
        corpus_file_sha256="4" * 64,
        index_manifests=(),
        question_ids=("q1",),
        full_dataset_question_count=1,
        configuration=config,
        configuration_fingerprint=config.fingerprint(),
    )


class _Counter:
    def count(self, text: str) -> int:
        return len(text.split())


class _Pipeline:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, question_id: str, question: str) -> FixedBaselineTrace:
        self.calls += 1
        assert (question_id, question) == ("q1", "Where was the director born?")
        return _fixed_trace()

    def take_query_embedding_observations(self):  # type: ignore[no-untyped-def]
        return ()


class _FailingPipeline(_Pipeline):
    def run(self, *, question_id: str, question: str) -> FixedBaselineTrace:
        raise RuntimeError("simulated generation failure")


def test_hotpotqa_answer_metrics_and_percentile() -> None:
    assert normalize_answer("The London!") == "london"
    assert answer_exact_match("London", "the london") == 1
    assert answer_token_f1("London, England", "London") == pytest.approx(2 / 3)
    assert answer_token_f1("", "") == 1
    assert answer_token_f1("Paris", "London") == 0
    assert percentile((1, 2, 10), 0.5) == 2
    assert percentile((), 0.95) == 0


def test_system_config_builds_the_complete_default_design() -> None:
    config = SystemEvaluationConfig(
        retriever=HybridRetrieverConfig(fusion=RRFConfig(result_k=30))
    )

    assert len(config.conditions()) == 12
    assert config.conditions()[0].key == "fixed-top-5"
    with pytest.raises(ValidationError, match="sorted, unique"):
        SystemEvaluationConfig(
            top_ks=(10, 5),
            retriever=HybridRetrieverConfig(fusion=RRFConfig(result_k=30)),
        )
    with pytest.raises(ValidationError, match="cover every"):
        SystemEvaluationConfig(
            retriever=HybridRetrieverConfig(fusion=RRFConfig(result_k=10))
        )


def test_runner_persists_metrics_and_export_integrity(tmp_path: Path) -> None:
    metadata = _metadata()
    store = ResumableSystemRunStore(tmp_path, metadata)
    pipeline = _Pipeline()

    summary = evaluate_systems(
        examples=(_example(),),
        metadata=metadata,
        store=store,
        pipeline_factory=lambda _: pipeline,
        token_counter=_Counter(),
    )

    assert summary.status == "completed"
    assert summary.full_benchmark
    assert summary.condition_summaries[0].metrics["exact_match"] == 1
    assert summary.condition_summaries[0].metrics["all_gold_available_hop1_rate"] == 1
    loaded_summary, traces = load_system_export(store.directory)
    assert loaded_summary == summary
    assert traces[0].retrieval_quality is not None
    assert traces[0].retrieval_quality.all_gold_available_final
    assert pipeline.calls == 1


def test_store_resumes_existing_question_without_repeating_pipeline(tmp_path: Path) -> None:
    metadata = _metadata()
    store = ResumableSystemRunStore(tmp_path, metadata)
    pipeline = _Pipeline()
    condition = SystemCondition(system=SystemVariant.FIXED, top_k=1)
    trace = SystemQuestionTrace(
        condition=condition,
        question=_example().inference,
        evaluation_labels=_example().evaluation,
        status="failed",
        error_type="InterruptedDependency",
        error_message="preserved diagnostic",
    )
    store.write_trace(trace)

    summary = evaluate_systems(
        examples=(_example(),),
        metadata=metadata,
        store=store,
        pipeline_factory=lambda _: pipeline,
        token_counter=_Counter(),
    )

    assert pipeline.calls == 0
    assert summary.status == "completed_with_failures"
    assert summary.failed_traces == 1


def test_runner_persists_pipeline_failure_and_validates_example_identity(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    with pytest.raises(ValueError, match="metadata order"):
        evaluate_systems(
            examples=(),
            metadata=metadata,
            store=ResumableSystemRunStore(tmp_path / "identity", metadata),
            pipeline_factory=lambda _: _Pipeline(),
            token_counter=_Counter(),
        )

    store = ResumableSystemRunStore(tmp_path / "failure", metadata)
    summary = evaluate_systems(
        examples=(_example(),),
        metadata=metadata,
        store=store,
        pipeline_factory=lambda _: _FailingPipeline(),
        token_counter=_Counter(),
        progress=lambda done, total: (done, total),
    )
    assert summary.failed_traces == 1
    failed = store.load_trace(SystemCondition(system=SystemVariant.FIXED, top_k=1), "q1")
    assert failed is not None
    assert failed.error_type == "RuntimeError"


def test_store_rejects_resume_conflicts_and_mutation(tmp_path: Path) -> None:
    metadata = _metadata()
    store = ResumableSystemRunStore(tmp_path, metadata)
    condition = SystemCondition(system=SystemVariant.FIXED, top_k=1)
    original = SystemQuestionTrace(
        condition=condition,
        question=_example().inference,
        evaluation_labels=_example().evaluation,
        status="failed",
        error_type="FirstError",
        error_message="first",
    )
    store.write_trace(original)
    store.write_trace(original)
    changed = original.model_copy(update={"error_message": "changed"})
    with pytest.raises(FrozenArtifactError, match="refusing to replace"):
        store.write_trace(changed)

    conflicting = metadata.model_copy(update={"dataset_version": "other-dataset"})
    with pytest.raises(SourceValidationError, match="resume metadata differs"):
        ResumableSystemRunStore(tmp_path, conflicting)


def test_completed_and_corrupt_exports_are_immutable(tmp_path: Path) -> None:
    metadata = _metadata()
    store = ResumableSystemRunStore(tmp_path, metadata)
    summary = evaluate_systems(
        examples=(_example(),),
        metadata=metadata,
        store=store,
        pipeline_factory=lambda _: _Pipeline(),
        token_counter=_Counter(),
    )
    with pytest.raises(FrozenArtifactError, match="completed run ID"):
        ResumableSystemRunStore(tmp_path, metadata)
    with pytest.raises(ValueError, match="summary metadata differs"):
        store.finalize(
            summary.model_copy(
                update={
                    "metadata": metadata.model_copy(update={"dataset_version": "wrong"})
                }
            )
        )

    trace_path = next((store.directory / "questions").rglob("*.json"))
    trace_path.write_text(trace_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(SourceValidationError, match="integrity failure"):
        load_system_export(store.directory)


def test_trace_rejects_a_pipeline_type_from_the_wrong_system() -> None:
    trace = _fixed_trace()
    with pytest.raises(ValidationError, match="trace type must match"):
        SystemQuestionTrace(
            condition=SystemCondition(system=SystemVariant.EPSA, top_k=1),
            question=_example().inference,
            evaluation_labels=_example().evaluation,
            status="completed",
            pipeline_trace=trace,
            generated_answer="London",
            quality={"exact_match": 1, "token_f1": 1, "generation_failed": False},
            efficiency={
                "final_context_tokens": 1,
                "candidate_context_tokens": 1,
                "llm_input_tokens": 1,
                "llm_output_tokens": 1,
                "llm_cached_input_tokens": 0,
                "hop2_activated": False,
                "insufficient_no_query": False,
                "latency_ms": 1,
            },
            retrieval_quality={
                "gold_chunk_ids": ("c1",),
                "hop1_found_chunk_ids": ("c1",),
                "final_found_chunk_ids": ("c1",),
                "all_gold_available_hop1": True,
                "all_gold_available_final": True,
                "hop2_evidence_gain_count": 0,
            },
        )


def test_model_token_counter_uses_locked_o200k_encoding() -> None:
    counter = TikTokenCounter(
        model="gpt-4o-mini-2024-07-18", expected_encoding="o200k_base"
    )

    assert counter.count("London") == 1
    with pytest.raises(ValueError, match="mapping mismatch"):
        TikTokenCounter(
            model="gpt-4o-mini-2024-07-18", expected_encoding="cl100k_base"
        )
