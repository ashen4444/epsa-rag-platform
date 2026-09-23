"""Storage-neutral contracts for paired end-to-end system evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel
from epsa_rag.data.models import EvaluationLabels, QuestionInput
from epsa_rag.pipeline.epsa_models import EPSAPipelineTrace
from epsa_rag.pipeline.models import AdaptiveBaselineTrace, FixedBaselineTrace
from epsa_rag.retrieval.config import HybridRetrieverConfig
from epsa_rag.retrieval.dense.query_cache import QueryEmbeddingObservation
from epsa_rag.retrieval.manifests import RetrievalIndexManifest

FiniteNonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class SystemVariant(StrEnum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    EPSA = "epsa"


class SystemCondition(ContractModel):
    system: SystemVariant
    top_k: Annotated[int, Field(ge=1)]

    @property
    def key(self) -> str:
        return f"{self.system.value}-top-{self.top_k}"


class SystemEvaluationConfig(ConfigModel):
    """Complete research-critical settings for one paired campaign."""

    evaluator_version: Literal["system-evaluation-v1"] = "system-evaluation-v1"
    systems: tuple[SystemVariant, ...] = (
        SystemVariant.FIXED,
        SystemVariant.ADAPTIVE,
        SystemVariant.EPSA,
    )
    top_ks: tuple[int, ...] = (5, 10, 20, 30)
    max_additional_hops: Literal[1] = 1
    question_limit: int | None = Field(default=None, ge=1)
    tokenizer_model: Literal["gpt-4o-mini-2024-07-18"] = "gpt-4o-mini-2024-07-18"
    tokenizer_encoding: Literal["o200k_base"] = "o200k_base"
    retriever: HybridRetrieverConfig
    query_embedding_cache_version: Identifier = "query-embeddings-openai-small-v1"
    openai_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    openai_max_retries: int = Field(default=2, ge=0)
    faiss_threads: int = Field(default=1, ge=1)
    percentile_method: Literal["linear-n-minus-one"] = "linear-n-minus-one"

    @model_validator(mode="after")
    def require_complete_unique_design(self) -> SystemEvaluationConfig:
        if tuple(dict.fromkeys(self.systems)) != self.systems or not self.systems:
            raise ValueError("systems must be unique and nonempty")
        if tuple(sorted(set(self.top_ks))) != self.top_ks or not self.top_ks:
            raise ValueError("top_ks must be sorted, unique, and nonempty")
        if self.top_ks[0] < 1:
            raise ValueError("top_ks must be positive")
        if self.retriever.fusion.result_k < max(self.top_ks):
            raise ValueError("retriever result_k must cover every system condition")
        return self

    def conditions(self) -> tuple[SystemCondition, ...]:
        return tuple(
            SystemCondition(system=system, top_k=top_k)
            for top_k in self.top_ks
            for system in self.systems
        )


class SystemRunMetadata(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,99}$")]
    experiment_type: Literal["paired-system-evaluation"] = "paired-system-evaluation"
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    git_dirty: bool
    source_sha256: dict[str, str]
    runtime: dict[str, str]
    dataset_version: Identifier
    corpus_version: Identifier
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_manifests: tuple[RetrievalIndexManifest, ...]
    question_ids: tuple[Identifier, ...]
    full_dataset_question_count: Annotated[int, Field(ge=1)]
    configuration: SystemEvaluationConfig
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class AnswerQuality(ContractModel):
    exact_match: FiniteNonNegative
    token_f1: FiniteNonNegative
    generation_failed: bool


class SystemEfficiency(ContractModel):
    final_context_tokens: Annotated[int, Field(ge=0)]
    candidate_context_tokens: Annotated[int, Field(ge=0)]
    pruning_reduction_percent: float | None = Field(default=None, le=100, allow_inf_nan=False)
    llm_input_tokens: Annotated[int, Field(ge=0)]
    llm_output_tokens: Annotated[int, Field(ge=0)]
    llm_cached_input_tokens: Annotated[int, Field(ge=0)]
    hop2_activated: bool
    insufficient_no_query: bool
    latency_ms: FiniteNonNegative


PipelineTrace = FixedBaselineTrace | AdaptiveBaselineTrace | EPSAPipelineTrace


class RetrievalQuality(ContractModel):
    """Evaluation-only evidence availability; never supplied to inference components."""

    gold_chunk_ids: tuple[Identifier, ...]
    hop1_found_chunk_ids: tuple[Identifier, ...]
    final_found_chunk_ids: tuple[Identifier, ...]
    all_gold_available_hop1: bool
    all_gold_available_final: bool
    hop2_evidence_gain_count: Annotated[int, Field(ge=0)]


class SystemQuestionTrace(ContractModel):
    schema_version: Literal["system-question-trace-v1"] = "system-question-trace-v1"
    condition: SystemCondition
    question: QuestionInput
    evaluation_labels: EvaluationLabels
    status: Literal["completed", "failed"]
    error_type: str | None = None
    error_message: str | None = None
    pipeline_trace: PipelineTrace | None = None
    query_embeddings: tuple[QueryEmbeddingObservation, ...] = ()
    generated_answer: str | None = None
    quality: AnswerQuality | None = None
    efficiency: SystemEfficiency | None = None
    retrieval_quality: RetrievalQuality | None = None

    @model_validator(mode="after")
    def require_status_payload(self) -> SystemQuestionTrace:
        complete = self.status == "completed"
        payloads = (
            self.pipeline_trace,
            self.generated_answer,
            self.quality,
            self.efficiency,
            self.retrieval_quality,
        )
        if complete and (any(value is None for value in payloads) or self.error_type is not None):
            raise ValueError("completed system traces require outputs and no error")
        if not complete and (
            any(value is not None for value in payloads) or self.error_type is None
        ):
            raise ValueError("failed system traces require only explicit error details")
        if not complete and self.query_embeddings:
            raise ValueError("failed system traces cannot retain partial embedding observations")
        if self.pipeline_trace is not None:
            expected_type = {
                SystemVariant.FIXED: FixedBaselineTrace,
                SystemVariant.ADAPTIVE: AdaptiveBaselineTrace,
                SystemVariant.EPSA: EPSAPipelineTrace,
            }[self.condition.system]
            if not isinstance(self.pipeline_trace, expected_type):
                raise ValueError("pipeline trace type must match the evaluated system")
        return self


class ConditionSummary(ContractModel):
    condition: SystemCondition
    planned_questions: Annotated[int, Field(ge=0)]
    completed_questions: Annotated[int, Field(ge=0)]
    failed_questions: Annotated[int, Field(ge=0)]
    metrics: dict[str, float]


class SystemRunSummary(ContractModel):
    schema_version: Literal["system-run-summary-v1"] = "system-run-summary-v1"
    metadata: SystemRunMetadata
    started_at: AwareDatetime
    ended_at: AwareDatetime
    status: Literal["completed", "completed_with_failures"]
    full_benchmark: bool
    completed_traces: Annotated[int, Field(ge=0)]
    failed_traces: Annotated[int, Field(ge=0)]
    condition_summaries: tuple[ConditionSummary, ...]
    paired_context_reductions: dict[str, float]


def utc_now() -> datetime:
    return datetime.now(UTC)
