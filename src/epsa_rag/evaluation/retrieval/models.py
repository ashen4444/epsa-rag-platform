"""Storage-neutral Phase 4 contracts; no database or web framework dependency."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel
from epsa_rag.data.models import QuestionInput, SupportingFactLabel
from epsa_rag.retrieval.config import HybridRetrieverConfig
from epsa_rag.retrieval.manifests import RetrievalIndexManifest
from epsa_rag.retrieval.models import RetrievalResult

FiniteNonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class EvaluationConfig(ConfigModel):
    """Persisted choices for one serial, live-query benchmark."""

    evaluator_version: Literal["retrieval-evaluation-v1"] = "retrieval-evaluation-v1"
    relevance: Literal["exact_chunk"] = "exact_chunk"
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"
    cutoffs: tuple[int, ...] = (1, 5, 10)
    retriever: HybridRetrieverConfig = Field(default_factory=HybridRetrieverConfig)
    warmup_questions: int = Field(default=0, ge=0)
    question_limit: int | None = Field(default=None, ge=1)
    concurrency: Literal[1] = 1
    query_embedding_cache: Literal["disabled"] = "disabled"
    openai_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    openai_max_retries: int = Field(default=2, ge=0)
    faiss_threads: int = Field(default=1, ge=1)
    percentile_method: Literal["linear-n-minus-one"] = "linear-n-minus-one"

    @model_validator(mode="after")
    def validate_depths(self) -> EvaluationConfig:
        if not self.cutoffs or tuple(sorted(set(self.cutoffs))) != self.cutoffs:
            raise ValueError("cutoffs must be sorted, unique, and nonempty")
        if self.cutoffs[0] < 1 or self.cutoffs[-1] > self.retriever.fusion.result_k:
            raise ValueError("cutoffs must be positive and fit within retrieval result_k")
        return self


class RunMetadata(ContractModel):
    """Provenance captured before the first request; no environment secrets."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,99}$")]
    experiment_type: Literal["retriever-evaluation"] = "retriever-evaluation"
    git_commit_sha: Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")]
    git_dirty: bool
    source_sha256: dict[str, str]
    runtime: dict[str, str]
    dataset_version: Identifier
    corpus_version: Identifier
    dataset_manifest_sha256: str
    dataset_file_sha256: str
    corpus_manifest_sha256: str
    corpus_file_sha256: str
    index_manifests: tuple[RetrievalIndexManifest, ...]
    question_ids: tuple[Identifier, ...]
    full_dataset_question_count: int = Field(ge=1)
    configuration: EvaluationConfig
    configuration_fingerprint: str


class QuestionTrace(ContractModel):
    """Full canonical retrieval output plus evaluation-only labels and diagnostics."""

    question: QuestionInput
    question_type: str
    difficulty: str
    gold_supporting_facts: tuple[SupportingFactLabel, ...]
    gold_identities: tuple[str, ...]
    missing_gold_identities: dict[str, tuple[str, ...]]
    status: Literal["completed", "failed"]
    error_type: str | None = None
    retrieval: RetrievalResult | None = None
    latency_ms: FiniteNonNegative
    metrics: dict[str, float]


class RunSummary(ContractModel):
    """A failed or partial run cannot be mistaken for a completed fixed benchmark."""

    metadata: RunMetadata
    started_at: AwareDatetime
    ended_at: AwareDatetime
    status: Literal["completed", "failed"]
    full_benchmark: bool
    planned_questions: int
    completed_questions: int
    failed_questions: int
    unattempted_questions: int
    warmup_completed: int
    error_type: str | None = None
    metrics: dict[str, float]
    metric_denominators: dict[str, int]
    latency_p50_ms: FiniteNonNegative | None
    latency_p95_ms: FiniteNonNegative | None
    successful_latency_p50_ms: FiniteNonNegative | None
    successful_latency_p95_ms: FiniteNonNegative | None
    retrieval_seconds: FiniteNonNegative
    evaluation_wall_seconds: FiniteNonNegative
    throughput_questions_per_second: FiniteNonNegative | None
    retrieval_throughput_questions_per_second: FiniteNonNegative | None


def utc_now() -> datetime:
    """Timezone-aware wall clock for provenance, separate from elapsed timing."""
    from datetime import UTC

    return datetime.now(UTC)
