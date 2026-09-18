"""Public, typed response models for the read-only observability API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

BenchmarkRole = Literal["development", "test", "diagnostic"]
RunStatus = Literal["registered", "running", "completed", "failed"]
QuestionStatus = Literal["pending", "completed", "failed"]


class ApiModel(BaseModel):
    """Reject accidental public fields while retaining immutable response values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiError(ApiModel):
    code: str
    message: str
    details: dict[str, JsonValue] | None = None


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"
    service: Literal["epsa-observability-api"] = "epsa-observability-api"
    version: str


class DatabaseHealthResponse(HealthResponse):
    schema_status: Literal["compatible"] = "compatible"
    expected_revision: str


class RunListItem(ApiModel):
    run_id: str
    experiment_type: str
    benchmark_role: BenchmarkRole
    dataset_version: str
    corpus_version: str
    retriever_version: str
    status: RunStatus
    registered_at: datetime
    planned_questions: int
    completed_questions: int
    failed_questions: int


class RunPage(ApiModel):
    items: list[RunListItem]
    next_cursor: str | None


class RunDetail(ApiModel):
    run_id: str
    experiment_type: str
    benchmark_role: BenchmarkRole
    dataset_version: str
    corpus_version: str
    git_commit_sha: str
    git_dirty: bool
    retriever_version: str
    evaluator_version: str
    retrieval_depth: int
    configuration_fingerprint: str
    metadata: dict[str, JsonValue]
    component_versions: dict[str, str]
    storage_provenance: dict[str, JsonValue]
    status: RunStatus
    registered_at: datetime
    started_event_at: datetime | None
    started_at: datetime | None
    ended_at: datetime | None
    planned_questions: int
    attempted_questions: int
    completed_questions: int
    failed_questions: int
    event_count: int
    summary: dict[str, JsonValue] | None


class QuestionListItem(ApiModel):
    question_id: str
    position: int
    status: QuestionStatus
    error_type: str | None
    metrics: dict[str, float] | None
    latency_ms: float | None
    retrieval_core_latency_ms: float | None


class QuestionPage(ApiModel):
    items: list[QuestionListItem]
    next_cursor: str | None


class TraceEvent(ApiModel):
    event_id: str
    trace_id: str
    event_type: str
    source: str
    source_version: str | None
    occurred_at: datetime
    ingested_at: datetime
    sequence: int
    content_sha256: str
    envelope: dict[str, JsonValue]


class QuestionTrace(ApiModel):
    question: QuestionListItem
    events: list[TraceEvent]


class MetricDelta(ApiModel):
    name: str
    baseline: float
    candidate: float
    delta: float
    baseline_denominator: int
    candidate_denominator: int


class SummaryFieldDelta(ApiModel):
    name: str
    baseline: float
    candidate: float
    delta: float


class RunComparison(ApiModel):
    baseline_run_id: str
    candidate_run_id: str
    benchmark_role: BenchmarkRole
    dataset_version: str
    corpus_version: str
    planned_questions: int
    metric_deltas: list[MetricDelta]
    summary_field_deltas: list[SummaryFieldDelta]
