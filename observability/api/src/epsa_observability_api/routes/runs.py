"""Read-only run, question, trace, and comparison routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from epsa_observability_api.dependencies import get_read_repository
from epsa_observability_api.models import (
    BenchmarkRole,
    QuestionPage,
    QuestionStatus,
    QuestionTrace,
    RunComparison,
    RunDetail,
    RunPage,
    RunStatus,
)
from epsa_observability_api.service import ReadRepository, RunQueryService

router = APIRouter(prefix="/api/v1", tags=["runs"])
PageLimit = Annotated[int, Query(ge=1, le=100, description="Maximum records returned.")]
ReadRepositoryDependency = Annotated[ReadRepository, Depends(get_read_repository)]


@router.get("/runs/compare", response_model=RunComparison)
def compare_runs(
    baseline_run_id: str,
    candidate_run_id: str,
    repository: ReadRepositoryDependency,
) -> RunComparison:
    """Compare aggregate metrics for two scientifically compatible completed runs."""

    return RunQueryService(repository).compare_runs(baseline_run_id, candidate_run_id)


@router.get("/runs", response_model=RunPage)
def list_runs(
    repository: ReadRepositoryDependency,
    limit: PageLimit = 50,
    cursor: str | None = None,
    benchmark_role: BenchmarkRole | None = None,
    status: RunStatus | None = None,
    experiment_type: str | None = None,
    dataset_version: str | None = None,
    corpus_version: str | None = None,
    retriever_version: str | None = None,
) -> RunPage:
    """List runs newest first using a stable opaque cursor."""

    return RunQueryService(repository).list_runs(
        limit=limit,
        cursor=cursor,
        benchmark_role=benchmark_role,
        status=status,
        experiment_type=experiment_type,
        dataset_version=dataset_version,
        corpus_version=corpus_version,
        retriever_version=retriever_version,
    )


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(
    run_id: str,
    repository: ReadRepositoryDependency,
) -> RunDetail:
    """Return immutable provenance, lifecycle counts, and recorded summary data."""

    return RunQueryService(repository).get_run(run_id)


@router.get("/runs/{run_id}/questions", response_model=QuestionPage)
def list_questions(
    run_id: str,
    repository: ReadRepositoryDependency,
    limit: PageLimit = 50,
    cursor: str | None = None,
    status: QuestionStatus | None = None,
    has_failure: bool | None = None,
    question_id: str | None = None,
) -> QuestionPage:
    """List a run's stored question projections in planned-question order."""

    return RunQueryService(repository).list_questions(
        run_id,
        limit=limit,
        cursor=cursor,
        status=status,
        has_failure=has_failure,
        question_id=question_id,
    )


@router.get("/runs/{run_id}/questions/{question_id}/trace", response_model=QuestionTrace)
def get_question_trace(
    run_id: str,
    question_id: str,
    repository: ReadRepositoryDependency,
) -> QuestionTrace:
    """Return the ordered, original Phase 5A event envelopes for one question."""

    return RunQueryService(repository).get_question_trace(run_id, question_id)
