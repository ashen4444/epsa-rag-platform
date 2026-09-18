from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from epsa_observability_api.app import create_app


class FakeReadRepository:
    def __init__(self) -> None:
        now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
        self.runs = {
            "baseline": _run("baseline", now, 0.5),
            "candidate": _run("candidate", now - timedelta(seconds=1), 0.7),
            "incomplete": _run("incomplete", now - timedelta(seconds=2), 0.2, status="running"),
        }
        self.questions = {
            "baseline": [_question("q1", 0, "completed"), _question("q2", 1, "failed")],
            "candidate": [_question("q1", 0, "completed"), _question("q2", 1, "failed")],
            "incomplete": [_question("q1", 0, "pending"), _question("q2", 1, "pending")],
        }

    def check_compatibility(self) -> None:
        return None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.runs.get(run_id)
        return deepcopy(row) if row else None

    def list_runs(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = list(self.runs.values())
        for field in (
            "benchmark_role",
            "status",
            "experiment_type",
            "dataset_version",
            "corpus_version",
            "retriever_version",
        ):
            if value := kwargs[field]:
                rows = [row for row in rows if row[field] == value]
        rows.sort(key=lambda row: (-row["registered_at"].timestamp(), row["run_id"]))
        after = kwargs["after_registered_at"]
        if after is not None:
            after_id = kwargs["after_run_id"]
            rows = [
                row
                for row in rows
                if row["registered_at"] < after
                or (row["registered_at"] == after and row["run_id"] > after_id)
            ]
        fields = (
            "run_id",
            "experiment_type",
            "benchmark_role",
            "dataset_version",
            "corpus_version",
            "retriever_version",
            "status",
            "registered_at",
            "planned_questions",
            "completed_questions",
            "failed_questions",
        )
        return [
            {field: deepcopy(row[field]) for field in fields}
            for row in rows[: kwargs["limit"] + 1]
        ]

    def list_questions(self, run_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        rows = self.questions[run_id]
        if status := kwargs["status"]:
            rows = [row for row in rows if row["status"] == status]
        if (has_failure := kwargs["has_failure"]) is not None:
            rows = [row for row in rows if (row["error_type"] is not None) == has_failure]
        if question_id := kwargs["question_id"]:
            rows = [row for row in rows if row["question_id"] == question_id]
        if (after := kwargs["after_position"]) is not None:
            rows = [row for row in rows if row["position"] > after]
        return deepcopy(rows[: kwargs["limit"] + 1])

    def get_question_trace(self, run_id: str, question_id: str) -> dict[str, Any] | None:
        question = next(
            (item for item in self.questions[run_id] if item["question_id"] == question_id), None
        )
        if question is None:
            return None
        event = {
            "event_id": "event:trace",
            "trace_id": "trace:question",
            "event_type": "evaluation.question.completed",
            "source": "retrieval-evaluation",
            "source_version": "retrieval-evaluation-v3",
            "occurred_at": datetime(2026, 9, 18, 12, 1, tzinfo=UTC),
            "ingested_at": datetime(2026, 9, 18, 12, 2, tzinfo=UTC),
            "sequence": 2,
            "content_sha256": "a" * 64,
            "envelope": {"payload": {"question": {"text": "native  whitespace"}}},
        }
        return {"question": deepcopy(question), "events": [event]}


def _run(
    run_id: str, registered_at: datetime, recall: float, *, status: str = "completed"
) -> dict[str, Any]:
    summary: dict[str, Any] | None
    if status == "completed":
        summary = {
            "status": "completed",
            "metrics": {"recall_at_10": recall},
            "metric_denominators": {"recall_at_10": 2},
            "latency_p50_ms": 10.0 + recall,
            "latency_p95_ms": 20.0 + recall,
            "successful_latency_p50_ms": 10.0 + recall,
            "successful_latency_p95_ms": 20.0 + recall,
            "retrieval_core_latency_p50_ms": 9.0 + recall,
            "retrieval_core_latency_p95_ms": 19.0 + recall,
            "retrieval_seconds": 1.0,
            "retrieval_core_seconds": 0.9,
            "evaluation_wall_seconds": 1.1,
            "throughput_questions_per_second": 2.0,
            "retrieval_throughput_questions_per_second": 2.1,
            "retrieval_core_throughput_questions_per_second": 2.2,
        }
    else:
        summary = None
    return {
        "run_id": run_id,
        "experiment_type": "retriever-evaluation",
        "benchmark_role": "development",
        "dataset_version": "hotpotqa_1000_v1",
        "corpus_version": "hotpotqa_10000_v1",
        "git_commit_sha": "a" * 40,
        "git_dirty": False,
        "retriever_version": "hybrid-retriever-v2",
        "evaluator_version": "retrieval-evaluation-v3",
        "retrieval_depth": 20,
        "configuration_fingerprint": "b" * 64,
        "metadata_payload": {
            "question_ids": ["q1", "q2"],
            "configuration": {
                "evaluator_version": "retrieval-evaluation-v3",
                "relevance": "exact_chunk",
                "cutoffs": [1, 5, 10],
            },
        },
        "component_versions": {},
        "storage_provenance": {"mode": "live"},
        "status": status,
        "registered_at": registered_at,
        "started_event_at": registered_at,
        "started_at": registered_at,
        "ended_at": registered_at if status == "completed" else None,
        "planned_questions": 2,
        "attempted_questions": 2 if status == "completed" else 0,
        "completed_questions": 2 if status == "completed" else 0,
        "failed_questions": 0,
        "event_count": 4 if status == "completed" else 1,
        "summary_payload": summary,
    }


def _question(question_id: str, position: int, status: str) -> dict[str, Any]:
    failed = status == "failed"
    return {
        "question_id": question_id,
        "position": position,
        "status": status,
        "error_type": "retrieval_error" if failed else None,
        "metrics": None if status == "pending" else {"recall_at_10": 0.5},
        "latency_ms": None if status == "pending" else 4.0,
        "retrieval_core_latency_ms": None if status == "pending" else 3.0,
    }


@pytest.fixture
def repository() -> FakeReadRepository:
    return FakeReadRepository()


@pytest.fixture
def client(repository: FakeReadRepository) -> TestClient:
    return TestClient(create_app(repository=repository))
