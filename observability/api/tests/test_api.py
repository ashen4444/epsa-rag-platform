from __future__ import annotations

from fastapi.testclient import TestClient

from epsa_observability_api.app import create_app

from .conftest import FakeReadRepository


def test_health_and_database_readiness(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    readiness = client.get("/health/database")
    assert readiness.status_code == 200
    assert readiness.json()["schema_status"] == "compatible"


def test_database_readiness_is_sanitized_without_configuration() -> None:
    response = TestClient(create_app()).get("/health/database")
    assert response.status_code == 503
    assert response.json() == {
        "code": "database_unavailable",
        "message": "Database is unavailable or its schema is incompatible.",
        "details": None,
    }


def test_runs_pagination_filters_and_invalid_cursor(client: TestClient) -> None:
    first = client.get("/api/v1/runs?limit=1")
    assert first.status_code == 200
    assert [row["run_id"] for row in first.json()["items"]] == ["baseline"]
    second = client.get("/api/v1/runs", params={"cursor": first.json()["next_cursor"]})
    assert [row["run_id"] for row in second.json()["items"]] == ["candidate", "incomplete"]
    assert client.get("/api/v1/runs?status=running").json()["items"][0]["run_id"] == "incomplete"
    invalid = client.get("/api/v1/runs?cursor=not-a-cursor")
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "validation_error"


def test_run_questions_and_exact_trace(client: TestClient) -> None:
    detail = client.get("/api/v1/runs/baseline")
    assert detail.status_code == 200
    assert detail.json()["metadata"]["question_ids"] == ["q1", "q2"]
    questions = client.get("/api/v1/runs/baseline/questions?has_failure=true")
    assert [row["question_id"] for row in questions.json()["items"]] == ["q2"]
    exact = client.get("/api/v1/runs/baseline/questions?question_id=q1")
    assert [row["question_id"] for row in exact.json()["items"]] == ["q1"]
    trace = client.get("/api/v1/runs/baseline/questions/q1/trace")
    assert trace.status_code == 200
    text = trace.json()["events"][0]["envelope"]["payload"]["question"]["text"]
    assert text == "native  whitespace"
    assert client.get("/api/v1/runs/missing/questions").status_code == 404
    assert client.get("/api/v1/runs/baseline/questions/missing/trace").status_code == 404


def test_compare_runs_and_report_incompatible_pair(
    client: TestClient, repository: FakeReadRepository
) -> None:
    comparison = client.get(
        "/api/v1/runs/compare",
        params={"baseline_run_id": "baseline", "candidate_run_id": "candidate"},
    )
    assert comparison.status_code == 200
    metric = comparison.json()["metric_deltas"][0]
    assert metric == {
        "name": "recall_at_10",
        "baseline": 0.5,
        "candidate": 0.7,
        "delta": 0.19999999999999996,
        "baseline_denominator": 2,
        "candidate_denominator": 2,
    }
    incompatible = client.get(
        "/api/v1/runs/compare",
        params={"baseline_run_id": "baseline", "candidate_run_id": "incomplete"},
    )
    assert incompatible.status_code == 409
    assert "both runs must be completed" in incompatible.json()["details"]["criteria"]
