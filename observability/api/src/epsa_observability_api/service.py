"""Application service for API mapping, cursors, and scientific comparison rules."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel, ValidationError

from epsa_observability_api.errors import IncompatibleRunsError, NotFoundError, StoredRecordError
from epsa_observability_api.models import (
    MetricDelta,
    QuestionListItem,
    QuestionPage,
    QuestionTrace,
    RunComparison,
    RunDetail,
    RunListItem,
    RunPage,
    SummaryFieldDelta,
    TraceEvent,
)


class ReadRepository(Protocol):
    """The small read surface used by routes and replaceable in unit tests."""

    def check_compatibility(self) -> None: ...

    def get_run(self, run_id: str) -> dict[str, Any] | None: ...

    def list_runs(
        self,
        *,
        limit: int,
        benchmark_role: str | None,
        status: str | None,
        experiment_type: str | None,
        dataset_version: str | None,
        corpus_version: str | None,
        retriever_version: str | None,
        after_registered_at: datetime | None,
        after_run_id: str | None,
    ) -> list[dict[str, Any]]: ...

    def list_questions(
        self,
        run_id: str,
        *,
        limit: int,
        status: str | None,
        has_failure: bool | None,
        question_id: str | None,
        after_position: int | None,
    ) -> list[dict[str, Any]]: ...

    def get_question_trace(self, run_id: str, question_id: str) -> dict[str, Any] | None: ...


_SUMMARY_FIELDS = (
    "latency_p50_ms",
    "latency_p95_ms",
    "successful_latency_p50_ms",
    "successful_latency_p95_ms",
    "retrieval_core_latency_p50_ms",
    "retrieval_core_latency_p95_ms",
    "retrieval_seconds",
    "retrieval_core_seconds",
    "evaluation_wall_seconds",
    "throughput_questions_per_second",
    "retrieval_throughput_questions_per_second",
    "retrieval_core_throughput_questions_per_second",
)
ModelT = TypeVar("ModelT", bound=BaseModel)


class RunQueryService:
    """Keep HTTP handlers thin and make comparison semantics independently testable."""

    def __init__(self, repository: ReadRepository) -> None:
        self._repository = repository

    def list_runs(
        self,
        *,
        limit: int,
        cursor: str | None,
        benchmark_role: str | None,
        status: str | None,
        experiment_type: str | None,
        dataset_version: str | None,
        corpus_version: str | None,
        retriever_version: str | None,
    ) -> RunPage:
        after_registered_at, after_run_id = self._decode_run_cursor(cursor)
        rows = self._repository.list_runs(
            limit=limit,
            benchmark_role=benchmark_role,
            status=status,
            experiment_type=experiment_type,
            dataset_version=dataset_version,
            corpus_version=corpus_version,
            retriever_version=retriever_version,
            after_registered_at=after_registered_at,
            after_run_id=after_run_id,
        )
        page_rows, extra = rows[:limit], rows[limit:]
        items = [self._validate_model(RunListItem, row) for row in page_rows]
        next_cursor = None
        if extra and items:
            last = items[-1]
            next_cursor = self._encode_run_cursor(last.registered_at, last.run_id)
        return RunPage(items=items, next_cursor=next_cursor)

    def get_run(self, run_id: str) -> RunDetail:
        row = self._get_run_row(run_id)
        metadata = row.pop("metadata_payload")
        summary = row.pop("summary_payload")
        public_row = {
            **row,
            "metadata": metadata,
            "summary": summary,
        }
        return self._validate_model(RunDetail, public_row)

    def list_questions(
        self,
        run_id: str,
        *,
        limit: int,
        cursor: str | None,
        status: str | None,
        has_failure: bool | None,
        question_id: str | None,
    ) -> QuestionPage:
        self._get_run_row(run_id)
        after_position = self._decode_question_cursor(cursor)
        rows = self._repository.list_questions(
            run_id,
            limit=limit,
            status=status,
            has_failure=has_failure,
            question_id=question_id,
            after_position=after_position,
        )
        page_rows, extra = rows[:limit], rows[limit:]
        items = [self._validate_model(QuestionListItem, row) for row in page_rows]
        next_cursor = self._encode_question_cursor(items[-1].position) if extra and items else None
        return QuestionPage(items=items, next_cursor=next_cursor)

    def get_question_trace(self, run_id: str, question_id: str) -> QuestionTrace:
        self._get_run_row(run_id)
        row = self._repository.get_question_trace(run_id, question_id)
        if row is None:
            raise NotFoundError("Question was not found in the requested run.")
        return QuestionTrace(
            question=self._validate_model(QuestionListItem, row["question"]),
            events=[self._validate_model(TraceEvent, event) for event in row["events"]],
        )

    def compare_runs(self, baseline_run_id: str, candidate_run_id: str) -> RunComparison:
        baseline = self._get_run_row(baseline_run_id)
        candidate = self._get_run_row(candidate_run_id)
        criteria = self._incompatible_criteria(baseline, candidate)
        if criteria:
            raise IncompatibleRunsError(tuple(criteria))
        baseline_summary = self._mapping(baseline["summary_payload"], "baseline summary")
        candidate_summary = self._mapping(candidate["summary_payload"], "candidate summary")
        baseline_metrics = self._mapping(baseline_summary.get("metrics"), "baseline metrics")
        candidate_metrics = self._mapping(candidate_summary.get("metrics"), "candidate metrics")
        baseline_denominators = self._mapping(
            baseline_summary.get("metric_denominators"), "baseline metric denominators"
        )
        candidate_denominators = self._mapping(
            candidate_summary.get("metric_denominators"), "candidate metric denominators"
        )
        metric_deltas = [
            MetricDelta(
                name=name,
                baseline=self._number(baseline_metrics[name], f"baseline metric {name}"),
                candidate=self._number(candidate_metrics[name], f"candidate metric {name}"),
                delta=self._number(candidate_metrics[name], f"candidate metric {name}")
                - self._number(baseline_metrics[name], f"baseline metric {name}"),
                baseline_denominator=self._integer(
                    baseline_denominators[name], f"baseline denominator {name}"
                ),
                candidate_denominator=self._integer(
                    candidate_denominators[name], f"candidate denominator {name}"
                ),
            )
            for name in sorted(baseline_metrics)
        ]
        summary_field_deltas = [
            SummaryFieldDelta(
                name=name,
                baseline=self._number(baseline_summary[name], f"baseline summary field {name}"),
                candidate=self._number(candidate_summary[name], f"candidate summary field {name}"),
                delta=self._number(candidate_summary[name], f"candidate summary field {name}")
                - self._number(baseline_summary[name], f"baseline summary field {name}"),
            )
            for name in _SUMMARY_FIELDS
            if baseline_summary.get(name) is not None and candidate_summary.get(name) is not None
        ]
        return RunComparison(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            benchmark_role=baseline["benchmark_role"],
            dataset_version=baseline["dataset_version"],
            corpus_version=baseline["corpus_version"],
            planned_questions=baseline["planned_questions"],
            metric_deltas=metric_deltas,
            summary_field_deltas=summary_field_deltas,
        )

    def _get_run_row(self, run_id: str) -> dict[str, Any]:
        row = self._repository.get_run(run_id)
        if row is None:
            raise NotFoundError("Run was not found.")
        return row.copy()

    @staticmethod
    def _validate_model(model: type[ModelT], value: Any) -> ModelT:
        try:
            return model.model_validate(value)
        except ValidationError:
            raise StoredRecordError("Stored experiment data is invalid.") from None

    def _incompatible_criteria(
        self, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> list[str]:
        criteria: list[str] = []
        for field in (
            "status",
            "experiment_type",
            "benchmark_role",
            "dataset_version",
            "corpus_version",
        ):
            if baseline.get(field) != candidate.get(field):
                criteria.append(field)
        if baseline.get("status") != "completed" or candidate.get("status") != "completed":
            criteria.append("both runs must be completed")
            return criteria
        baseline_metadata = self._mapping(baseline.get("metadata_payload"), "baseline metadata")
        candidate_metadata = self._mapping(candidate.get("metadata_payload"), "candidate metadata")
        if baseline_metadata.get("question_ids") != candidate_metadata.get("question_ids"):
            criteria.append("planned question IDs and order")
        baseline_config = self._mapping(
            baseline_metadata.get("configuration"), "baseline configuration"
        )
        candidate_config = self._mapping(
            candidate_metadata.get("configuration"), "candidate configuration"
        )
        for field in ("evaluator_version", "relevance", "cutoffs"):
            if baseline_config.get(field) != candidate_config.get(field):
                criteria.append(f"configuration.{field}")
        baseline_summary = self._mapping(baseline.get("summary_payload"), "baseline summary")
        candidate_summary = self._mapping(candidate.get("summary_payload"), "candidate summary")
        if (
            baseline_summary.get("status") != "completed"
            or candidate_summary.get("status") != "completed"
        ):
            criteria.append("completed summaries")
        for field in ("metrics", "metric_denominators"):
            baseline_values = self._mapping(baseline_summary.get(field), f"baseline {field}")
            candidate_values = self._mapping(candidate_summary.get(field), f"candidate {field}")
            if baseline_values.keys() != candidate_values.keys() or (
                field == "metric_denominators" and baseline_values != candidate_values
            ):
                criteria.append(field)
        return criteria

    @staticmethod
    def _mapping(value: Any, description: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise StoredRecordError(f"Stored {description} is invalid.")
        return value

    @staticmethod
    def _number(value: Any, description: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StoredRecordError(f"Stored {description} is invalid.")
        return float(value)

    @staticmethod
    def _integer(value: Any, description: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise StoredRecordError(f"Stored {description} is invalid.")
        return cast(int, value)

    @staticmethod
    def _encode_run_cursor(registered_at: datetime, run_id: str) -> str:
        payload = json.dumps(
            {"registered_at": registered_at.isoformat(), "run_id": run_id},
            separators=(",", ":"),
        ).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_run_cursor(cursor: str | None) -> tuple[datetime | None, str | None]:
        if cursor is None:
            return None, None
        data = RunQueryService._decode_cursor(cursor)
        registered_at, run_id = data.get("registered_at"), data.get("run_id")
        if not isinstance(registered_at, str) or not isinstance(run_id, str) or not run_id:
            raise ValueError("Cursor is invalid.")
        try:
            parsed = datetime.fromisoformat(registered_at)
        except ValueError:
            raise ValueError("Cursor is invalid.") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Cursor is invalid.")
        return parsed, run_id

    @staticmethod
    def _encode_question_cursor(position: int) -> str:
        payload = json.dumps({"position": position}, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_question_cursor(cursor: str | None) -> int | None:
        if cursor is None:
            return None
        position = RunQueryService._decode_cursor(cursor).get("position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise ValueError("Cursor is invalid.")
        return position

    @staticmethod
    def _decode_cursor(cursor: str) -> Mapping[str, Any]:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            value = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        except (UnicodeEncodeError, binascii.Error, ValueError):
            raise ValueError("Cursor is invalid.") from None
        if not isinstance(value, Mapping):
            raise ValueError("Cursor is invalid.")
        return value
