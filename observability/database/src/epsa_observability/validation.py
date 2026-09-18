"""Storage checks preserve original payloads and never recalculate research results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Literal

from epsa_rag.evaluation.retrieval.models import QuestionTrace, RunMetadata, RunSummary
from epsa_rag.instrumentation.events import InstrumentationEvent
from pydantic import BaseModel, ValidationError

from epsa_observability.errors import StorageError

BenchmarkRole = Literal["development", "test", "diagnostic"]
START = "evaluation.run.started"
QUESTION = "evaluation.question.completed"
FINISH = "evaluation.run.finished"
BENCHMARKS = {
    "development": ("hotpotqa_1000_v1", "hotpotqa_10000_v1"),
    "test": ("hotpotqa_hard_10000_test_v1", "hotpotqa_hard_10000_test_corpus_v1"),
}


def canonical(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        raise StorageError("Payload must contain finite, JSON-serializable values.") from None


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def parse[T: BaseModel](model: type[T], payload: Any) -> T:
    try:
        return model.model_validate(payload)
    except ValidationError:
        raise StorageError("Payload does not match the supported research contract.") from None


def snapshot(event: InstrumentationEvent) -> tuple[InstrumentationEvent, dict[str, Any]]:
    # Snapshot nested mutable dictionaries as well as revalidating model_copy() inputs.
    envelope = json.loads(canonical(event.model_dump(mode="json")))
    return parse(InstrumentationEvent, envelope), envelope


def validate_metadata(payload: dict[str, Any], role: BenchmarkRole) -> RunMetadata:
    metadata = parse(RunMetadata, payload)
    canonical(payload)
    if role not in {"development", "test", "diagnostic"}:
        raise StorageError("An explicit benchmark role is required.")
    pair = (metadata.dataset_version, metadata.corpus_version)
    if role in BENCHMARKS and pair != BENCHMARKS[role]:
        raise StorageError("Benchmark role does not match the documented dataset/corpus pair.")
    if (
        role in BENCHMARKS
        and metadata.full_dataset_question_count
        != {
            "development": 1000,
            "test": 10000,
        }[role]
    ):
        raise StorageError("Full dataset count differs from the frozen benchmark definition.")
    if role == "diagnostic" and any(
        pair[0] == known[0] or pair[1] == known[1] for known in BENCHMARKS.values()
    ):
        raise StorageError("Frozen benchmark versions require their development or test role.")
    ids = metadata.question_ids
    if not ids or len(set(ids)) != len(ids) or len(ids) > metadata.full_dataset_question_count:
        raise StorageError("Run requires a nonempty, unique, ordered question selection.")
    limit = metadata.configuration.question_limit
    expected = metadata.full_dataset_question_count if limit is None else limit
    if len(ids) != expected or metadata.configuration.warmup_questions > len(ids):
        raise StorageError("Question selection does not agree with the recorded configuration.")
    # Hash the ORIGINAL configuration, before Pydantic fills legacy defaults.
    if digest(payload["configuration"]) != metadata.configuration_fingerprint:
        raise StorageError("Configuration fingerprint does not match the original configuration.")
    hashes = [
        metadata.dataset_manifest_sha256,
        metadata.dataset_file_sha256,
        metadata.corpus_manifest_sha256,
        metadata.corpus_file_sha256,
        *metadata.source_sha256.values(),
    ]
    if any(not re.fullmatch("[0-9a-f]{64}", value) for value in hashes):
        raise StorageError("Provenance checksums must be SHA-256 hex digests.")
    for index in metadata.index_manifests:
        if (
            index.corpus_version != metadata.corpus_version
            or index.corpus_manifest_sha256 != metadata.corpus_manifest_sha256
            or index.corpus_file_sha256 != metadata.corpus_file_sha256
        ):
            raise StorageError("Index provenance does not match the run corpus.")
    return metadata


def validate_question(
    event: InstrumentationEvent,
    *,
    retriever_version: str,
    retrieval_depth: int,
) -> QuestionTrace:
    trace = parse(QuestionTrace, event.payload)
    if event.context.question_id != trace.question.question_id:
        raise StorageError("Event question identity does not match its trace.")
    if (trace.status == "failed") != (trace.error_type is not None):
        raise StorageError("Question status and error classification disagree.")
    if trace.status == "completed":
        result = trace.retrieval
        if (
            result is None
            or result.query.question_id != trace.question.question_id
            or result.query.text != trace.question.text
            or result.retriever_version != retriever_version
        ):
            raise StorageError("Successful trace requires matching canonical retrieval output.")
        ranks = [hit.rank for hit in result.results]
        ids = [hit.chunk.chunk_id for hit in result.results]
        if (
            ranks != list(range(1, len(ranks) + 1))
            or len(ids) != len(set(ids))
            or len(ids) > retrieval_depth
        ):
            raise StorageError("Successful retrieval ranking violates the canonical contract.")
    return trace


def validate_summary(
    event: InstrumentationEvent,
    run: dict[str, Any],
    metrics: list[dict[str, float]],
) -> RunSummary:
    summary = parse(RunSummary, event.payload)
    if event.payload["metadata"] != run["metadata_payload"]:
        raise StorageError("Final metadata differs from the registered original metadata.")
    planned, completed, failed = (
        run["planned_questions"],
        run["completed_questions"],
        run["failed_questions"],
    )
    if (
        summary.planned_questions != planned
        or summary.completed_questions != completed
        or summary.failed_questions != failed
        or summary.unattempted_questions != planned - completed - failed
    ):
        raise StorageError("Final counts disagree with stored question outcomes.")
    if summary.ended_at < summary.started_at:
        raise StorageError("Run end precedes its start.")
    if summary.status == "completed" and (
        failed or completed != planned or summary.error_type is not None
    ):
        raise StorageError("Completed run must contain only successful planned questions.")
    if summary.status == "failed" and (summary.error_type is None or not (failed or not metrics)):
        raise StorageError("Failed run must contain a question failure or a warmup failure.")
    config = summary.metadata.configuration
    if not 0 <= summary.warmup_completed <= config.warmup_questions:
        raise StorageError("Warmup count is inconsistent with configuration.")
    if (summary.status == "completed" or metrics) and (
        summary.warmup_completed != config.warmup_questions
    ):
        raise StorageError("Scored attempts require completed warmups.")
    full = (
        summary.status == "completed" and completed == summary.metadata.full_dataset_question_count
    )
    if summary.full_benchmark != full:
        raise StorageError("Full-benchmark flag disagrees with the recorded selection/outcomes.")
    # Validate aggregation of recorded numbers, not metric definitions or retrieval quality.
    names = {name for row in metrics for name in row}
    if names != set(summary.metrics) or names != set(summary.metric_denominators):
        raise StorageError("Summary metric names disagree with the stored question metrics.")
    for name in names:
        values = [row[name] for row in metrics if name in row]
        if summary.metric_denominators[name] != len(values) or not math.isclose(
            summary.metrics[name], math.fsum(values) / len(values), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise StorageError("Aggregate metric or denominator disagrees with question metrics.")
    return summary
