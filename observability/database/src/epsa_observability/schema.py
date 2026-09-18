"""Query metadata. Migration revisions independently freeze each schema version."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

SCHEMA = "epsa_experiments"
REVISION = "0001_experiments"
metadata = sa.MetaData(schema=SCHEMA)

runs = sa.Table(
    "runs",
    metadata,
    sa.Column("run_id", sa.Text, primary_key=True),
    sa.Column("experiment_type", sa.Text, nullable=False),
    sa.Column("benchmark_role", sa.Text, nullable=False),
    sa.Column("dataset_version", sa.Text, nullable=False),
    sa.Column("corpus_version", sa.Text, nullable=False),
    sa.Column("git_commit_sha", sa.Text, nullable=False),
    sa.Column("git_dirty", sa.Boolean, nullable=False),
    sa.Column("retriever_version", sa.Text, nullable=False),
    sa.Column("evaluator_version", sa.Text, nullable=False),
    sa.Column("retrieval_depth", sa.Integer, nullable=False),
    sa.Column("configuration_fingerprint", sa.Text, nullable=False),
    sa.Column("metadata_payload", JSONB, nullable=False),
    sa.Column("component_versions", JSONB, nullable=False),
    sa.Column("storage_provenance", JSONB, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("started_event_at", sa.DateTime(timezone=True)),
    sa.Column("started_at", sa.DateTime(timezone=True)),
    sa.Column("ended_at", sa.DateTime(timezone=True)),
    sa.Column("planned_questions", sa.Integer, nullable=False),
    sa.Column("attempted_questions", sa.Integer, nullable=False),
    sa.Column("completed_questions", sa.Integer, nullable=False),
    sa.Column("failed_questions", sa.Integer, nullable=False),
    sa.Column("event_count", sa.Integer, nullable=False),
    sa.Column("summary_payload", JSONB),
)

run_questions = sa.Table(
    "run_questions",
    metadata,
    sa.Column("run_id", sa.Text, sa.ForeignKey(f"{SCHEMA}.runs.run_id"), primary_key=True),
    sa.Column("question_id", sa.Text, primary_key=True),
    sa.Column("position", sa.Integer, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("error_type", sa.Text),
    sa.Column("metrics", JSONB),
    sa.Column("latency_ms", sa.Double),
    sa.Column("retrieval_core_latency_ms", sa.Double),
)

trace_events = sa.Table(
    "trace_events",
    metadata,
    sa.Column("event_id", sa.Text, primary_key=True),
    sa.Column("run_id", sa.Text, sa.ForeignKey(f"{SCHEMA}.runs.run_id"), nullable=False),
    sa.Column("question_id", sa.Text),
    sa.Column("trace_id", sa.Text, nullable=False),
    sa.Column("event_type", sa.Text, nullable=False),
    sa.Column("source", sa.Text, nullable=False),
    sa.Column("source_version", sa.Text),
    sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("sequence", sa.Integer, nullable=False),
    sa.Column("content_sha256", sa.Text, nullable=False),
    sa.Column("envelope", JSONB, nullable=False),
    sa.ForeignKeyConstraint(
        ["run_id", "question_id"],
        [f"{SCHEMA}.run_questions.run_id", f"{SCHEMA}.run_questions.question_id"],
    ),
)
