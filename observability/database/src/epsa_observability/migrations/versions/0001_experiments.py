"""Initial experiment storage. Frozen DDL; do not import application metadata here."""

from alembic import op

revision = "0001_experiments"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE epsa_experiments.runs (
            run_id text PRIMARY KEY CHECK (run_id ~ '^[a-z0-9][a-z0-9_-]{0,99}$'),
            experiment_type text NOT NULL,
            benchmark_role text NOT NULL
                CHECK (benchmark_role IN ('development', 'test', 'diagnostic')),
            dataset_version text NOT NULL,
            corpus_version text NOT NULL,
            git_commit_sha text NOT NULL CHECK (git_commit_sha ~ '^[0-9a-f]{40,64}$'),
            git_dirty boolean NOT NULL,
            retriever_version text NOT NULL,
            evaluator_version text NOT NULL,
            retrieval_depth integer NOT NULL CHECK (retrieval_depth > 0),
            configuration_fingerprint text NOT NULL
                CHECK (configuration_fingerprint ~ '^[0-9a-f]{64}$'),
            metadata_payload jsonb NOT NULL CHECK (jsonb_typeof(metadata_payload) = 'object'),
            component_versions jsonb NOT NULL CHECK (jsonb_typeof(component_versions) = 'object'),
            storage_provenance jsonb NOT NULL CHECK (jsonb_typeof(storage_provenance) = 'object'),
            status text NOT NULL CHECK (status IN ('registered', 'running', 'completed', 'failed')),
            registered_at timestamptz NOT NULL,
            started_event_at timestamptz,
            started_at timestamptz,
            ended_at timestamptz,
            planned_questions integer NOT NULL CHECK (planned_questions > 0),
            attempted_questions integer NOT NULL DEFAULT 0 CHECK (attempted_questions >= 0),
            completed_questions integer NOT NULL DEFAULT 0 CHECK (completed_questions >= 0),
            failed_questions integer NOT NULL DEFAULT 0 CHECK (failed_questions >= 0),
            event_count integer NOT NULL DEFAULT 0 CHECK (event_count >= 0),
            summary_payload jsonb CHECK (jsonb_typeof(summary_payload) = 'object'),
            CHECK (attempted_questions = completed_questions + failed_questions),
            CHECK (attempted_questions <= planned_questions),
            CHECK ((status IN ('completed', 'failed')) = (summary_payload IS NOT NULL)),
            CHECK ((status IN ('completed', 'failed')) = (ended_at IS NOT NULL)),
            CHECK (ended_at IS NULL OR (started_at IS NOT NULL AND ended_at >= started_at))
        )
    """)
    op.execute("""
        CREATE INDEX ix_runs_benchmark ON epsa_experiments.runs
            (benchmark_role, dataset_version, corpus_version, registered_at)
    """)
    op.execute("""
        CREATE INDEX ix_runs_status ON epsa_experiments.runs (status, registered_at)
    """)
    op.execute("""
        CREATE TABLE epsa_experiments.run_questions (
            run_id text NOT NULL REFERENCES epsa_experiments.runs(run_id),
            question_id text NOT NULL,
            position integer NOT NULL CHECK (position >= 0),
            status text NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
            error_type text,
            metrics jsonb CHECK (jsonb_typeof(metrics) = 'object'),
            latency_ms double precision CHECK (
                latency_ms >= 0 AND latency_ms < 'Infinity'::double precision),
            retrieval_core_latency_ms double precision CHECK (
                retrieval_core_latency_ms >= 0
                AND retrieval_core_latency_ms < 'Infinity'::double precision),
            PRIMARY KEY (run_id, question_id),
            UNIQUE (run_id, position),
            CHECK ((status = 'pending') = (metrics IS NULL)),
            CHECK ((status = 'pending') = (latency_ms IS NULL)),
            CHECK ((status = 'failed') = (error_type IS NOT NULL))
        )
    """)
    op.execute("""
        CREATE INDEX ix_questions_status ON epsa_experiments.run_questions
            (run_id, status, position)
    """)
    op.execute("""
        CREATE TABLE epsa_experiments.trace_events (
            event_id text PRIMARY KEY,
            run_id text NOT NULL REFERENCES epsa_experiments.runs(run_id),
            question_id text,
            trace_id text NOT NULL,
            event_type text NOT NULL,
            source text NOT NULL,
            source_version text,
            occurred_at timestamptz NOT NULL,
            ingested_at timestamptz NOT NULL,
            sequence integer NOT NULL CHECK (sequence > 0),
            content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
            envelope jsonb NOT NULL CHECK (jsonb_typeof(envelope) = 'object'),
            UNIQUE (run_id, sequence),
            FOREIGN KEY (run_id, question_id)
                REFERENCES epsa_experiments.run_questions(run_id, question_id)
        )
    """)
    op.execute("""
        CREATE INDEX ix_events_question ON epsa_experiments.trace_events
            (run_id, question_id, sequence)
    """)
    op.execute("""
        CREATE UNIQUE INDEX uq_run_lifecycle ON epsa_experiments.trace_events (run_id, event_type)
        WHERE event_type IN ('evaluation.run.started', 'evaluation.run.finished')
    """)
    op.execute("""
        CREATE UNIQUE INDEX uq_question_result ON epsa_experiments.trace_events
            (run_id, question_id)
        WHERE event_type = 'evaluation.question.completed'
    """)


def downgrade() -> None:
    # Downgrade is for an empty development database only. Never erase experiment history.
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM epsa_experiments.runs) THEN
                RAISE EXCEPTION 'Cannot downgrade a database containing experiment runs';
            END IF;
        END $$
    """)
    op.drop_table("trace_events", schema="epsa_experiments")
    op.drop_table("run_questions", schema="epsa_experiments")
    op.drop_table("runs", schema="epsa_experiments")
