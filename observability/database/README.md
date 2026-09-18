# Phase 5A experiment storage

PostgreSQL stores reproducible runs and full per-question traces. This optional package depends
on the research contracts; the research package never imports it. FastAPI, dashboard, research
algorithms, benchmark data, indexes, and the existing evaluator CLI are unchanged.

## Install and start locally

Use Python 3.12 and run from the repository root in an activated environment:

```powershell
python -m pip install -e ".[dev]" -e ./observability/database
# Generate a local password once. Save it securely for later sessions.
$env:EPSA_POSTGRES_PASSWORD = [guid]::NewGuid().ToString('N')
docker compose -f observability/database/compose.yaml up -d --wait
$env:EPSA_DATABASE_URL = "postgresql+psycopg://epsa:$env:EPSA_POSTGRES_PASSWORD@localhost:55432/epsa_experiments"
epsa-experiments migrate
epsa-experiments check
```

Docker Desktop must be running with Linux containers. The PostgreSQL 18 service binds only to
localhost and persists data in a named volume. Changing the environment password does not change
an already initialized database password. Keep using your original password on later starts.
An existing PostgreSQL installation can be used instead by setting `EPSA_DATABASE_URL`.
Neither this CLI nor the research CLI automatically loads `.env` files. Never commit credentials.

Migrations are explicit, transactional and serialized. Opening a repository checks the schema
revision; it never creates or upgrades tables. The initial downgrade refuses to remove a database
containing runs. The storage CLI intentionally exposes no delete/reset/downgrade command.

## Import and inspect existing results

```powershell
epsa-experiments import 'E:\EPSA RAG Final Implementation\epsa-rag-platform\data\exports\retrieval\RUN_ID' --benchmark-role development
epsa-experiments list
epsa-experiments list --status running
epsa-experiments inspect RUN_ID
epsa-experiments inspect RUN_ID --question-id QUESTION_ID
```

Use `test` for the held-out hard benchmark. The documented dataset/corpus pair is checked against
the role. `diagnostic` is reserved for synthetic/custom datasets; frozen benchmark versions cannot
be relabelled diagnostic. Dirty-code provenance is preserved even on a development/test run;
database completion is not research acceptance.

Import requires finalized `manifest.json`, `run.json` and `events.jsonl`, including finalized failed
runs. It validates file names, checksums, sizes, record counts, versions, lifecycle, question order,
configuration fingerprint, counts and aggregate denominators/values. Events stream into one
transaction, so corrupt/truncated exports leave no partial import. Full trace bodies stream one at
a time; metadata, event IDs and the small metric projections are retained for validation.
No dataset or index is needed,
and no retrieval or embedding is performed. Sources are read only. Repeat import of exactly the
same events is safe. Conflicting identities are rejected, never overwritten.

Original payloads remain the authoritative scientific records. Query columns are projections,
written in the same transaction. Older evaluator v1/v2/v3 payloads retain their metric names and
absent timing/cache fields; parsing defaults are never written back into those payloads. Numerical
aggregation is checked against recorded question values, not recalculated from rankings. This is
storage validation, not fresh evaluation or acceptance of reported results.

## Optional live instrumentation

```python
import os
from epsa_observability.connection import create_engine
from epsa_observability.repository import ExperimentRepository
from epsa_observability.sink import PostgresInstrumentationSink

engine = create_engine(os.environ["EPSA_DATABASE_URL"])
try:
    sink = PostgresInstrumentationSink(ExperimentRepository(engine), benchmark_role="development")
    # Pass downstream=sink to the existing run_benchmark(...) Python entry point.
    # All benchmark paths, run ID, and research configuration must still be explicit.
finally:
    engine.dispose()
```

The sink is never enabled automatically. It implements the existing synchronous `emit(event)`
protocol. The start event registers the run, question events persist results, and the finish event
finalizes it. `register_run(metadata_payload, benchmark_role=...)` can reserve the same identity
before execution. Components are currently `{}` because no EPSA component has been implemented.

Database delivery errors raise a sanitized `InstrumentationError` subclass. They are not swallowed
or turned into scientific failure summaries. A run without a finish event remains `running`
(incomplete); missing question results stay `pending`. The database may be ahead of an unfinished
local export if its finish commit succeeded before local finalization failed. There is no distributed
transaction between PostgreSQL and export files. Exact known events may be retried; unfinished
export recovery and resuming research execution are not implemented. A new execution needs a new
run ID. A finalized export can reconcile the same already persisted events safely.

Synchronous delivery is outside query-latency measurements but inside the evaluation-loop wall time
and therefore affects run throughput. Storage mode, adapter source fingerprint and library versions
are recorded separately from unchanged research provenance. Imported timings retain their original
measurement conditions; importing does not make them database-instrumented timings.

## Schema and research boundaries

The dedicated `epsa_experiments` schema contains three tables plus Alembic's revision table:

* `runs`: immutable original metadata, benchmark role, component versions, first-ingestion
  provenance, lifecycle, progress counts and original final summary (including aggregate metrics).
* `run_questions`: ordered planned IDs, pending/completed/failed status, metric and timing projections.
* `trace_events`: immutable event envelopes, original correlation IDs/timestamps, ingestion sequence
  and timestamp, and a canonical SHA-256. Full rankings, native sentences and evaluation labels are
  retained here, not duplicated into a table per algorithm stage.

Identifiers and common filters use ordinary indexes. Flexible research payloads use JSONB; text
values and array order survive round trips, while JSON formatting/key order is not archival byte
identity. Original export byte hashes are recorded separately. `started_event_at` is the start
event timestamp; exact `started_at` comes from the final summary, never an invented precision.

Run row locks serialize delivery within a run. Independent runs can progress concurrently. Exact
event replay is idempotent, including after completion; a reused ID with changed content fails.
Foreign keys and uniqueness/check constraints protect identity and basic consistency. The repository
exposes no mutation of finalized results; database owners still have administrative SQL privileges.
Treat direct SQL edits as unsupported. Back up PostgreSQL regularly (for example with `pg_dump`)
and keep the original immutable diagnostic exports. `docker compose down` preserves the volume;
do not use `down -v` on a database containing research results.

Known retrieval lifecycle events are validated against existing research contracts. Other diagnostic
events can be retained for a running registered run/question, but future component evaluation
registration/finalization needs a versioned adapter when those contracts actually exist. No future
EPSA schemas, workers, queues, partitions, API servers or dashboard components are prebuilt.

## Verification

```powershell
python -m pytest
python -m ruff check .
python -m mypy
python -m mypy --config-file observability/database/pyproject.toml observability/database/src/epsa_observability
# Use only a disposable database named with a _test suffix; tests recreate its EPSA schema.
$env:EPSA_TEST_DATABASE_URL = 'postgresql+psycopg://postgres:YOUR_TEST_PASSWORD@localhost:55439/epsa_test'
python -m pytest -c observability/database/pyproject.toml observability/database/tests --cov=epsa_observability --cov-report=term-missing
```

Without `EPSA_TEST_DATABASE_URL`, PostgreSQL integration tests explicitly skip; that is not a
successful database validation. Tests exercise real PostgreSQL migrations, round trips, concurrent
registration/delivery, atomic imports, replay/conflicts, failed/partial runs and schema guards.

For an additional 10,000-question synthetic import check, set `EPSA_RUN_SCALE_TEST=1` and rerun
the storage suite with `-s`. This uses generated fixture traces, never a real benchmark, and is
disabled by default to keep routine checks fast. Its timing measures import overhead on that test
machine, not retrieval performance or production capacity. The CI workflow runs the research suite
and PostgreSQL storage suite with a 95% coverage gate for each.

See [VALIDATION.md](VALIDATION.md) for the implementation's verified results and local runtime notes.
