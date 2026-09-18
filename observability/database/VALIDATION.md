# Phase 5A validation — 2026-09-18

Implementation base: `0d6b6d6199c883b512378b16e2e43dcffbb92e89`, verified against local main
and live remote main before implementation. Instructions and private architecture/protocol
documents were read directly from the E: source checkout because they are Git-ignored.

## Verified results

| Check | Result |
| --- | --- |
| Existing research suite | 224 passed; 95.41% coverage |
| New storage suite, including synthetic scale check | 83 passed; 97.83% branch-inclusive coverage |
| Final Docker-backed storage suite | 83 passed; 1 optional scale test skipped |
| Repository Ruff checks | Passed |
| Existing research strict mypy checks | Passed, 49 source files |
| Storage strict mypy checks | Passed, 12 source files |
| Built wheel | Contains migration environment and initial revision |
| Protected source scope | No changes to research `src/`, existing `tests/`, `data/`, or root `pyproject.toml` |

The storage tests used actual PostgreSQL 18.6 on Windows and Python 3.12.10, with SQLAlchemy
2.0.54, Psycopg 3.3.5, and Alembic 1.20.0. Both suites meet the 95% coverage gate. The new CI
workflow repeats the standard checks with PostgreSQL 18; it has been authored but not remotely run.

Database tests exercised migration upgrade/repeat/downgrade on an empty schema; refusal to
downgrade populated storage; incompatible schema detection; lifecycle and provenance validation;
concurrent run registration and duplicate delivery; full trace/text round trips; rollback after a
database write failure; atomic corrupt-export rejection; immutable replay/conflicts; failed warmups,
failed questions and subsets; evaluator v1/v2/v3 payload preservation; and CLI migration, import,
listing and inspection.

## Synthetic scale check

An entirely synthetic export containing 10,000 questions, 10,002 events and 22,559,395 bytes in
`events.jsonl` imported in 138.91 seconds under coverage instrumentation. Each question had two
small canonical paragraph results. The final question and completed counts were verified. This
checks registration/ingestion at the required question count; it is not a production capacity
measurement, a realistic Top-20 payload-size benchmark, or a retrieval-quality result.

Full trace bodies stream individually. Question delivery avoids repeatedly loading the complete
run manifest, while finalization reads the small per-question metric projections to verify summary
aggregation. No real development or held-out experiment was run or imported for this validation.

## Local runtime note

Docker Desktop was subsequently started manually. The supplied Compose configuration started
PostgreSQL 18 on loopback port 55432, and the Phase 5A migration and schema check both passed.
The complete storage suite then ran against a separate disposable `epsa_phase5a_test` database in
that container: 83 tests passed and the opt-in synthetic scale test was skipped. A completed
development export was imported to verify the Docker-backed workflow and then removed, leaving the
durable `epsa_experiments` database schema ready but empty. An existing native PostgreSQL
installation can also be used with `EPSA_DATABASE_URL`.

No persistent research database was provisioned and no live sink was enabled. Setup and operational
instructions are in [README.md](README.md). Changes are confined to the storage package, its tests,
the verification workflow, and the repository README.
