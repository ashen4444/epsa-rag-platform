# Phase 5B observability API

This package provides a small read-only FastAPI interface over Phase 5A PostgreSQL experiment
storage. It never runs research code, delivers instrumentation events, migrates a schema, or
mutates experiment records.

Install the storage and API packages from the repository root:

```powershell
python -m pip install -e ".[dev]" -e ./observability/database -e ./observability/api
```

Set `EPSA_DATABASE_URL` to the Phase 5A PostgreSQL URL, migrate it separately with
`epsa-experiments migrate`, then start the API:

```powershell
epsa-observability-api
```

The API listens on `127.0.0.1:8000` by default. `GET /health` is a liveness check and does not
connect to PostgreSQL. `GET /health/database` verifies connectivity and the exact Phase 5A schema
revision. All experiment endpoints are under `/api/v1`.

For a stronger operational boundary, configure `EPSA_DATABASE_URL` with a PostgreSQL role that has
only `USAGE` on `epsa_experiments` and `SELECT` on its tables. The application itself contains no
write statements or write endpoints.
