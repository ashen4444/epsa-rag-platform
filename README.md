# EPSA-RAG Platform

EPSA-RAG is a Python 3.12 research platform for developing and evaluating the Evidence Path
Sufficiency Algorithm. The repository will contain a research system and a logically separate
observability application. The research package must remain independent of web and database
frameworks.

## Current scope

Only the Phase 1 software foundation exists:

- validated, serialization-friendly structural and retrieval contracts;
- stable identifier and configuration fingerprint helpers;
- an exception hierarchy;
- instrumentation event, context, and sink contracts;
- test, coverage, lint, and type-check configuration.

Dataset preparation, retrieval implementations, evaluation, the observability backend, EPSA
components, and RAG pipelines are intentionally not implemented yet.

## Development setup (PowerShell)

Install Python 3.12, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the quality checks:

```powershell
python -m pytest
python -m ruff check .
python -m mypy
```

Repository-wide development and research rules are defined in `AGENTS.md` and the current
documents under `docs/architecture/`.

