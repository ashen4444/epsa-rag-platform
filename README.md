# EPSA-RAG Platform

EPSA-RAG is a Python 3.12 research platform for developing and evaluating the Evidence Path
Sufficiency Algorithm. The repository will contain a research system and a logically separate
observability application. The research package must remain independent of web and database
frameworks.

## Current scope

The Phase 1 foundation, Phase 2 deterministic data pipeline, and Phase 3 Hybrid Retriever exist:

- validated, serialization-friendly structural and retrieval contracts;
- stable identifier and configuration fingerprint helpers;
- an exception hierarchy;
- instrumentation event, context, and sink contracts;
- deterministic HotPotQA question selection and global corpus construction;
- sentence-preserving benchmark records with evaluation-only gold labels;
- immutable, checksummed dataset and corpus manifests;
- deterministic Okapi BM25 retrieval;
- OpenAI `text-embedding-3-small` document and query embedding adapters;
- exact cosine retrieval using normalized FAISS `IndexFlatIP` vectors;
- weighted Reciprocal Rank Fusion with canonical ranked paragraph outputs;
- immutable, corpus-bound BM25 and dense index manifests;
- test, coverage, lint, and type-check configuration.

Retriever evaluation, the observability backend, EPSA components, and RAG pipelines are
intentionally not implemented yet.

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

## Prepare the Phase 2 benchmark

The official HotPotQA distractor development source is downloaded into ignored local storage. The
derived JSONL files are also ignored, while their reproducibility manifests remain trackable.

```powershell
epsa-prepare-hotpotqa --download
```

The command creates `data/datasets/hotpotqa_1000_v1/` and
`data/corpus/hotpotqa_10000_v1/`. It refuses to overwrite either version.

## Build and inspect the Phase 3 retriever

Set `OPENAI_API_KEY` in the process environment before building the paid dense embeddings. A fresh
workspace can build both index branches together:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
epsa-build-retrieval-indexes --kind all
```

Each index version is immutable. If the BM25 index already exists, build only the missing dense
index:

```powershell
epsa-build-retrieval-indexes --kind dense
```

Run one inspectable hybrid query after both indexes exist:

```powershell
epsa-retrieve "Operation Cold Comfort was a failed raid by which special forces unit?"
```

Index binaries and ID mappings are ignored by Git. Their small checksummed manifests remain
trackable.

Repository-wide development and research rules are defined in `AGENTS.md` and the current
documents under `docs/architecture/`.
