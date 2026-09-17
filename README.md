# EPSA-RAG Platform

EPSA-RAG is a Python 3.12 research platform for developing and evaluating the Evidence Path
Sufficiency Algorithm. The repository will contain a research system and a logically separate
observability application. The research package must remain independent of web and database
frameworks.

## Current scope

The Phase 1 foundation, Phase 2 data pipeline, Phase 3 retriever, and Phase 4 evaluator exist:

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
- exact-paragraph retrieval metrics, serial latency/throughput measurements, and benchmark runs;
- reproducible run metadata, per-question traces, immutable diagnostic exports, and paired comparisons;
- test, coverage, lint, and type-check configuration.

The observability backend, EPSA components, and RAG pipelines are
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

The same pipeline has a separate, immutable profile for the held-out hard-question test
benchmark. It filters the official HotPotQA training source to hard questions and selects 10,000
question IDs by the same seeded SHA-256 ranking method. Do not generate this benchmark until the
development configuration is frozen.

```powershell
epsa-prepare-hotpotqa --profile hard-test --download
```

This profile creates `data/datasets/hotpotqa_hard_10000_test_v1/` and
`data/corpus/hotpotqa_hard_10000_test_corpus_v1/`. Preparation does not create embeddings or
retrieval indexes.

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

## Evaluate the Phase 4 retriever

Refresh the editable installation, then commit the implementation before creating a research
artifact. The command does not automatically load `.env`.

```powershell
python -m pip install -e ".[dev]"
epsa-evaluate-retriever cache-queries
epsa-evaluate-retriever run --run-id retriever-hybrid-v1-eval-02 --query-embedding-cache read-only
```

`cache-queries` requires `OPENAI_API_KEY` and batches the exact 1,000 frozen questions once. It
publishes immutable, checksummed float32 vectors under the ignored `data/cache/` directory. A
read-only research run validates the complete frozen collection before creating its export and
never initializes an OpenAI client. Use `--query-embedding-cache disabled` for a live end-to-end
latency run. `read-write` reuses hits and embeds/stores misses, which supports future dynamic Hop-2
queries without putting EPSA logic into the retriever.

The default benchmark evaluates all 1,000 frozen questions against the global corpus. Relevance
requires the exact supporting chunk ID. The headline quality results are Recall@5, Recall@10,
both supporting paragraphs found@10, MRR@10, nDCG@10, and top-1 supporting-document hit rate.
Additional cutoffs and missing-evidence measures remain available for diagnosis. Recall measures
the fraction of gold paragraphs found: retrieving one of two supporting paragraphs gives 0.5.

Each aggregate includes its question denominator. Failed requests stop the run, remain visible,
and receive zero retrieval credit; unattempted questions are reported separately. End-to-end
p50/p95 includes the configured live embedding or cache lookup. Retrieval-core p50/p95 subtracts
that embedding boundary and measures BM25, FAISS, fusion, and canonical result construction.
Throughput is measured during serial execution, with a separate retrieval-core value.

For an offline BM25 development diagnostic while changes remain uncommitted:

```powershell
python -m epsa_rag.evaluation.retrieval.cli run --run-id retriever-bm25-v1-dev-01 --mode bm25 --allow-dirty-dev-run
```

Use `--mode dense` for a dense-only ablation, `--question-limit 10` for an explicit smoke-test
subset, and `--help` for timing and fusion options. Development and subset runs are marked in
their metadata. Each run ID is immutable and must be unique.

```powershell
epsa-evaluate-retriever inspect data/exports/retrieval/retriever-hybrid-v1-eval-01 --failures
epsa-evaluate-retriever inspect data/exports/retrieval/retriever-hybrid-v1-eval-01 --question-id QUESTION_ID
epsa-evaluate-retriever compare data/exports/retrieval/RUN_A data/exports/retrieval/RUN_B --metric recall@10
```

The ignored `data/exports/retrieval/{run_id}/` directory contains `run.json`, `events.jsonl`, and
a checksum manifest. Traces retain canonical rankings, branch scores/ranks, gold references, and
missing supporting chunk IDs. Paired comparisons list improved/regressed/unchanged questions.

These are Phase 4 diagnostic exports. PostgreSQL remains the planned authoritative experiment
store in Phase 5. A full hybrid benchmark and failure review are required before accepting
retriever quality; passing software tests alone does not establish retrieval accuracy.

Repository-wide development and research rules are defined in `AGENTS.md` and the current
documents under `docs/architecture/`.
