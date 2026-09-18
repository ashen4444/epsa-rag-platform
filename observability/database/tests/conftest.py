"""Synthetic research outputs and an explicitly disposable real PostgreSQL database."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import (
    BenchmarkExample,
    EvaluationLabels,
    QuestionInput,
    SupportingFactLabel,
)
from epsa_rag.evaluation.retrieval.evaluator import evaluate
from epsa_rag.evaluation.retrieval.models import EvaluationConfig, RunMetadata
from epsa_rag.instrumentation.sinks import InMemoryInstrumentationSink
from epsa_rag.retrieval.models import RetrievalResult

from epsa_observability.connection import create_engine
from epsa_observability.migrate import upgrade
from epsa_observability.repository import ExperimentRepository


def make_events(run_id="storage-test", *, fail_at=None, limit=None, warmups=0):
    chunks = tuple(
        ParagraphChunk(
            chunk_id=f"chunk-{i}",
            title=f"Title {i}",
            paragraph_text=" First native sentence. \nSecond.",
            sentences=(
                Sentence(index=0, text=" First native sentence. \n"),
                Sentence(index=1, text="Second."),
            ),
        )
        for i in range(2)
    )
    examples = tuple(
        BenchmarkExample(
            inference=QuestionInput(question_id=f"q-{i}", text=f"  Question {i}?\n"),
            evaluation=EvaluationLabels(
                answer="EVALUATION ONLY",
                question_type="bridge",
                difficulty="hard",
                supporting_facts=tuple(
                    SupportingFactLabel(
                        chunk_id=chunk.chunk_id,
                        title=chunk.title,
                        sentence_index=0,
                        evidence_unit_id=f"{chunk.chunk_id}::s0",
                    )
                    for chunk in chunks
                ),
            ),
        )
        for i in range(2)
    )
    config = EvaluationConfig(question_limit=limit, warmup_questions=warmups)
    selected = examples if limit is None else examples[:limit]
    metadata = RunMetadata(
        run_id=run_id,
        git_commit_sha="a" * 40,
        git_dirty=False,
        source_sha256={"src/fixture.py": "b" * 64},
        runtime={"python": "3.12"},
        dataset_version="synthetic-dataset-v1",
        corpus_version="synthetic-corpus-v1",
        dataset_manifest_sha256="c" * 64,
        dataset_file_sha256="d" * 64,
        corpus_manifest_sha256="e" * 64,
        corpus_file_sha256="f" * 64,
        index_manifests=(),
        question_ids=tuple(item.inference.question_id for item in selected),
        full_dataset_question_count=len(examples),
        configuration=config,
        configuration_fingerprint=config.fingerprint(),
    )

    class FixtureRetriever:
        def retrieve(self, query, **kwargs):
            if query.question_id == fail_at:
                raise RuntimeError("sensitive error text must not be persisted")
            return RetrievalResult(
                query=query,
                retriever_version="hybrid-retriever-v2",
                results=tuple(
                    RankedParagraphChunk(
                        chunk=chunk,
                        rank=i + 1,
                        score=0.5,
                        source_scores={"bm25": 0.25},
                        source_ranks={"bm25": i + 1},
                    )
                    for i, chunk in enumerate(chunks)
                ),
            )

    sink = InMemoryInstrumentationSink()
    summary = evaluate(
        examples=selected,
        corpus=SimpleNamespace(by_id={c.chunk_id: c for c in chunks}),
        retriever=FixtureRetriever(),
        metadata=metadata,
        sink=sink,
    )
    return sink.events, summary


@pytest.fixture
def events():
    return make_events()[0]


@pytest.fixture
def database_engine():
    url = os.environ.get("EPSA_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set EPSA_TEST_DATABASE_URL to run real PostgreSQL integration tests")
    parsed = sa.engine.make_url(url)
    if not parsed.database or not parsed.database.endswith("_test"):
        pytest.fail("Integration tests require an explicitly disposable database ending in _test")
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(sa.text("DROP SCHEMA IF EXISTS epsa_experiments CASCADE"))
    try:
        yield engine
    finally:
        with engine.begin() as connection:
            connection.execute(sa.text("DROP SCHEMA IF EXISTS epsa_experiments CASCADE"))
        engine.dispose()


@pytest.fixture
def repository(database_engine):
    upgrade(database_engine)
    return ExperimentRepository(database_engine)
