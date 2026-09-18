"""Explicit connections and migration checks; no automatic schema creation."""

import json
from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.exc import SQLAlchemyError

from epsa_observability.errors import StorageError
from epsa_observability.schema import REVISION, SCHEMA


def create_engine(database_url: str) -> Engine:
    """Use a small pool, bounded waits, and hidden SQL parameters."""
    try:
        url = make_url(database_url)
        if url.drivername not in {"postgresql", "postgresql+psycopg"}:
            raise StorageError("A PostgreSQL connection URL is required.")
        return sa.create_engine(
            url.set(drivername="postgresql+psycopg"),
            pool_size=2,
            max_overflow=0,
            pool_timeout=10,
            pool_pre_ping=True,
            hide_parameters=True,
            connect_args={"connect_timeout": 10, "options": "-c statement_timeout=60000"},
            json_serializer=lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False),
        )
    except (SQLAlchemyError, ValueError):
        raise StorageError("Invalid PostgreSQL connection configuration.") from None


@contextmanager
def transaction(engine: Engine) -> Iterator[Connection]:
    try:
        with engine.begin() as connection:
            yield connection
    except SQLAlchemyError:
        raise StorageError("Database operation failed; transaction was not acknowledged.") from None


def check_schema(engine: Engine) -> None:
    with transaction(engine) as connection:
        if not sa.inspect(connection).has_table("alembic_version", schema=SCHEMA):
            raise StorageError("Database schema is missing. Run epsa-experiments migrate first.")
        revisions = (
            connection.execute(sa.text(f"SELECT version_num FROM {SCHEMA}.alembic_version"))
            .scalars()
            .all()
        )
        if revisions != [REVISION]:
            raise StorageError(
                "Database schema revision is incompatible with this storage package."
            )
