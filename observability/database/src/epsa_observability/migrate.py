"""One explicit, serialized migration entry point."""

from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError
from sqlalchemy.engine import Engine

from epsa_observability.connection import check_schema, transaction
from epsa_observability.errors import StorageError


def migration_config(connection: sa.Connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    config.attributes["connection"] = connection
    return config


def upgrade(engine: Engine) -> None:
    with transaction(engine) as connection:
        # Serialize explicit migration commands, without locking ordinary research artifacts.
        connection.execute(sa.text("SELECT pg_advisory_xact_lock(512005001)"))
        connection.execute(sa.text("CREATE SCHEMA IF NOT EXISTS epsa_experiments"))
        try:
            command.upgrade(migration_config(connection), "head")
        except CommandError:
            raise StorageError("Migration history is incompatible with this package.") from None
    check_schema(engine)
