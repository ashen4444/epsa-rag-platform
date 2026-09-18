"""Alembic uses the caller's connection; credentials never enter config or logs."""

from alembic import context

connection = context.config.attributes["connection"]
context.configure(
    connection=connection,
    version_table_schema="epsa_experiments",
    transactional_ddl=True,
)
with context.begin_transaction():
    context.run_migrations()
