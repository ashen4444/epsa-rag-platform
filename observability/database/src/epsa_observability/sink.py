"""Opt-in adapter for the existing synchronous research instrumentation protocol."""

from epsa_rag.instrumentation.events import InstrumentationEvent

from epsa_observability.repository import ExperimentRepository
from epsa_observability.validation import BenchmarkRole


class PostgresInstrumentationSink:
    """Acknowledge only committed events. The caller owns repository/engine lifetime."""

    def __init__(
        self,
        repository: ExperimentRepository,
        *,
        benchmark_role: BenchmarkRole,
        component_versions: dict[str, str] | None = None,
    ) -> None:
        self.repository = repository
        self.benchmark_role = benchmark_role
        self.component_versions = dict(component_versions or {})

    def emit(self, event: InstrumentationEvent) -> None:
        self.repository.ingest_event(
            event, benchmark_role=self.benchmark_role, component_versions=self.component_versions
        )
