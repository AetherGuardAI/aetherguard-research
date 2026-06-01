"""Base experiment class for VectorBleed attack experiments."""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from vectorbleed.config import Settings, get_settings
from vectorbleed.databases.base import TenantConfig, VectorDatabaseAdapter
from vectorbleed.embeddings import EmbeddingEngine

console = Console()


@dataclass
class ExperimentResult:
    """Result of a single experiment probe."""

    probe_id: int
    probe_query: str
    results_count: int
    top_scores: list[float]
    latency_ms: float
    cross_tenant_detected: bool
    cross_tenant_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ExperimentSummary:
    """Summary of an entire experiment run."""

    experiment_name: str
    experiment_id: str
    description: str
    total_probes: int
    cross_tenant_detections: int
    detection_rate: float
    avg_latency_ms: float
    max_similarity_score: float
    results: list[ExperimentResult]
    start_time: str
    end_time: str
    duration_seconds: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, output_dir: Path) -> Path:
        """Save experiment summary to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.experiment_id}.json"
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return output_path


class BaseExperiment(ABC):
    """Abstract base class for all VectorBleed experiments."""

    def __init__(
        self,
        db: VectorDatabaseAdapter,
        embedding_engine: EmbeddingEngine,
        attacker_tenant: TenantConfig,
        victim_tenant: TenantConfig,
        settings: Optional[Settings] = None,
    ):
        self.db = db
        self.embedding_engine = embedding_engine
        self.attacker = attacker_tenant
        self.victim = victim_tenant
        self.settings = settings or get_settings()
        self.results: list[ExperimentResult] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable experiment name."""
        ...

    @property
    @abstractmethod
    def experiment_id(self) -> str:
        """Unique experiment identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Experiment description."""
        ...

    @abstractmethod
    def run(self) -> ExperimentSummary:
        """Execute the experiment and return summary."""
        ...

    def _detect_cross_tenant(self, results: list) -> tuple[bool, int]:
        """Check if any results belong to the victim tenant."""
        cross_tenant_count = 0
        for result in results:
            metadata = result.metadata or {}
            result_tenant = metadata.get("tenant_id", "")
            if result_tenant == self.victim.tenant_id:
                cross_tenant_count += 1

        return cross_tenant_count > 0, cross_tenant_count

    def _build_summary(self, start_time: float, end_time: float) -> ExperimentSummary:
        """Build experiment summary from collected results."""
        all_scores = []
        for r in self.results:
            all_scores.extend(r.top_scores)

        cross_tenant_detections = sum(1 for r in self.results if r.cross_tenant_detected)

        return ExperimentSummary(
            experiment_name=self.name,
            experiment_id=self.experiment_id,
            description=self.description,
            total_probes=len(self.results),
            cross_tenant_detections=cross_tenant_detections,
            detection_rate=cross_tenant_detections / max(len(self.results), 1),
            avg_latency_ms=sum(r.latency_ms for r in self.results) / max(len(self.results), 1),
            max_similarity_score=max(all_scores) if all_scores else 0.0,
            results=self.results,
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            end_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time)),
            duration_seconds=end_time - start_time,
        )
