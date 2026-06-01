"""Experiment 1: Direct Proximity Probing.

Tests whether post-retrieval namespace filtering leaks behavioral signals.
Crafts queries semantically similar to victim tenant's content and observes:
- Similarity scores (do they suggest proximity to victim content?)
- Response latency (does it increase when HNSW traverses into victim's region?)
- Result count anomalies (fewer results than expected = filtering happened)
"""

import time
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress

from vectorbleed.config import Settings
from vectorbleed.databases.base import TenantConfig, VectorDatabaseAdapter
from vectorbleed.embeddings import EmbeddingEngine
from vectorbleed.experiments.base import BaseExperiment, ExperimentResult, ExperimentSummary

console = Console()

# Probes designed to target healthcare content (victim tenant B)
CROSS_TENANT_PROBES = [
    "What are the clinical protocols for ICU patient admission?",
    "Describe the HIPAA compliance requirements for patient data",
    "What oncology drugs are in the current formulary?",
    "Explain the surgical consent form procedures",
    "What is the infection control policy for MRSA isolation?",
    "Describe the emergency department triage protocol",
    "What are the pharmacy prior authorization criteria?",
    "Explain the telemedicine remote patient monitoring protocol",
    "What is the organ transplant donor matching algorithm?",
    "Describe the palliative care end-of-life decision framework",
    "What are the ventilator weaning protocol steps?",
    "Explain the diabetes management insulin titration process",
    "What is the blood transfusion reaction protocol?",
    "Describe the NICU admission criteria and scoring system",
    "What are the opioid prescribing guidelines for pain management?",
    "Explain the antibiotic stewardship program requirements",
    "What is the fall prevention assessment tool for geriatrics?",
    "Describe the concussion return-to-play protocol",
    "What are the anaphylaxis emergency response procedures?",
    "Explain the chronic wound assessment and treatment protocol",
]

# Control probes — should match attacker's own financial content
SAME_TENANT_PROBES = [
    "What are the Q3 earnings projections?",
    "Describe the portfolio rebalancing strategy",
    "What is the merger arbitrage position?",
    "Explain the credit default swap exposure",
    "What are the regulatory capital requirements?",
]


class ProximityProbingExperiment(BaseExperiment):
    """Experiment 1: Direct Proximity Probing Attack."""

    @property
    def name(self) -> str:
        return "Direct Proximity Probing"

    @property
    def experiment_id(self) -> str:
        return "exp1_proximity_probing"

    @property
    def description(self) -> str:
        return (
            "Tests whether namespace filtering leaks behavioral signals by crafting "
            "queries semantically similar to victim tenant content and observing "
            "similarity scores, latency patterns, and result count anomalies."
        )

    def run(self) -> ExperimentSummary:
        """Execute proximity probing experiment."""
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Experiment 1: {self.name}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        start_time = time.time()
        self.results = []

        # Phase 1: Baseline — query attacker's own content
        console.print("[bold]Phase 1: Baseline (same-tenant queries)[/bold]")
        baseline_latencies = []
        baseline_scores = []

        for i, probe in enumerate(SAME_TENANT_PROBES):
            query_emb = self.embedding_engine.embed_text(probe)
            results, latency = self.db.query_cross_namespace_timed(
                self.attacker, query_emb, top_k=10
            )
            baseline_latencies.append(latency * 1000)
            if results:
                baseline_scores.append(results[0].score)

        avg_baseline_latency = np.mean(baseline_latencies) if baseline_latencies else 0
        avg_baseline_score = np.mean(baseline_scores) if baseline_scores else 0

        console.print(f"  Baseline avg latency: {avg_baseline_latency:.1f}ms")
        console.print(f"  Baseline avg top score: {avg_baseline_score:.4f}")

        # Phase 2: Cross-tenant probing
        console.print("\n[bold]Phase 2: Cross-tenant probing (targeting victim content)[/bold]")

        with Progress(console=console) as progress:
            task = progress.add_task("Probing...", total=len(CROSS_TENANT_PROBES))

            for i, probe in enumerate(CROSS_TENANT_PROBES):
                query_emb = self.embedding_engine.embed_text(probe)

                # Query from attacker's namespace
                results, latency = self.db.query_cross_namespace_timed(
                    self.attacker, query_emb, top_k=10
                )

                latency_ms = latency * 1000
                top_scores = [r.score for r in results[:5]]
                detected, cross_count = self._detect_cross_tenant(results)

                # Anomaly detection
                latency_anomaly = latency_ms > (avg_baseline_latency * 1.5)
                score_anomaly = top_scores and top_scores[0] < (avg_baseline_score * 0.5)
                result_count_anomaly = len(results) < 5  # fewer than expected

                self.results.append(
                    ExperimentResult(
                        probe_id=i,
                        probe_query=probe,
                        results_count=len(results),
                        top_scores=top_scores,
                        latency_ms=latency_ms,
                        cross_tenant_detected=detected,
                        cross_tenant_count=cross_count,
                        metadata={
                            "latency_anomaly": latency_anomaly,
                            "score_anomaly": score_anomaly,
                            "result_count_anomaly": result_count_anomaly,
                            "baseline_latency_ms": avg_baseline_latency,
                            "baseline_score": avg_baseline_score,
                        },
                    )
                )

                progress.advance(task)

        end_time = time.time()
        summary = self._build_summary(start_time, end_time)

        # Add experiment-specific metadata
        summary.metadata = {
            "baseline_avg_latency_ms": avg_baseline_latency,
            "baseline_avg_score": avg_baseline_score,
            "latency_anomalies": sum(
                1 for r in self.results if r.metadata.get("latency_anomaly")
            ),
            "score_anomalies": sum(
                1 for r in self.results if r.metadata.get("score_anomaly")
            ),
            "result_count_anomalies": sum(
                1 for r in self.results if r.metadata.get("result_count_anomaly")
            ),
        }

        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: ExperimentSummary) -> None:
        """Print experiment results."""
        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print(f"[bold green]Results: {self.name}[/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]")
        console.print(f"  Total probes: {summary.total_probes}")
        console.print(f"  Cross-tenant detections: {summary.cross_tenant_detections}")
        console.print(f"  Detection rate: {summary.detection_rate:.2%}")
        console.print(f"  Avg latency: {summary.avg_latency_ms:.1f}ms")
        console.print(f"  Max similarity score: {summary.max_similarity_score:.4f}")
        console.print(f"  Latency anomalies: {summary.metadata.get('latency_anomalies', 0)}")
        console.print(f"  Score anomalies: {summary.metadata.get('score_anomalies', 0)}")
        console.print(f"  Result count anomalies: {summary.metadata.get('result_count_anomalies', 0)}")
        console.print(f"  Duration: {summary.duration_seconds:.1f}s")
