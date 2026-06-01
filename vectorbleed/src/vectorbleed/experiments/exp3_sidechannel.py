"""Experiment 3: Similarity Score Side-Channel Attack.

Uses similarity scores as a side channel to infer victim tenant's document content.
Tests whether information leaked in similarity scores (which tenants can legitimately
see) enables reconstruction of adjacent tenant content topics.

Method:
1. Probe with known vocabulary to map victim's content
2. Use score patterns to reconstruct victim's topic clusters
3. Progressive refinement to narrow in on content structure
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

# Phase 1: Broad vocabulary probes — healthcare domain keywords
HEALTHCARE_VOCABULARY = [
    "patient", "clinical", "drug", "formulary", "ICU", "HIPAA",
    "oncology", "consent", "protocol", "admission", "dosage",
    "surgery", "diagnosis", "treatment", "prescription", "radiology",
    "pathology", "anesthesia", "rehabilitation", "palliative",
    "ventilator", "dialysis", "transfusion", "neonatal", "geriatric",
    "cardiology", "neurology", "orthopedics", "dermatology", "urology",
    "endocrinology", "rheumatology", "nephrology", "hematology",
    "infectious disease", "antibiotic", "vaccination", "triage",
    "discharge", "nursing", "pharmacy", "laboratory",
]

# Phase 2: Refined probes — combining keywords for topic reconstruction
REFINED_PROBES = [
    "ICU admission protocol patient triage",
    "oncology drug formulary pricing review",
    "HIPAA patient data retention policy",
    "clinical trial enrollment criteria phase",
    "surgical consent forms standard procedure",
    "infection control MRSA isolation policy",
    "emergency department triage trauma level",
    "pharmacy benefit prior authorization criteria",
    "telemedicine remote patient monitoring protocol",
    "organ transplant donor matching algorithm",
    "palliative care end-of-life decision",
    "ventilator weaning protocol pulmonology",
    "diabetes insulin titration management",
    "blood transfusion reaction protocol",
    "NICU admission criteria neonatal scoring",
    "opioid prescribing guidelines pain management",
    "antibiotic stewardship program infectious",
    "fall prevention assessment geriatrics",
    "concussion return-to-play sports medicine",
    "anaphylaxis emergency response allergy",
]

# Control vocabulary — financial domain (should score HIGH from attacker's namespace)
FINANCIAL_VOCABULARY = [
    "portfolio", "earnings", "merger", "acquisition", "hedge fund",
    "derivatives", "equity", "bond", "dividend", "compliance",
]


class SideChannelExperiment(BaseExperiment):
    """Experiment 3: Similarity Score Side-Channel Attack."""

    @property
    def name(self) -> str:
        return "Similarity Score Side-Channel"

    @property
    def experiment_id(self) -> str:
        return "exp3_score_sidechannel"

    @property
    def description(self) -> str:
        return (
            "Uses similarity scores as a side channel to infer victim tenant's "
            "document content. Probes with healthcare vocabulary from the attacker's "
            "namespace and analyzes score patterns to reconstruct topic clusters."
        )

    def run(self) -> ExperimentSummary:
        """Execute side-channel experiment."""
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Experiment 3: {self.name}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        start_time = time.time()
        self.results = []

        # Phase 1: Establish baseline with financial vocabulary (attacker's own domain)
        console.print("[bold]Phase 1: Baseline — financial vocabulary scores[/bold]")
        financial_scores = {}

        for word in FINANCIAL_VOCABULARY:
            probe = f"Information about {word} in enterprise context"
            query_emb = self.embedding_engine.embed_text(probe)
            results, latency = self.db.query_cross_namespace_timed(
                self.attacker, query_emb, top_k=5
            )
            scores = [r.score for r in results]
            financial_scores[word] = {
                "max_score": max(scores) if scores else 0,
                "avg_score": np.mean(scores) if scores else 0,
                "result_count": len(results),
            }

        avg_financial_score = np.mean([v["max_score"] for v in financial_scores.values()])
        console.print(f"  Financial vocabulary avg max score: {avg_financial_score:.4f}")

        # Phase 2: Probe with healthcare vocabulary
        console.print("\n[bold]Phase 2: Healthcare vocabulary probing[/bold]")
        healthcare_scores = {}

        with Progress(console=console) as progress:
            task = progress.add_task("Probing healthcare terms...", total=len(HEALTHCARE_VOCABULARY))

            for i, word in enumerate(HEALTHCARE_VOCABULARY):
                probe = f"Information about {word} in enterprise context"
                query_emb = self.embedding_engine.embed_text(probe)
                results, latency = self.db.query_cross_namespace_timed(
                    self.attacker, query_emb, top_k=5
                )

                scores = [r.score for r in results]
                max_score = max(scores) if scores else 0
                detected, cross_count = self._detect_cross_tenant(results)

                healthcare_scores[word] = {
                    "max_score": max_score,
                    "avg_score": np.mean(scores) if scores else 0,
                    "result_count": len(results),
                    "latency_ms": latency * 1000,
                }

                # A healthcare term scoring high from financial namespace is anomalous
                score_elevation = max_score / avg_financial_score if avg_financial_score > 0 else 0

                self.results.append(
                    ExperimentResult(
                        probe_id=i,
                        probe_query=f"[vocab] {word}",
                        results_count=len(results),
                        top_scores=scores[:3],
                        latency_ms=latency * 1000,
                        cross_tenant_detected=detected,
                        cross_tenant_count=cross_count,
                        metadata={
                            "phase": "vocabulary_probe",
                            "word": word,
                            "score_elevation_ratio": score_elevation,
                            "baseline_financial_avg": avg_financial_score,
                        },
                    )
                )

                progress.advance(task)

        # Phase 3: Refined topic reconstruction probes
        console.print("\n[bold]Phase 3: Refined topic reconstruction[/bold]")

        with Progress(console=console) as progress:
            task = progress.add_task("Refined probing...", total=len(REFINED_PROBES))

            for i, probe in enumerate(REFINED_PROBES):
                query_emb = self.embedding_engine.embed_text(probe)
                results, latency = self.db.query_cross_namespace_timed(
                    self.attacker, query_emb, top_k=5
                )

                scores = [r.score for r in results]
                detected, cross_count = self._detect_cross_tenant(results)

                self.results.append(
                    ExperimentResult(
                        probe_id=len(HEALTHCARE_VOCABULARY) + i,
                        probe_query=f"[refined] {probe}",
                        results_count=len(results),
                        top_scores=scores[:3],
                        latency_ms=latency * 1000,
                        cross_tenant_detected=detected,
                        cross_tenant_count=cross_count,
                        metadata={
                            "phase": "refined_probe",
                            "probe_text": probe,
                        },
                    )
                )

                progress.advance(task)

        end_time = time.time()
        summary = self._build_summary(start_time, end_time)

        # Compute topic reconstruction metrics
        # Treat as elevated only when a healthcare probe exceeds the attacker's
        # own-domain baseline by a meaningful margin.
        elevation_threshold = avg_financial_score * 1.2
        elevated_terms = [
            word for word, data in healthcare_scores.items()
            if data["max_score"] > elevation_threshold
        ]

        summary.metadata = {
            "baseline_financial_avg_score": avg_financial_score,
            "healthcare_terms_tested": len(HEALTHCARE_VOCABULARY),
            "elevated_healthcare_terms": len(elevated_terms),
            "elevated_terms_list": elevated_terms,
            "elevation_threshold": elevation_threshold,
            "topic_reconstruction_rate": len(elevated_terms) / len(HEALTHCARE_VOCABULARY),
            "refined_probes_tested": len(REFINED_PROBES),
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
        console.print(f"  Baseline financial avg score: {summary.metadata.get('baseline_financial_avg_score', 0):.4f}")
        console.print(
            f"  Elevated healthcare terms: "
            f"{summary.metadata.get('elevated_healthcare_terms', 0)}/{summary.metadata.get('healthcare_terms_tested', 0)}"
        )
        console.print(f"  Topic reconstruction rate: {summary.metadata.get('topic_reconstruction_rate', 0):.2%}")
        console.print(f"  Duration: {summary.duration_seconds:.1f}s")

        if summary.metadata.get("elevated_terms_list"):
            console.print(f"\n  [yellow]Elevated terms (potential leakage signal):[/yellow]")
            for term in summary.metadata["elevated_terms_list"][:10]:
                console.print(f"    • {term}")
