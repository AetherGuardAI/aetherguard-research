"""Experiment 2: Black-Hole Centroid Injection.

Tests whether centroid-position embeddings cross namespace boundaries.
Computes the mathematical centroid of the full embedding space (across both tenants),
crafts a document whose embedding lands near this centroid, uploads it to the
attacker's namespace, and observes whether victim documents appear as neighbors.
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

# Chameleon documents — semantically generic, designed to land near embedding centroid
CHAMELEON_DOCUMENTS = [
    (
        "Compliance and risk management procedures ensure organizational security "
        "and data protection across all regulated environments and operational contexts. "
        "Healthcare protocols and financial regulations share common frameworks for "
        "access control, audit trails, and data minimization requirements."
    ),
    (
        "Enterprise governance frameworks mandate comprehensive documentation of "
        "operational procedures, risk assessments, and compliance verification across "
        "all business units. Regulatory requirements span multiple domains including "
        "data privacy, financial reporting, and patient safety standards."
    ),
    (
        "Information security policies require systematic classification of sensitive "
        "data, implementation of access controls, and regular audit procedures. "
        "Both financial institutions and healthcare organizations must maintain "
        "detailed records of data handling and processing activities."
    ),
    (
        "Organizational risk assessment methodologies evaluate threats across "
        "operational, financial, and regulatory dimensions. Standard frameworks "
        "apply to diverse sectors including banking, insurance, pharmaceuticals, "
        "and hospital systems requiring unified compliance approaches."
    ),
    (
        "Data retention and lifecycle management policies govern the storage, "
        "archival, and destruction of sensitive records. Regulatory mandates "
        "from HIPAA, SOX, GDPR, and PCI-DSS impose overlapping requirements "
        "on organizations handling protected information."
    ),
]


class CentroidInjectionExperiment(BaseExperiment):
    """Experiment 2: Black-Hole Centroid Injection Attack."""

    @property
    def name(self) -> str:
        return "Black-Hole Centroid Injection"

    @property
    def experiment_id(self) -> str:
        return "exp2_centroid_injection"

    @property
    def description(self) -> str:
        return (
            "Tests whether centroid-position embeddings cross namespace boundaries. "
            "Injects documents positioned near the global embedding space centroid "
            "and observes whether victim tenant documents appear as neighbors."
        )

    def run(
        self,
        tenant_a_embeddings: Optional[np.ndarray] = None,
        tenant_b_embeddings: Optional[np.ndarray] = None,
    ) -> ExperimentSummary:
        """Execute centroid injection experiment."""
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Experiment 2: {self.name}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        start_time = time.time()
        self.results = []

        # Phase 1: Compute global centroid
        console.print("[bold]Phase 1: Computing global embedding space centroid[/bold]")

        if tenant_a_embeddings is not None and tenant_b_embeddings is not None:
            all_embeddings = np.vstack([tenant_a_embeddings, tenant_b_embeddings])
        else:
            # If embeddings not provided, we'll use chameleon docs directly
            all_embeddings = None

        global_centroid = None
        if all_embeddings is not None:
            global_centroid = np.mean(all_embeddings, axis=0)
            centroid_a = np.mean(tenant_a_embeddings, axis=0)
            centroid_b = np.mean(tenant_b_embeddings, axis=0)

            dist_a_to_global = self.embedding_engine.cosine_distance(centroid_a, global_centroid)
            dist_b_to_global = self.embedding_engine.cosine_distance(centroid_b, global_centroid)

            console.print(f"  Tenant A centroid → global centroid distance: {dist_a_to_global:.4f}")
            console.print(f"  Tenant B centroid → global centroid distance: {dist_b_to_global:.4f}")

        # Phase 2: Embed chameleon documents
        console.print("\n[bold]Phase 2: Embedding chameleon documents[/bold]")
        chameleon_embeddings = self.embedding_engine.embed_batch(CHAMELEON_DOCUMENTS)

        if global_centroid is not None:
            for i, emb in enumerate(chameleon_embeddings):
                dist = self.embedding_engine.cosine_distance(emb, global_centroid)
                console.print(f"  Chameleon {i} → centroid distance: {dist:.4f}")

        # Phase 3: Inject chameleon documents into attacker's namespace
        console.print("\n[bold]Phase 3: Injecting chameleon documents into attacker namespace[/bold]")
        chameleon_docs = [
            {"id": f"chameleon_{i:03d}", "content": doc, "topic": "chameleon_centroid", "domain": "cross-domain"}
            for i, doc in enumerate(CHAMELEON_DOCUMENTS)
        ]
        self.db.upsert_documents(self.attacker, chameleon_docs, chameleon_embeddings)
        console.print(f"  Injected {len(chameleon_docs)} chameleon documents")

        # Phase 4: Query using chameleon embeddings and observe results
        console.print("\n[bold]Phase 4: Querying with chameleon embeddings[/bold]")

        with Progress(console=console) as progress:
            task = progress.add_task("Testing centroid queries...", total=len(CHAMELEON_DOCUMENTS))

            for i, (doc, emb) in enumerate(zip(CHAMELEON_DOCUMENTS, chameleon_embeddings)):
                # Query from attacker's namespace using chameleon embedding
                results, latency = self.db.query_cross_namespace_timed(
                    self.attacker, emb, top_k=10
                )

                top_scores = [r.score for r in results[:5]]
                detected, cross_count = self._detect_cross_tenant(results)

                # Also test without namespace (misconfiguration scenario)
                results_no_ns = self.db.query_without_namespace(emb, top_k=10)
                _, cross_count_no_ns = self._detect_cross_tenant(results_no_ns)

                self.results.append(
                    ExperimentResult(
                        probe_id=i,
                        probe_query=f"[Chameleon Doc {i}] {doc[:80]}...",
                        results_count=len(results),
                        top_scores=top_scores,
                        latency_ms=latency * 1000,
                        cross_tenant_detected=detected,
                        cross_tenant_count=cross_count,
                        metadata={
                            "chameleon_doc_index": i,
                            "centroid_distance": float(
                                self.embedding_engine.cosine_distance(emb, global_centroid)
                            ) if global_centroid is not None else None,
                            "no_namespace_cross_tenant_count": cross_count_no_ns,
                            "no_namespace_total_results": len(results_no_ns),
                        },
                    )
                )

                progress.advance(task)

        # Phase 5: Test with synthetic centroid-targeted queries
        console.print("\n[bold]Phase 5: Synthetic centroid-targeted queries[/bold]")
        centroid_queries = [
            "compliance risk management data protection regulated environments",
            "governance documentation operational procedures audit verification",
            "security classification access controls sensitive data handling",
            "regulatory assessment financial healthcare unified compliance",
            "retention lifecycle management archival destruction protected records",
        ]

        for i, query in enumerate(centroid_queries):
            query_emb = self.embedding_engine.embed_text(query)
            results, latency = self.db.query_cross_namespace_timed(
                self.attacker, query_emb, top_k=10
            )

            top_scores = [r.score for r in results[:5]]
            detected, cross_count = self._detect_cross_tenant(results)

            self.results.append(
                ExperimentResult(
                    probe_id=len(CHAMELEON_DOCUMENTS) + i,
                    probe_query=query,
                    results_count=len(results),
                    top_scores=top_scores,
                    latency_ms=latency * 1000,
                    cross_tenant_detected=detected,
                    cross_tenant_count=cross_count,
                    metadata={"phase": "centroid_targeted_query"},
                )
            )

        end_time = time.time()
        summary = self._build_summary(start_time, end_time)

        # Add experiment-specific metadata
        summary.metadata = {
            "chameleon_docs_injected": len(CHAMELEON_DOCUMENTS),
            "no_namespace_cross_tenant_total": sum(
                r.metadata.get("no_namespace_cross_tenant_count", 0)
                for r in self.results
                if "no_namespace_cross_tenant_count" in r.metadata
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
        console.print(f"  Cross-tenant detections (with namespace): {summary.cross_tenant_detections}")
        console.print(f"  Detection rate: {summary.detection_rate:.2%}")
        console.print(
            f"  Cross-tenant hits (no namespace): "
            f"{summary.metadata.get('no_namespace_cross_tenant_total', 0)}"
        )
        console.print(f"  Max similarity score: {summary.max_similarity_score:.4f}")
        console.print(f"  Duration: {summary.duration_seconds:.1f}s")
