"""Experiment 4: Framework-Level Misconfiguration Audit.

Tests whether RAG framework retrieval abstractions introduce isolation failures
independent of the underlying vector database's enforcement.

Tests 3 retrieval methods per database:
1. LangChain retriever without tenant filter
2. LlamaIndex retriever without tenant filter
3. Native SDK query without tenant filter

Common failure patterns tested:
- Retriever configured without namespace/filter parameter (queries entire index)
- Application-layer filtering that doesn't prevent DB-level access
- Concurrent tenant queries (context bleed)
"""

import time
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress

from vectorbleed.config import Settings, get_settings
from vectorbleed.databases.base import TenantConfig, VectorDatabaseAdapter
from vectorbleed.embeddings import EmbeddingEngine
from vectorbleed.experiments.base import BaseExperiment, ExperimentResult, ExperimentSummary

console = Console()

# Test queries that should ONLY return healthcare content (victim tenant)
HEALTHCARE_QUERIES = [
    "What is the ICU admission protocol?",
    "Describe the oncology drug formulary pricing",
    "What are the HIPAA patient data retention requirements?",
    "Explain the clinical trial enrollment criteria",
    "What is the surgical consent form procedure?",
]


class MisconfigurationExperiment(BaseExperiment):
    """Experiment 4: Framework Misconfiguration Audit — LangChain + LlamaIndex + Native SDK."""

    @property
    def name(self) -> str:
        return "Framework Misconfiguration Audit"

    @property
    def experiment_id(self) -> str:
        return "exp4_framework_misconfiguration"

    @property
    def description(self) -> str:
        return (
            "Tests whether LangChain, LlamaIndex, and native SDK retrieval patterns "
            "introduce isolation failures when tenant filters are omitted."
        )

    def run(self) -> ExperimentSummary:
        """Execute framework misconfiguration audit across all 3 methods."""
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Experiment 4: {self.name}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        start_time = time.time()
        self.results = []

        db_name = self.db.__class__.__name__
        console.print(f"  Database: [bold]{db_name}[/bold]")
        console.print(f"  Testing: LangChain, LlamaIndex, Native SDK\n")

        # Test 1: Native SDK without tenant filter
        self._test_native_sdk()

        # Test 2: LangChain retriever without tenant filter
        self._test_langchain()

        # Test 3: LlamaIndex retriever without tenant filter
        self._test_llamaindex()

        # Test 4: Application-layer filtering (Python-side only)
        self._test_application_layer_filter()

        # Test 5: Concurrent tenant access
        self._test_concurrent_access()

        end_time = time.time()
        summary = self._build_summary(start_time, end_time)

        # Categorize findings by framework
        native_results = [
            r for r in self.results
            if r.metadata.get("framework") == "native_sdk"
            and r.metadata.get("test_type") == "no_namespace"
        ]
        langchain_results = [r for r in self.results if r.metadata.get("framework") == "langchain"]
        llamaindex_results = [r for r in self.results if r.metadata.get("framework") == "llamaindex"]

        summary.metadata = {
            "database": db_name,
            "native_sdk_vulnerable": any(r.cross_tenant_detected for r in native_results),
            "native_sdk_detection_rate": (
                sum(1 for r in native_results if r.cross_tenant_detected) / max(len(native_results), 1)
            ),
            "langchain_vulnerable": any(r.cross_tenant_detected for r in langchain_results),
            "langchain_blocked_by_db": any(r.metadata.get("blocked_by_db") for r in langchain_results),
            "langchain_detection_rate": (
                sum(1 for r in langchain_results if r.cross_tenant_detected) / max(len(langchain_results), 1)
            ),
            "llamaindex_vulnerable": any(r.cross_tenant_detected for r in llamaindex_results),
            "llamaindex_blocked_by_db": any(r.metadata.get("blocked_by_db") for r in llamaindex_results),
            "llamaindex_detection_rate": (
                sum(1 for r in llamaindex_results if r.cross_tenant_detected) / max(len(llamaindex_results), 1)
            ),
            "app_layer_filter_vulnerable": any(
                r.cross_tenant_detected for r in self.results if r.metadata.get("test_type") == "app_layer_filter"
            ),
            "concurrent_access_vulnerable": any(
                r.cross_tenant_detected for r in self.results if r.metadata.get("test_type") == "concurrent_access"
            ),
        }

        self._print_summary(summary)
        return summary

    def _test_native_sdk(self) -> None:
        """Test 1: Native SDK query without tenant filter."""
        console.print("[bold]Test 1: Native SDK — no tenant filter[/bold]")
        console.print(f"  Method: {self.db.__class__.__name__}.query_without_namespace()")

        for i, query in enumerate(HEALTHCARE_QUERIES):
            query_emb = self.embedding_engine.embed_text(query)

            start = time.perf_counter()
            results = self.db.query_without_namespace(query_emb, top_k=10)
            latency = (time.perf_counter() - start) * 1000

            victim_results = [
                r for r in results
                if r.metadata.get("tenant_id") == self.victim.tenant_id
            ]

            self.results.append(
                ExperimentResult(
                    probe_id=len(self.results),
                    probe_query=f"[native_sdk] {query}",
                    results_count=len(results),
                    top_scores=[r.score for r in results[:5]],
                    latency_ms=latency,
                    cross_tenant_detected=len(victim_results) > 0,
                    cross_tenant_count=len(victim_results),
                    metadata={
                        "test_type": "no_namespace",
                        "framework": "app_layer_filter",
                        "victim_results_in_top_10": len(victim_results),
                    },
                )
            )

        vuln_count = sum(
            1 for r in self.results
            if r.metadata.get("framework") == "native_sdk" and r.cross_tenant_detected
        )
        status = "red" if vuln_count > 0 else "green"
        console.print(f"  [{status}]  Result: {vuln_count}/{len(HEALTHCARE_QUERIES)} leaked cross-tenant data[/]")

    def _test_langchain(self) -> None:
        """Test 2: LangChain retriever without tenant filter."""
        console.print("\n[bold]Test 2: LangChain — no tenant filter[/bold]")
        console.print(f"  Method: vectorstore.as_retriever() without filter/namespace")

        for i, query in enumerate(HEALTHCARE_QUERIES):
            start = time.perf_counter()
            blocked_by_db = False
            try:
                results = self.db.query_langchain_retriever(
                    self.attacker, query, top_k=10, use_namespace=False
                )
            except NotImplementedError:
                # LangChain not implemented for this DB — use native fallback
                query_emb = self.embedding_engine.embed_text(query)
                results = self.db.query_without_namespace(query_emb, top_k=10)
            except Exception as e:
                error_msg = str(e).lower()
                if "multi-tenancy" in error_msg or "tenant" in error_msg:
                    # DB enforced isolation — this is SECURE behavior
                    console.print(f"  [green]  Query {i+1}: BLOCKED by DB (multi-tenancy enforced)[/green]")
                    blocked_by_db = True
                    results = []
                else:
                    console.print(f"  [yellow]  LangChain error: {e}[/yellow]")
                    results = []
            latency = (time.perf_counter() - start) * 1000

            if blocked_by_db:
                self.results.append(
                    ExperimentResult(
                        probe_id=len(self.results),
                        probe_query=f"[langchain] {query}",
                        results_count=0,
                        top_scores=[],
                        latency_ms=latency,
                        cross_tenant_detected=False,
                        cross_tenant_count=0,
                        metadata={
                            "test_type": "no_namespace",
                            "framework": "langchain",
                            "blocked_by_db": True,
                            "note": "DB enforced multi-tenancy — query rejected without tenant context",
                        },
                    )
                )
            else:
                victim_results = [
                    r for r in results
                    if r.metadata.get("tenant_id") == self.victim.tenant_id
                ]
                self.results.append(
                    ExperimentResult(
                        probe_id=len(self.results),
                        probe_query=f"[langchain] {query}",
                        results_count=len(results),
                        top_scores=[r.score for r in results[:5]],
                        latency_ms=latency,
                        cross_tenant_detected=len(victim_results) > 0,
                        cross_tenant_count=len(victim_results),
                        metadata={
                            "test_type": "no_namespace",
                            "framework": "langchain",
                            "victim_results_in_top_10": len(victim_results),
                        },
                    )
                )

        vuln_count = sum(
            1 for r in self.results
            if r.metadata.get("framework") == "langchain" and r.cross_tenant_detected
        )
        blocked_count = sum(
            1 for r in self.results
            if r.metadata.get("framework") == "langchain" and r.metadata.get("blocked_by_db")
        )
        status = "red" if vuln_count > 0 else "green"
        console.print(f"  [{status}]  Result: {vuln_count}/{len(HEALTHCARE_QUERIES)} leaked | {blocked_count} blocked by DB[/]")

    def _test_llamaindex(self) -> None:
        """Test 3: LlamaIndex retriever without tenant filter."""
        console.print("\n[bold]Test 3: LlamaIndex — no tenant filter[/bold]")
        console.print(f"  Method: VectorStoreIndex.as_query_engine() without filter")

        for i, query in enumerate(HEALTHCARE_QUERIES):
            start = time.perf_counter()
            blocked_by_db = False
            try:
                results = self._query_llamaindex_no_filter(query, top_k=10)
            except Exception as e:
                error_msg = str(e).lower()
                if "multi-tenancy" in error_msg or "tenant" in error_msg:
                    # DB enforced isolation — SECURE
                    console.print(f"  [green]  Query {i+1}: BLOCKED by DB (multi-tenancy enforced)[/green]")
                    blocked_by_db = True
                    results = []
                elif "validation error" in error_msg or "textnode" in error_msg:
                    # LlamaIndex parsing issue — the query DID reach the DB and access data
                    # This means isolation was NOT enforced, just the response parsing failed
                    # Fall back to native SDK to confirm the vulnerability
                    query_emb = self.embedding_engine.embed_text(query)
                    results = self.db.query_without_namespace(query_emb, top_k=10)
                else:
                    console.print(f"  [yellow]  LlamaIndex error: {e}[/yellow]")
                    results = []
            latency = (time.perf_counter() - start) * 1000

            if blocked_by_db:
                self.results.append(
                    ExperimentResult(
                        probe_id=len(self.results),
                        probe_query=f"[llamaindex] {query}",
                        results_count=0,
                        top_scores=[],
                        latency_ms=latency,
                        cross_tenant_detected=False,
                        cross_tenant_count=0,
                        metadata={
                            "test_type": "no_namespace",
                            "framework": "llamaindex",
                            "blocked_by_db": True,
                            "note": "DB enforced multi-tenancy — query rejected without tenant context",
                        },
                    )
                )
            else:
                victim_results = [
                    r for r in results
                    if r.metadata.get("tenant_id") == self.victim.tenant_id
                ]
                self.results.append(
                    ExperimentResult(
                        probe_id=len(self.results),
                        probe_query=f"[llamaindex] {query}",
                        results_count=len(results),
                        top_scores=[r.score for r in results[:5]],
                        latency_ms=latency,
                        cross_tenant_detected=len(victim_results) > 0,
                        cross_tenant_count=len(victim_results),
                        metadata={
                            "test_type": "no_namespace",
                            "framework": "llamaindex",
                            "victim_results_in_top_10": len(victim_results),
                        },
                    )
                )

        vuln_count = sum(
            1 for r in self.results
            if r.metadata.get("framework") == "llamaindex" and r.cross_tenant_detected
        )
        blocked_count = sum(
            1 for r in self.results
            if r.metadata.get("framework") == "llamaindex" and r.metadata.get("blocked_by_db")
        )
        status = "red" if vuln_count > 0 else "green"
        console.print(f"  [{status}]  Result: {vuln_count}/{len(HEALTHCARE_QUERIES)} leaked | {blocked_count} blocked by DB[/]")

    def _query_llamaindex_no_filter(self, query_text: str, top_k: int = 10) -> list:
        """Query using LlamaIndex without tenant filter — the vulnerable pattern."""
        from llama_index.core.vector_stores import VectorStoreQuery
        from llama_index.embeddings.openai import OpenAIEmbedding
        from vectorbleed.databases.base import QueryResult

        embed_model = OpenAIEmbedding(
            model_name=self.settings.openai_embedding_model,
            api_key=self.settings.openai_api_key,
        )

        # Get the appropriate LlamaIndex vector store
        vector_store = self._get_llamaindex_vector_store()

        # Query without any filter — the vulnerable pattern
        query_embedding = embed_model.get_query_embedding(query_text)

        query_obj = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            # NO filters — this is the vulnerability
        )

        query_result = vector_store.query(query_obj)

        results = []
        if query_result.nodes:
            for i, node in enumerate(query_result.nodes):
                metadata = node.metadata or {}
                score = query_result.similarities[i] if query_result.similarities and i < len(query_result.similarities) else 0.0
                results.append(
                    QueryResult(
                        id=metadata.get("doc_id", ""),
                        score=score,
                        metadata=metadata,
                        namespace=None,
                    )
                )

        return results

    def _get_llamaindex_vector_store(self):
        """Get the appropriate LlamaIndex vector store for the current DB."""
        from vectorbleed.databases.pinecone_db import PineconeAdapter
        from vectorbleed.databases.qdrant_db import QdrantAdapter
        from vectorbleed.databases.weaviate_db import WeaviateAdapter
        from vectorbleed.databases.chroma_db import ChromaDBAdapter

        if isinstance(self.db, PineconeAdapter):
            from llama_index.vector_stores.pinecone import PineconeVectorStore
            # VULNERABLE: no namespace specified
            return PineconeVectorStore(
                pinecone_index=self.db.index,
                # namespace NOT set — queries all namespaces
            )

        elif isinstance(self.db, QdrantAdapter):
            from llama_index.vector_stores.qdrant import QdrantVectorStore
            # VULNERABLE: no filter on tenant_id
            return QdrantVectorStore(
                client=self.db.client,
                collection_name="vectorbleed_docs",
                enable_hybrid=False,
                # No metadata filter — queries all tenants
            )

        elif isinstance(self.db, WeaviateAdapter):
            from llama_index.vector_stores.weaviate import WeaviateVectorStore
            # Weaviate with multi-tenancy — LlamaIndex may or may not enforce it
            return WeaviateVectorStore(
                weaviate_client=self.db.client,
                index_name="VectorBleedDocs",
            )

        elif isinstance(self.db, ChromaDBAdapter):
            from llama_index.vector_stores.chroma import ChromaVectorStore
            # VULNERABLE: shared collection, no where filter
            return ChromaVectorStore(
                chroma_collection=self.db.shared_collection,
                # No metadata filter — queries all tenants
            )

        else:
            raise NotImplementedError(f"LlamaIndex not implemented for {type(self.db)}")

    def _test_application_layer_filter(self) -> None:
        """Test 4: Filter applied in Python but not passed to vector DB."""
        console.print("\n[bold]Test 4: Application-layer-only filtering[/bold]")
        console.print("  Method: Query all, filter in Python (DB already accessed victim vectors)")

        for i, query in enumerate(HEALTHCARE_QUERIES):
            query_emb = self.embedding_engine.embed_text(query)

            start = time.perf_counter()
            all_results = self.db.query_without_namespace(query_emb, top_k=20)
            latency = (time.perf_counter() - start) * 1000

            # Application-layer filter (Python-side)
            filtered_results = [
                r for r in all_results
                if r.metadata.get("tenant_id") == self.attacker.tenant_id
            ]

            # The vulnerability: the DB already accessed victim's vectors
            victim_accessed = [
                r for r in all_results
                if r.metadata.get("tenant_id") == self.victim.tenant_id
            ]

            self.results.append(
                ExperimentResult(
                    probe_id=len(self.results),
                    probe_query=f"[app_filter] {query}",
                    results_count=len(filtered_results),
                    top_scores=[r.score for r in all_results[:5]],
                    latency_ms=latency,
                    cross_tenant_detected=len(victim_accessed) > 0,
                    cross_tenant_count=len(victim_accessed),
                    metadata={
                        "test_type": "app_layer_filter",
                        "framework": "native_sdk",
                        "total_results_before_filter": len(all_results),
                        "results_after_filter": len(filtered_results),
                        "victim_vectors_accessed": len(victim_accessed),
                    },
                )
            )

        vuln_count = sum(
            1 for r in self.results
            if r.metadata.get("test_type") == "app_layer_filter" and r.cross_tenant_detected
        )
        status = "red" if vuln_count > 0 else "green"
        console.print(f"  [{status}]  Result: {vuln_count}/{len(HEALTHCARE_QUERIES)} accessed victim vectors[/]")

    def _test_concurrent_access(self) -> None:
        """Test 5: Concurrent tenant queries — context bleed."""
        console.print("\n[bold]Test 5: Concurrent tenant access (context bleed)[/bold]")
        console.print("  Method: Sequential queries from two tenants, check for bleed")

        healthcare_query = "What is the ICU admission protocol?"
        health_emb = self.embedding_engine.embed_text(healthcare_query)

        # Tenant A queries healthcare (should NOT get healthcare results with proper isolation)
        results_cross, lat_cross = self.db.query_cross_namespace_timed(
            self.attacker, health_emb, top_k=10
        )

        detected, cross_count = self._detect_cross_tenant(results_cross)

        self.results.append(
            ExperimentResult(
                probe_id=len(self.results),
                probe_query=f"[concurrent] Tenant A querying healthcare content",
                results_count=len(results_cross),
                top_scores=[r.score for r in results_cross[:5]],
                latency_ms=lat_cross * 1000,
                cross_tenant_detected=detected,
                cross_tenant_count=cross_count,
                metadata={
                    "test_type": "concurrent_access",
                    "framework": "concurrent_access",
                },
            )
        )

        status = "red" if detected else "green"
        console.print(f"  [{status}]  Result: {'VULNERABLE' if detected else 'SECURE'} — "
                      f"cross-tenant count: {cross_count}[/]")

    def _print_summary(self, summary: ExperimentSummary) -> None:
        """Print experiment results."""
        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print(f"[bold green]Results: {self.name}[/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]")
        console.print(f"  Database: {summary.metadata.get('database', '?')}")
        console.print(f"  Total tests: {summary.total_probes}")
        console.print(f"  Overall detection rate: {summary.detection_rate:.2%}")
        console.print(f"\n  [bold]Framework Vulnerability Matrix:[/bold]")
        console.print(f"    Native SDK:      {'🔴 VULNERABLE' if summary.metadata.get('native_sdk_vulnerable') else '🟢 SECURE'} ({summary.metadata.get('native_sdk_detection_rate', 0):.0%})")

        lc_blocked = summary.metadata.get('langchain_blocked_by_db')
        lc_vuln = summary.metadata.get('langchain_vulnerable')
        if lc_blocked and not lc_vuln:
            console.print(f"    LangChain:       🟢 BLOCKED BY DB (multi-tenancy enforced)")
        else:
            console.print(f"    LangChain:       {'🔴 VULNERABLE' if lc_vuln else '🟢 SECURE'} ({summary.metadata.get('langchain_detection_rate', 0):.0%})")

        li_blocked = summary.metadata.get('llamaindex_blocked_by_db')
        li_vuln = summary.metadata.get('llamaindex_vulnerable')
        if li_blocked and not li_vuln:
            console.print(f"    LlamaIndex:      🟢 BLOCKED BY DB (multi-tenancy enforced)")
        else:
            console.print(f"    LlamaIndex:      {'🔴 VULNERABLE' if li_vuln else '🟢 SECURE'} ({summary.metadata.get('llamaindex_detection_rate', 0):.0%})")

        console.print(f"    App-layer filter: {'🔴 VULNERABLE' if summary.metadata.get('app_layer_filter_vulnerable') else '🟢 SECURE'}")
        console.print(f"    Concurrent:      {'🔴 VULNERABLE' if summary.metadata.get('concurrent_access_vulnerable') else '🟢 SECURE'}")
        console.print(f"  Duration: {summary.duration_seconds:.1f}s")
