"""Pinecone vector database adapter using LangChain for multi-tenant isolation testing.

Uses LangChain's PineconeVectorStore for retrieval operations, which is the exact
pattern enterprise RAG applications use — making our misconfiguration findings
directly applicable to real-world deployments.
"""

import time
from typing import Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from rich.console import Console

from vectorbleed.config import Settings, get_settings
from vectorbleed.databases.base import QueryResult, TenantConfig, VectorDatabaseAdapter

console = Console()


class PineconeAdapter(VectorDatabaseAdapter):
    """Pinecone implementation using LangChain with namespace-based multi-tenancy.

    This adapter uses LangChain's PineconeVectorStore — the same abstraction
    that enterprise RAG applications use. This makes Experiment 4 (framework
    misconfiguration) directly test real-world vulnerable patterns.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.pc: Optional[Pinecone] = None
        self.index = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self._vectorstores: dict[str, PineconeVectorStore] = {}

    def initialize(self) -> None:
        """Initialize Pinecone client, LangChain embeddings, and ensure index exists."""
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)

        # LangChain OpenAI embeddings — shared across tenants (realistic scenario)
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )

        index_name = self.settings.pinecone_index_name

        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            console.print(f"[yellow]Creating Pinecone index: {index_name}[/yellow]")
            self.pc.create_index(
                name=index_name,
                dimension=self.settings.pinecone_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.settings.pinecone_environment,
                ),
            )
            # Wait for index to be ready
            while not self.pc.describe_index(index_name).status.get("ready"):
                time.sleep(1)
            console.print(f"[green]✓ Index '{index_name}' created and ready[/green]")
        else:
            console.print(f"[green]✓ Using existing index: {index_name}[/green]")

        self.index = self.pc.Index(index_name)

    def _get_vectorstore(self, namespace: str) -> PineconeVectorStore:
        """Get or create a LangChain PineconeVectorStore for a namespace."""
        if namespace not in self._vectorstores:
            self._vectorstores[namespace] = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace=namespace,
                text_key="content_preview",
            )
        return self._vectorstores[namespace]

    def _get_vectorstore_no_namespace(self) -> PineconeVectorStore:
        """Get a LangChain PineconeVectorStore WITHOUT namespace — the vulnerable pattern."""
        # This is the exact pattern that causes cross-tenant leakage:
        # PineconeVectorStore without namespace queries ALL namespaces
        return PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            namespace="",  # Empty string = query all namespaces
            text_key="content_preview",
        )

    def upsert_documents(
        self,
        tenant: TenantConfig,
        documents: list[dict],
        embeddings: np.ndarray,
    ) -> int:
        """Upsert documents into tenant's namespace using LangChain."""
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vectors.append({
                "id": f"{tenant.tenant_id}_{doc['id']}",
                "values": embedding.tolist(),
                "metadata": {
                    "tenant_id": tenant.tenant_id,
                    "doc_id": doc["id"],
                    "topic": doc.get("topic", ""),
                    "domain": doc.get("domain", ""),
                    "content_preview": doc.get("content", "")[:200],
                },
            })

        # Upsert in batches of 100 via raw Pinecone (LangChain add_texts is slower)
        batch_size = 100
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=tenant.namespace)
            total_upserted += len(batch)

        return total_upserted

    def query(
        self,
        tenant: TenantConfig,
        query_embedding: np.ndarray,
        top_k: int = 10,
        include_scores: bool = True,
        include_vectors: bool = False,
    ) -> list[QueryResult]:
        """Query within a tenant's namespace using LangChain retriever (SECURE pattern).

        This uses: vectorstore.as_retriever(search_kwargs={"namespace": ..., "k": ...})
        """
        vectorstore = self._get_vectorstore(tenant.namespace)

        # Use similarity_search_by_vector_with_score for detailed results
        results_with_scores = vectorstore.similarity_search_by_vector_with_score(
            embedding=query_embedding.tolist(),
            k=top_k,
        )

        results = []
        for doc, score in results_with_scores:
            results.append(
                QueryResult(
                    id=doc.metadata.get("doc_id", ""),
                    score=score,
                    metadata=doc.metadata,
                    namespace=tenant.namespace,
                    vector=None,
                )
            )

        return results

    def query_without_namespace(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[QueryResult]:
        """Query WITHOUT namespace restriction — the VULNERABLE LangChain pattern.

        This simulates the common misconfiguration:
            vectorstore = PineconeVectorStore(index, embeddings)
            retriever = vectorstore.as_retriever()  # NO namespace!

        In this pattern, LangChain queries across ALL namespaces.
        """
        vectorstore = self._get_vectorstore_no_namespace()

        results_with_scores = vectorstore.similarity_search_by_vector_with_score(
            embedding=query_embedding.tolist(),
            k=top_k,
        )

        results = []
        for doc, score in results_with_scores:
            results.append(
                QueryResult(
                    id=doc.metadata.get("doc_id", ""),
                    score=score,
                    metadata=doc.metadata,
                    namespace=None,
                )
            )

        return results

    def query_with_metadata_filter(
        self,
        tenant: TenantConfig,
        query_embedding: np.ndarray,
        metadata_filter: dict,
        top_k: int = 10,
    ) -> list[QueryResult]:
        """Query with LangChain metadata filter — tests post-retrieval filtering.

        This uses search_kwargs with filter, which Pinecone applies AFTER
        HNSW graph traversal. The vectors are already accessed before filtering.
        """
        vectorstore = self._get_vectorstore(tenant.namespace)

        # LangChain pattern: similarity_search with filter kwarg
        results_with_scores = vectorstore.similarity_search_by_vector_with_score(
            embedding=query_embedding.tolist(),
            k=top_k,
            filter=metadata_filter,
        )

        results = []
        for doc, score in results_with_scores:
            results.append(
                QueryResult(
                    id=doc.metadata.get("doc_id", ""),
                    score=score,
                    metadata=doc.metadata,
                    namespace=tenant.namespace,
                )
            )

        return results

    def query_langchain_retriever(
        self,
        tenant: TenantConfig,
        query_text: str,
        top_k: int = 10,
        use_namespace: bool = True,
    ) -> list[QueryResult]:
        """Query using LangChain's retriever interface — the standard RAG pattern.

        Args:
            use_namespace: If True, uses secure pattern. If False, simulates misconfiguration.
        """
        if use_namespace:
            vectorstore = self._get_vectorstore(tenant.namespace)
        else:
            vectorstore = self._get_vectorstore_no_namespace()

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        docs = retriever.invoke(query_text)

        results = []
        for doc in docs:
            results.append(
                QueryResult(
                    id=doc.metadata.get("doc_id", ""),
                    score=doc.metadata.get("score", 0.0),
                    metadata=doc.metadata,
                    namespace=tenant.namespace if use_namespace else None,
                )
            )

        return results

    def query_cross_namespace_timed(
        self,
        tenant: TenantConfig,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> tuple[list[QueryResult], float]:
        """Query with timing — measures latency for side-channel analysis."""
        start = time.perf_counter()
        results = self.query(tenant, query_embedding, top_k)
        latency = time.perf_counter() - start
        return results, latency

    def get_namespace_stats(self, tenant: TenantConfig) -> dict:
        """Get statistics about a tenant's namespace."""
        stats = self.index.describe_index_stats()
        ns_stats = stats.namespaces.get(tenant.namespace, {})
        return {
            "namespace": tenant.namespace,
            "vector_count": getattr(ns_stats, "vector_count", 0),
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
        }

    def delete_namespace(self, tenant: TenantConfig) -> None:
        """Delete all vectors in a tenant's namespace."""
        self.index.delete(delete_all=True, namespace=tenant.namespace)

    def teardown(self) -> None:
        """Clean up resources."""
        self._vectorstores.clear()
        self.index = None
        self.pc = None
