"""ChromaDB vector database adapter for multi-tenant isolation testing.

ChromaDB uses collections for logical isolation. Multi-tenancy is typically
implemented via separate collections or metadata filtering within a single collection.
This tests whether in-process filtering holds under adversarial conditions.
"""

import time
from typing import Optional

import chromadb
import numpy as np
from langchain_openai import OpenAIEmbeddings
from rich.console import Console

from vectorbleed.config import Settings, get_settings
from vectorbleed.databases.base import QueryResult, TenantConfig, VectorDatabaseAdapter

console = Console()


class ChromaDBAdapter(VectorDatabaseAdapter):
    """ChromaDB implementation with collection-based multi-tenancy.

    ChromaDB's isolation model uses either:
    - Separate collections per tenant (stronger isolation)
    - Single collection with metadata filtering (weaker, common pattern)

    We test the single-collection pattern (more common in tutorials/production)
    to evaluate whether metadata filtering provides real isolation.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.client: Optional[chromadb.ClientAPI] = None
        self.shared_collection = None  # Single collection (weak isolation)
        self.tenant_collections: dict = {}  # Per-tenant collections (strong isolation)
        self.embeddings: Optional[OpenAIEmbeddings] = None

    def initialize(self) -> None:
        """Initialize ChromaDB client (cloud or embedded)."""
        chroma_host = self.settings.chroma_host

        if chroma_host == "localhost_embedded":
            # Use persistent local storage (no server needed)
            self.client = chromadb.PersistentClient(path="results/chromadb_data")
            console.print(f"[green]✓ ChromaDB initialized (embedded/persistent)[/green]")
        elif "trychroma.com" in chroma_host or "chromadb.dev" in chroma_host:
            # Chroma Cloud
            self.client = chromadb.CloudClient(
                tenant=self.settings.chroma_tenant if self.settings.chroma_tenant != "default_tenant" else None,
                database=self.settings.chroma_database if self.settings.chroma_database != "default_database" else None,
                api_key=self.settings.chroma_api_key,
            )
            console.print(f"[green]✓ ChromaDB Cloud connected (db: {self.settings.chroma_database})[/green]")
        elif self.settings.chroma_api_key:
            # Self-hosted with auth
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=self.settings.chroma_port,
                headers={"Authorization": f"Bearer {self.settings.chroma_api_key}"},
            )
            console.print(f"[green]✓ ChromaDB connected ({chroma_host})[/green]")
        else:
            # Local Docker ChromaDB
            self.client = chromadb.HttpClient(host=chroma_host, port=self.settings.chroma_port)
            console.print(f"[green]✓ ChromaDB connected ({chroma_host}:{self.settings.chroma_port})[/green]")

        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )

        # Create shared collection (the common multi-tenant pattern)
        self.shared_collection = self.client.get_or_create_collection(
            name="vectorbleed_shared",
            metadata={"hnsw:space": "cosine"},
        )

        console.print(f"[green]✓ ChromaDB initialized (shared collection: vectorbleed_shared)[/green]")

    def upsert_documents(
        self,
        tenant: TenantConfig,
        documents: list[dict],
        embeddings: np.ndarray,
    ) -> int:
        """Upsert documents into shared collection with tenant metadata."""
        ids = []
        metadatas = []
        docs_text = []

        for doc, embedding in zip(documents, embeddings):
            doc_id = f"{tenant.tenant_id}_{doc['id']}"
            ids.append(doc_id)
            metadatas.append({
                "tenant_id": tenant.tenant_id,
                "doc_id": doc["id"],
                "topic": doc.get("topic", ""),
                "domain": doc.get("domain", ""),
            })
            docs_text.append(doc.get("content", "")[:200])

        # Upsert in batches (ChromaDB has a batch limit)
        batch_size = 100
        total = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size].tolist()
            batch_meta = metadatas[i : i + batch_size]
            batch_docs = docs_text[i : i + batch_size]

            self.shared_collection.upsert(
                ids=batch_ids,
                embeddings=batch_embs,
                metadatas=batch_meta,
                documents=batch_docs,
            )
            total += len(batch_ids)

        # Also create per-tenant collection for comparison
        tenant_col = self.client.get_or_create_collection(
            name=f"vectorbleed_{tenant.namespace}",
            metadata={"hnsw:space": "cosine"},
        )
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size].tolist()
            batch_meta = metadatas[i : i + batch_size]
            batch_docs = docs_text[i : i + batch_size]

            tenant_col.upsert(
                ids=batch_ids,
                embeddings=batch_embs,
                metadatas=batch_meta,
                documents=batch_docs,
            )

        self.tenant_collections[tenant.namespace] = tenant_col

        return total

    def query(
        self,
        tenant: TenantConfig,
        query_embedding: np.ndarray,
        top_k: int = 10,
        include_scores: bool = True,
        include_vectors: bool = False,
    ) -> list[QueryResult]:
        """Query with tenant filter on shared collection (SECURE pattern)."""
        response = self.shared_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where={"tenant_id": tenant.tenant_id},
            include=["metadatas", "distances", "documents"],
        )

        results = []
        if response["ids"] and response["ids"][0]:
            for i, doc_id in enumerate(response["ids"][0]):
                # ChromaDB returns distances; convert to similarity for cosine
                distance = response["distances"][0][i] if response["distances"] else 0
                score = 1.0 - distance  # cosine distance → similarity

                metadata = response["metadatas"][0][i] if response["metadatas"] else {}
                metadata["content_preview"] = (
                    response["documents"][0][i] if response["documents"] else ""
                )

                results.append(
                    QueryResult(
                        id=metadata.get("doc_id", doc_id),
                        score=score,
                        metadata=metadata,
                        namespace=tenant.namespace,
                    )
                )

        return results

    def query_without_namespace(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[QueryResult]:
        """Query WITHOUT tenant filter — the VULNERABLE pattern.

        In ChromaDB, if the developer forgets the `where` clause,
        the query searches across ALL tenants in the shared collection.
        """
        response = self.shared_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            # NO where clause — queries everything
            include=["metadatas", "distances", "documents"],
        )

        results = []
        if response["ids"] and response["ids"][0]:
            for i, doc_id in enumerate(response["ids"][0]):
                distance = response["distances"][0][i] if response["distances"] else 0
                score = 1.0 - distance

                metadata = response["metadatas"][0][i] if response["metadatas"] else {}
                metadata["content_preview"] = (
                    response["documents"][0][i] if response["documents"] else ""
                )

                results.append(
                    QueryResult(
                        id=metadata.get("doc_id", doc_id),
                        score=score,
                        metadata=metadata,
                        namespace=None,
                    )
                )

        return results

    def query_cross_namespace_timed(
        self,
        tenant: TenantConfig,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> tuple[list[QueryResult], float]:
        """Query with timing."""
        start = time.perf_counter()
        results = self.query(tenant, query_embedding, top_k)
        latency = time.perf_counter() - start
        return results, latency

    def query_langchain_retriever(
        self,
        tenant: TenantConfig,
        query_text: str,
        top_k: int = 10,
        use_namespace: bool = True,
    ) -> list[QueryResult]:
        """Query using LangChain's Chroma retriever.

        Args:
            use_namespace: If True, uses tenant filter. If False, no filter (vulnerable).
        """
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )

        vectorstore = Chroma(
            client=self.client,
            collection_name="vectorbleed_shared",
            embedding_function=embeddings,
        )

        if use_namespace:
            # SECURE: with tenant_id filter
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": top_k, "filter": {"tenant_id": tenant.tenant_id}}
            )
        else:
            # VULNERABLE: no filter — queries all tenants
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        docs = retriever.invoke(query_text)

        results = []
        for doc in docs:
            metadata = doc.metadata or {}
            results.append(
                QueryResult(
                    id=metadata.get("doc_id", ""),
                    score=metadata.get("score", 0.0),
                    metadata=metadata,
                    namespace=tenant.namespace if use_namespace else None,
                )
            )

        return results

    def get_namespace_stats(self, tenant: TenantConfig) -> dict:
        """Get statistics about a tenant's vectors."""
        total_count = self.shared_collection.count()

        # Count tenant-specific vectors
        tenant_col = self.tenant_collections.get(tenant.namespace)
        tenant_count = tenant_col.count() if tenant_col else 0

        return {
            "namespace": tenant.namespace,
            "vector_count": tenant_count,
            "total_vector_count": total_count,
            "isolation_model": "collection-metadata-filter",
        }

    def delete_namespace(self, tenant: TenantConfig) -> None:
        """Delete all vectors for a tenant."""
        # Delete from shared collection
        # ChromaDB requires IDs for deletion, so we query first
        results = self.shared_collection.get(
            where={"tenant_id": tenant.tenant_id},
        )
        if results["ids"]:
            self.shared_collection.delete(ids=results["ids"])

        # Delete tenant-specific collection
        try:
            self.client.delete_collection(f"vectorbleed_{tenant.namespace}")
        except Exception:
            pass

    def teardown(self) -> None:
        """Clean up resources."""
        self.shared_collection = None
        self.tenant_collections.clear()
        self.client = None
