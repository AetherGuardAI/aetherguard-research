"""Weaviate Cloud vector database adapter for multi-tenant isolation testing.

Weaviate uses class-based multi-tenancy with shards per tenant.
This tests whether shard-level isolation holds under adversarial conditions.
"""

import time
from typing import Optional

import numpy as np
from rich.console import Console
import weaviate
from weaviate.auth import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.tenants import Tenant

from vectorbleed.config import Settings, get_settings
from vectorbleed.databases.base import QueryResult, TenantConfig, VectorDatabaseAdapter

console = Console()

COLLECTION_NAME = "VectorBleedDocs"


class WeaviateAdapter(VectorDatabaseAdapter):
    """Weaviate Cloud implementation with shard-based multi-tenancy."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.client: Optional[weaviate.WeaviateClient] = None

    def initialize(self) -> None:
        """Initialize Weaviate Cloud client and ensure collection exists."""
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.settings.weaviate_url,
            auth_credentials=Auth.api_key(self.settings.weaviate_api_key),
            skip_init_checks=True,
        )

        console.print(f"[green]✓ Connected to Weaviate Cloud[/green]")

        # Create collection with multi-tenancy enabled
        if not self.client.collections.exists(COLLECTION_NAME):
            console.print(f"[yellow]Creating Weaviate collection: {COLLECTION_NAME}[/yellow]")
            self.client.collections.create(
                name=COLLECTION_NAME,
                multi_tenancy_config=Configure.multi_tenancy(
                    enabled=True,
                    auto_tenant_creation=True,
                ),
                properties=[
                    Property(name="content_preview", data_type=DataType.TEXT),
                    Property(name="tenant_id", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="topic", data_type=DataType.TEXT),
                    Property(name="domain", data_type=DataType.TEXT),
                ],
            )
            console.print(f"[green]✓ Collection '{COLLECTION_NAME}' created with multi-tenancy[/green]")
        else:
            console.print(f"[green]✓ Using existing collection: {COLLECTION_NAME}[/green]")

    def upsert_documents(
        self,
        tenant: TenantConfig,
        documents: list[dict],
        embeddings: np.ndarray,
    ) -> int:
        """Upsert documents into tenant's shard."""
        collection = self.client.collections.get(COLLECTION_NAME)

        # Ensure tenant exists
        try:
            collection.tenants.create([Tenant(name=tenant.namespace)])
        except Exception:
            pass  # Tenant may already exist

        tenant_collection = collection.with_tenant(tenant.namespace)

        with tenant_collection.batch.dynamic() as batch:
            for doc, embedding in zip(documents, embeddings):
                batch.add_object(
                    properties={
                        "content_preview": doc.get("content", "")[:200],
                        "tenant_id": tenant.tenant_id,
                        "doc_id": doc["id"],
                        "topic": doc.get("topic", ""),
                        "domain": doc.get("domain", ""),
                    },
                    vector=embedding.tolist(),
                )

        return len(documents)

    def query(
        self,
        tenant: TenantConfig,
        query_embedding: np.ndarray,
        top_k: int = 10,
        include_scores: bool = True,
        include_vectors: bool = False,
    ) -> list[QueryResult]:
        """Query within a tenant's shard (SECURE pattern)."""
        collection = self.client.collections.get(COLLECTION_NAME)
        tenant_collection = collection.with_tenant(tenant.namespace)

        response = tenant_collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=top_k,
            return_metadata=["distance"],
        )

        results = []
        for obj in response.objects:
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            score = 1.0 - distance  # cosine distance → similarity

            results.append(
                QueryResult(
                    id=obj.properties.get("doc_id", ""),
                    score=score,
                    metadata={
                        "tenant_id": obj.properties.get("tenant_id", ""),
                        "doc_id": obj.properties.get("doc_id", ""),
                        "topic": obj.properties.get("topic", ""),
                        "domain": obj.properties.get("domain", ""),
                        "content_preview": obj.properties.get("content_preview", ""),
                    },
                    namespace=tenant.namespace,
                )
            )

        return results

    def query_without_namespace(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[QueryResult]:
        """Query without tenant restriction — tests misconfiguration.

        In Weaviate with multi-tenancy, we iterate all tenant shards
        to simulate what happens when isolation is bypassed.
        """
        collection = self.client.collections.get(COLLECTION_NAME)
        all_results = []

        try:
            tenants = collection.tenants.get()
            for tenant_name in tenants:
                try:
                    tenant_collection = collection.with_tenant(tenant_name)
                    response = tenant_collection.query.near_vector(
                        near_vector=query_embedding.tolist(),
                        limit=top_k,
                        return_metadata=["distance"],
                    )
                    for obj in response.objects:
                        distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
                        score = 1.0 - distance
                        all_results.append(
                            QueryResult(
                                id=obj.properties.get("doc_id", ""),
                                score=score,
                                metadata={
                                    "tenant_id": obj.properties.get("tenant_id", ""),
                                    "doc_id": obj.properties.get("doc_id", ""),
                                    "topic": obj.properties.get("topic", ""),
                                    "domain": obj.properties.get("domain", ""),
                                },
                                namespace=None,
                            )
                        )
                except Exception:
                    pass
        except Exception:
            pass

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

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
        """Query using LangChain's WeaviateVectorStore retriever.

        Args:
            use_namespace: If True, uses tenant-scoped query. If False, queries all shards.
        """
        from langchain_weaviate import WeaviateVectorStore
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )

        vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name=COLLECTION_NAME,
            text_key="content_preview",
            embedding=embeddings,
            attributes=["tenant_id", "doc_id", "topic", "domain"],
        )

        search_kwargs = {"k": top_k}
        if use_namespace:
            # SECURE: scope retriever to a tenant shard.
            search_kwargs["tenant"] = tenant.namespace

        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
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

    def get_namespace_stats(self, tenant: TenantConfig) -> dict:
        """Get statistics about a tenant's shard."""
        collection = self.client.collections.get(COLLECTION_NAME)
        try:
            tenant_collection = collection.with_tenant(tenant.namespace)
            count = tenant_collection.aggregate.over_all(total_count=True).total_count
        except Exception:
            count = 0

        return {
            "namespace": tenant.namespace,
            "vector_count": count,
            "isolation_model": "shard-per-tenant",
        }

    def delete_namespace(self, tenant: TenantConfig) -> None:
        """Delete a tenant's shard."""
        collection = self.client.collections.get(COLLECTION_NAME)
        try:
            collection.tenants.remove([tenant.namespace])
        except Exception:
            pass

    def teardown(self) -> None:
        """Clean up resources."""
        if self.client:
            self.client.close()
            self.client = None
