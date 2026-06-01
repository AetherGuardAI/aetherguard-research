"""Qdrant vector database adapter for multi-tenant isolation testing.

Qdrant uses collections with payload-based filtering for multi-tenancy.
This tests whether filter-at-query-time isolation holds under adversarial conditions.
"""

import time
import uuid
import warnings
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rich.console import Console

from vectorbleed.config import Settings, get_settings
from vectorbleed.databases.base import QueryResult, TenantConfig, VectorDatabaseAdapter

# Suppress opentelemetry deprecation warnings
warnings.filterwarnings("ignore", message="SelectableGroups dict interface is deprecated")

console = Console()

COLLECTION_NAME = "vectorbleed_docs"


class QdrantAdapter(VectorDatabaseAdapter):
    """Qdrant implementation with collection + payload filter multi-tenancy.

    Qdrant's isolation model relies on payload filtering at query time.
    All tenants share the same collection, isolation is enforced by
    filtering on tenant_id in the payload.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.client: Optional[QdrantClient] = None

    def initialize(self) -> None:
        """Initialize Qdrant Cloud client and ensure collection exists."""
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=120,
        )

        # Create collection if not exists
        collections = [c.name for c in self.client.get_collections().collections]

        if COLLECTION_NAME not in collections:
            console.print(f"[yellow]Creating Qdrant collection: {COLLECTION_NAME}[/yellow]")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.settings.pinecone_dimension,
                    distance=Distance.COSINE,
                ),
            )
            console.print(f"[green]✓ Collection '{COLLECTION_NAME}' created[/green]")
        else:
            console.print(f"[green]✓ Using existing collection: {COLLECTION_NAME}[/green]")

        # Ensure payload index exists for tenant_id filtering
        try:
            from qdrant_client.models import PayloadSchemaType
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tenant_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # Index may already exist

    def upsert_documents(
        self,
        tenant: TenantConfig,
        documents: list[dict],
        embeddings: np.ndarray,
    ) -> int:
        """Upsert documents with tenant_id in payload."""
        points = []
        for doc, embedding in zip(documents, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant.tenant_id}_{doc['id']}")),
                    vector=embedding.tolist(),
                    payload={
                        "tenant_id": tenant.tenant_id,
                        "doc_id": doc["id"],
                        "topic": doc.get("topic", ""),
                        "domain": doc.get("domain", ""),
                        "content_preview": doc.get("content", "")[:200],
                    },
                )
            )

        # Upsert in smaller batches for cloud reliability
        batch_size = 20
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)

        return len(points)

    def _query_points(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        query_filter: Optional[Filter] = None,
    ) -> list[QueryResult]:
        """Internal query using qdrant-client v1.18+ query_points API."""
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        results = []
        for point in response.points:
            results.append(
                QueryResult(
                    id=point.payload.get("doc_id", ""),
                    score=point.score,
                    metadata=point.payload,
                    namespace=None,
                )
            )

        return results

    def query(
        self,
        tenant: TenantConfig,
        query_embedding: np.ndarray,
        top_k: int = 10,
        include_scores: bool = True,
        include_vectors: bool = False,
    ) -> list[QueryResult]:
        """Query with tenant filter (SECURE pattern)."""
        tenant_filter = Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant.tenant_id),
                )
            ]
        )

        results = self._query_points(query_embedding, top_k, query_filter=tenant_filter)
        for r in results:
            r.namespace = tenant.namespace
        return results

    def query_without_namespace(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[QueryResult]:
        """Query WITHOUT tenant filter — the VULNERABLE pattern.

        In Qdrant, if the developer forgets to add the tenant_id filter,
        the query searches across ALL tenants in the shared collection.
        """
        return self._query_points(query_embedding, top_k, query_filter=None)

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
        """Query using LangChain's QdrantVectorStore retriever.

        Args:
            use_namespace: If True, uses tenant filter. If False, no filter (vulnerable).
        """
        from langchain_qdrant import QdrantVectorStore
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )

        vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )

        if use_namespace:
            # SECURE: with tenant filter
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": top_k,
                    "filter": Filter(
                        must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant.tenant_id))]
                    ),
                }
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
        count_result = self.client.count(
            collection_name=COLLECTION_NAME,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant.tenant_id),
                    )
                ]
            ),
        )

        collection_info = self.client.get_collection(COLLECTION_NAME)

        return {
            "namespace": tenant.namespace,
            "vector_count": count_result.count,
            "total_vector_count": collection_info.points_count,
            "isolation_model": "payload-filter",
        }

    def delete_namespace(self, tenant: TenantConfig) -> None:
        """Delete all vectors for a tenant."""
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant.tenant_id),
                    )
                ]
            ),
        )

    def teardown(self) -> None:
        """Clean up resources."""
        self.client = None
