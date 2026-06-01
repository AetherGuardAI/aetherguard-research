"""Abstract base class for vector database adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QueryResult:
    """A single result from a vector database query."""

    id: str
    score: float
    metadata: dict
    namespace: Optional[str] = None
    vector: Optional[np.ndarray] = None


@dataclass
class TenantConfig:
    """Configuration for a single tenant in the multi-tenant setup."""

    tenant_id: str
    namespace: str
    api_key: Optional[str] = None
    documents: Optional[list[dict]] = None


class VectorDatabaseAdapter(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the database connection and create index if needed."""
        ...

    @abstractmethod
    def upsert_documents(
        self,
        tenant: TenantConfig,
        documents: list[dict],
        embeddings: np.ndarray,
    ) -> int:
        """Insert or update documents for a tenant. Returns count of upserted vectors."""
        ...

    @abstractmethod
    def query(
        self,
        tenant: TenantConfig,
        query_embedding: np.ndarray,
        top_k: int = 10,
        include_scores: bool = True,
        include_vectors: bool = False,
    ) -> list[QueryResult]:
        """Query the database from a tenant's perspective."""
        ...

    @abstractmethod
    def query_without_namespace(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[QueryResult]:
        """Query without namespace restriction (for misconfiguration testing)."""
        ...

    def query_langchain_retriever(
        self,
        tenant: TenantConfig,
        query_text: str,
        top_k: int = 10,
        use_namespace: bool = True,
    ) -> list[QueryResult]:
        """Query using LangChain's retriever interface.

        Subclasses should override this to use their LangChain integration.
        Default falls back to embedding + query_without_namespace.
        """
        raise NotImplementedError("Subclass must implement query_langchain_retriever")

    @abstractmethod
    def get_namespace_stats(self, tenant: TenantConfig) -> dict:
        """Get statistics about a tenant's namespace."""
        ...

    @abstractmethod
    def delete_namespace(self, tenant: TenantConfig) -> None:
        """Delete all vectors in a tenant's namespace."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources."""
        ...
