"""Configuration management for VectorBleed experiments."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """VectorBleed configuration loaded from environment variables."""

    # OpenAI
    openai_api_key: str = Field(description="OpenAI API key for embeddings and corpus generation")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_generation_model: str = Field(default="gpt-4o-mini")

    # Pinecone
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    pinecone_index_name: str = Field(default="vectorbleed-research")
    pinecone_environment: str = Field(default="us-east-1")
    pinecone_dimension: int = Field(default=1536, description="Embedding dimension for text-embedding-3-small")

    # Weaviate (cloud)
    weaviate_url: str = Field(default="", description="Weaviate Cloud URL (e.g. https://xxx.weaviate.network)")
    weaviate_api_key: str = Field(default="", description="Weaviate Cloud API key")

    # Qdrant (cloud)
    qdrant_url: str = Field(default="", description="Qdrant Cloud URL (e.g. https://xxx.cloud.qdrant.io)")
    qdrant_api_key: str = Field(default="", description="Qdrant Cloud API key")

    # ChromaDB (cloud)
    chroma_host: str = Field(default="localhost_embedded", description="ChromaDB host or 'localhost_embedded' for in-process")
    chroma_port: int = Field(default=8000)
    chroma_tenant: str = Field(default="default_tenant")
    chroma_database: str = Field(default="default_database")
    chroma_api_key: str = Field(default="", description="ChromaDB Cloud API key (if using cloud)")

    # Tenant configuration
    tenant_a_namespace: str = Field(default="tenant_financial")
    tenant_b_namespace: str = Field(default="tenant_healthcare")
    corpus_size_per_tenant: int = Field(default=50)

    # Experiment settings
    experiment_probe_count: int = Field(default=100, description="Number of probe queries per experiment")
    experiment_inversion_samples: int = Field(default=1000, description="Alignment samples for Exp 5")
    similarity_threshold: float = Field(default=0.7, description="Cosine similarity threshold for anomaly detection")

    # Output
    results_dir: Path = Field(default=Path("results"))
    reports_dir: Path = Field(default=Path("results/reports"))

    # Logging
    log_level: str = Field(default="INFO")
    verbose: bool = Field(default=False)

    model_config = {"env_prefix": "VECTORBLEED_", "env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    """Load and return settings from environment."""
    return Settings()
