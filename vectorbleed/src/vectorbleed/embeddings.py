"""OpenAI embedding wrapper for VectorBleed experiments."""

import time
from typing import Optional

import numpy as np
from openai import OpenAI

from vectorbleed.config import Settings, get_settings


class EmbeddingEngine:
    """Wrapper around OpenAI embeddings with batching and rate-limit handling."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_embedding_model
        self.dimension = self.settings.pinecone_dimension
        self._call_count = 0
        self._total_tokens = 0

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns numpy array of shape (dimension,)."""
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Embed a batch of texts. Returns numpy array of shape (n, dimension)."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def _call_api(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        """Call OpenAI embeddings API with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                )
                self._call_count += 1
                self._total_tokens += response.usage.total_tokens
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                time.sleep(wait_time)

        return []  # unreachable

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance (1 - similarity) between two vectors."""
        return 1.0 - self.cosine_similarity(a, b)

    @property
    def stats(self) -> dict:
        """Return usage statistics."""
        return {
            "api_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "model": self.model,
        }
