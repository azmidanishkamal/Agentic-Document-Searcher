"""OpenAI embedding client for chunk text."""

from __future__ import annotations

from openai import OpenAI

from src.ingestion.config import EmbeddingConfig


class OpenAIEmbedder:
    def __init__(self, config: EmbeddingConfig, client: OpenAI | None = None) -> None:
        self.config = config
        self.client = client or OpenAI()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches of `config.batch_size`, preserving input order."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            response = self.client.embeddings.create(
                model=self.config.model,
                input=batch,
                dimensions=self.config.dimensions,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings
