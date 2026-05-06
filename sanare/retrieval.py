from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalHit:
    id: str
    score: float
    text: str


class ClinicalRetriever:
    def search(self, query: str, limit: int = 3) -> list[RetrievalHit]:
        return []


class QdrantClinicalRetriever(ClinicalRetriever):
    def __init__(self, collection_name: str | None = None) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Install qdrant-client to use QdrantClinicalRetriever") from exc

        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "clinical_guidelines")
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

    def search(self, query: str, limit: int = 3) -> list[RetrievalHit]:
        # Embedding generation is intentionally injected outside this class in real deployments.
        # Without an embedding provider, this hook stays disabled by default.
        return []


def build_retriever() -> ClinicalRetriever:
    if os.getenv("QDRANT_URL"):
        try:
            return QdrantClinicalRetriever()
        except RuntimeError:
            return ClinicalRetriever()
    return ClinicalRetriever()

