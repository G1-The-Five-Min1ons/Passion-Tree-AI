from typing import List
from app.core.config import settings
from qdrant_client import QdrantClient
import logging
import requests

logger = logging.getLogger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v5-text-small"
JINA_DIMENSIONS = 384
JINA_TIMEOUT = 30


class EmbeddingService:
    def __init__(self, model_name: str = JINA_MODEL):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT,
            prefer_grpc=False,
        )
        self.model_name = model_name

        if not settings.JINA_API_KEY:
            raise RuntimeError("JINA_API_KEY is not configured")

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _embed(self, texts: List[str], task: str) -> List[List[float]]:
        payload = {
            "model": self.model_name,
            "task": task,
            "dimensions": JINA_DIMENSIONS,
            "normalized": True,
            "input": texts,
        }
        try:
            resp = self._session.post(JINA_API_URL, json=payload, timeout=JINA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Jina embedding request failed: {e}")
            raise Exception(f"Failed to generate embeddings: {e}")

        items = data.get("data") or []
        if len(items) != len(texts):
            raise Exception(
                f"Jina returned {len(items)} embeddings for {len(texts)} inputs"
            )
        # API may return items out of order; sort by index to be safe.
        items.sort(key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in items]

    def generate_vector(self, text: str, task: str = "retrieval.query") -> List[float]:
        return self._embed([text], task=task)[0]

    def generate_vectors_batch(
        self, texts: List[str], task: str = "retrieval.query"
    ) -> List[List[float]]:
        if not texts:
            return []
        logger.info(f"Embedding: Generating vectors for {len(texts)} texts (batch mode)")
        vectors = self._embed(texts, task=task)
        logger.info(f"Embedding: Generated {len(vectors)} vectors successfully")
        return vectors

    def prepare_learning_path_text(self, title: str, description: str) -> str:
        return f"Learning Path Title: {title}. Content Summary: {description}"

    def get_path_vector(self, title: str, description: str) -> List[float]:
        combined_text = self.prepare_learning_path_text(title, description)
        # Indexing a document, so use passage task for asymmetric retrieval.
        return self.generate_vector(combined_text, task="retrieval.passage")
