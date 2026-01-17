from typing import List
from app.core.config import settings
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import logging
import os

logger = logging.getLogger(__name__)

class EmbeddingService:
    _model_cache = None  # Class-level cache to avoid re-downloading
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT,
            prefer_grpc=False
        )
        self.model_name = model_name
        
        # Ensure cache directory exists for model persistence
        cache_dir = settings.MODEL_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use cached model if available
        if EmbeddingService._model_cache is None:
            logger.info(f"Loading FastEmbed model: {model_name} (first time only)")
            # Set threads for ONNX Runtime to use all available CPU cores
            threads = os.cpu_count() or 4
            EmbeddingService._model_cache = TextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,  # Persist model in specified directory
                threads=threads  # Optimize for multi-threading
            )
            logger.info(f"FastEmbed initialized with {threads} threads at {cache_dir}")
        
        self.model = EmbeddingService._model_cache

    def generate_vector(self, text: str) -> List[float]:
        """Generate embeddings using FastEmbed with parallel processing."""
        try:
            # Use parallel parameter for better concurrent request handling
            # parallel=None uses optimal batch processing automatically
            embeddings = list(self.model.embed(
                [text],
                batch_size=32,      # Process multiple texts in batches
                parallel=None       # Auto-optimize parallelism
            ))
            vector = embeddings[0].tolist()
            logger.info(f"Embedding: Generated {len(vector)}-dimensional vector")
            return vector
            
        except Exception as e:
            logger.error(f"FastEmbed error: {e}")
            raise Exception(f"Failed to generate embeddings: {e}")
    
    def generate_vectors_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        try:
            logger.info(f"Embedding: Generating vectors for {len(texts)} texts (batch mode)")
            embeddings = list(self.model.embed(
                texts,
                batch_size=32,
                parallel=None
            ))
            vectors = [emb.tolist() for emb in embeddings]
            logger.info(f"Embedding: Generated {len(vectors)} vectors successfully")
            return vectors
            
        except Exception as e:
            logger.error(f"FastEmbed batch error: {e}")
            raise Exception(f"Failed to generate batch embeddings: {e}")

    def prepare_learning_path_text(self, title: str, description: str) -> str:
        return f"Learning Path Title: {title}. Content Summary: {description}"

    def get_path_vector(self, title: str, description: str) -> List[float]:
        combined_text = self.prepare_learning_path_text(title, description)
        return self.generate_vector(combined_text)
