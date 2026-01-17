import logging
import requests
from typing import List, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)

class JinaRerankerService:
    """Jina Reranker API service for reranking search results."""
    
    def __init__(self):
        self.api_url = "https://api.jina.ai/v1/rerank"
        self.model = "jina-reranker-v2-base-multilingual"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.JINA_API_KEY}"
        }
    
    def predict(self, pairs: List[List[str]]) -> List[float]:
        """Rerank query-document pairs using Jina API.
        
        Args:
            pairs: List of [query, document] pairs
            
        Returns:
            List of relevance scores
        """
        if not pairs:
            return []
        
        # Extract query (same for all pairs) and documents
        query = pairs[0][0]
        documents = [pair[1] for pair in pairs]
        
        logger.info(f"Reranker: Processing {len(documents)} documents...")
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents)  # Return all documents with scores
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            
            results = response.json()
            
            # Jina returns results sorted by relevance_score
            # We need to map them back to original order
            scores = [0.0] * len(documents)
            for result in results.get("results", []):
                idx = result["index"]
                score = result["relevance_score"]
                scores[idx] = score
            
            logger.info(f"Reranker: Completed (score range: [{min(scores):.3f}, {max(scores):.3f}])")
            return scores
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Jina Reranker API error: {e}")
            # Return neutral scores on error
            return [0.5] * len(documents)
        except Exception as e:
            logger.error(f"Unexpected error in Jina reranking: {e}")
            return [0.5] * len(documents)

# Singleton instance
_reranker_service = None

def get_reranker_service() -> JinaRerankerService:
    """Get or create singleton reranker service instance."""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = JinaRerankerService()
    return _reranker_service