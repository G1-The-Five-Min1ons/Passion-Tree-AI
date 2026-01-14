import asyncio
import logging
import re
from fastapi import Depends
from app.features.search.repository import SearchRepository
from app.core.embedding import EmbeddingService
from app.core.vector_database import get_qdrant_client
from qdrant_client import QdrantClient
from app.core.llm_client import call_groq_api
from app.core.reranker_store import RerankerModelStore
from .schema import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)

def get_search_repository(client: QdrantClient = Depends(get_qdrant_client)) -> SearchRepository:
    return SearchRepository(client=client)

class GeneratorService:
    def __init__(
        self,
        search_repo: SearchRepository = Depends(get_search_repository),
        embedding: EmbeddingService = Depends(EmbeddingService)
    ):
        self.search_repo = search_repo
        self.embedding = embedding
        self.collection_name = "learning_paths_nodes"
        self.reranker_model = None

    async def generate_learning_path(self, request: GenerateRequest) -> GenerateResponse:
        query = request.topic
        logger.info(f"Generating path for query: {query}")
        query_vector = await asyncio.to_thread(self.embedding.generate_vector, query)
        
        initial_results = await asyncio.to_thread(
            self.search_repo.search,
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=20
        )
        
        reranked_results = await asyncio.to_thread(self._rerank_results, query, initial_results, 5)

        context_str = ""
        used_examples = []
        for i, res in enumerate(reranked_results):
            q_text = res.payload.get("question", "")
            a_text = res.payload.get("answer", "")
            
            used_examples.append(f"Q: {q_text}")
            
            context_str += f"[Example {i+1}]\nUser Requirement: {q_text}\nDesired Output: {a_text}\n\n"

        prompt = f"""
        You are an expert curriculum designer. 
        Your task is to generate a structured learning path based on the user's request.
        
        Use the following examples to understand the required JSON-like string format:
        --------------------------------------------------
        {context_str}
        --------------------------------------------------
        
        Current Request: "{query}"
        
        Instructions:
        1. Create a step-by-step learning path relevant to the request.
        2. STRICTLY follow the output format: "Node 1: [Topic], Node 2: [Topic], ..."
        3. Do not add introductions or explanations. Just the nodes.
        
        Answer:
        """

        try:
            raw_text = await call_groq_api(prompt)
        except Exception as e:
            logger.error(f"AI Generation Failed after retries: {e}")
            raw_text = ""
        
        final_result = self._validate_and_clean_output(raw_text, query)
        
        return GenerateResponse(
            topic=query,
            result=final_result,
            used_context=used_examples
        )
        
    def _validate_and_clean_output(self, text: str, topic: str) -> str:
        pattern = r"(Node\s+\d+:\s+[^,]+)"
        matches = re.findall(pattern, text)
        
        if not matches:
            logger.critical(f"AI Output Malformed for topic '{topic}': {text}")
            return self._get_fallback_path(topic)

        return ", ".join(matches)

    def _get_fallback_path(self, topic: str) -> str:
        return (
            f"Node 1: Introduction to {topic}, "
            f"Node 2: Key Concepts of {topic}, "
            f"Node 3: Practice and Implementation, "
            f"Node 4: Advanced Topics in {topic}"
        )
        
    def _get_reranker_safe(self):
        model = RerankerModelStore.get_model()
        if model is None:
             logger.warning("Model not ready yet. Force loading (Sync blocking)...")
             RerankerModelStore.load_model()
             model = RerankerModelStore.get_model()
        return model
        
    def _rerank_results(self, query: str, results: list, top_k: int = 5) -> list:
        if not results:
            return []
        
        pairs = [
            [query, f"{hit.payload.get('question', '')} {hit.payload.get('answer', '')}"] 
            for hit in results
        ]
        
        reranker = self._get_reranker_safe()
        scores = reranker.predict(pairs)
        results_with_scores = list(zip(results, scores))
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        final_results = [hit for hit, score in results_with_scores[:top_k]]
        
        return final_results