import asyncio, logging, re
import time
from fastapi import Depends
from qdrant_client import QdrantClient
from app.core.embedding import EmbeddingService
from app.core.vector_database import get_qdrant_client
from app.core.llm_client import call_groq_api
from app.core.reranker_store import get_reranker_service, RerankerUnavailableError
from app.features.search.repository import SearchRepository
from app.features.generator import prompts
from .schema import GenerateRequest, GenerateResponse

logger = logging.getLogger(f"passion-tree-ai.{__name__}")

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
        self.source_context = "app.features.generator.service"

    async def generate_learning_path(self, request: GenerateRequest) -> GenerateResponse:
        query = request.topic
        start_total = time.perf_counter()
        
        logger.info(f"[{self.source_context}] Starting generation for: '{query}'")
        
        try:
            # Embedding Stage
            query_vector = await asyncio.to_thread(self.embedding.generate_vector, query)
            
            # Retrieval Stage
            initial_results = await asyncio.to_thread(
                self.search_repo.search,
                collection_name=self.collection_name,
                query_vector=query_vector,
                top_k=20
            )
            
            # Reranking Stage
            reranked_results = await asyncio.to_thread(self._rerank_results, query, initial_results, 5)
            
            # LLM Generation Stage
            context_str = ""
            used_examples = []
            for i, res in enumerate(reranked_results):
                q_text = res.payload.get("question", "")
                a_text = res.payload.get("answer", "")
                used_examples.append(f"Q: {q_text}")
                context_str += f"[Example {i+1}]\nUser Requirement: {q_text}\nDesired Output: {a_text}\n\n"
                
            t3 = time.perf_counter()
            prompt = prompts.LEARNING_PATH_PROMPT.format(
                context_str=context_str,
                query=query
            )
            
            raw_text = await call_groq_api(prompt)
            logger.info(f"LLM Response received in {time.perf_counter() - t3:.4f}s")
            
            # Validation
            final_result = self._validate_and_clean_output(raw_text, query)
            
            total_duration = time.perf_counter() - start_total
            logger.info(f"SUCCESS: Learning path generated for '{query}' | Total Time: {total_duration:.2f}s")
            
            return GenerateResponse(
                topic=query,
                result=final_result,
                used_context=used_examples
            )
            
        except Exception as e:
            logger.error(f"AI Generation Failed after retries: {e}")
            raw_text = ""
            raise
        
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
        
    def _rerank_results(self, query: str, results: list, top_k: int = 5) -> list:
        if not results:
            return []

        pairs = [
            [query, f"{hit.payload.get('question', '')} {hit.payload.get('answer', '')}"]
            for hit in results
        ]

        reranker = get_reranker_service()
        try:
            scores = reranker.predict(pairs)
        except RerankerUnavailableError:
            logger.warning(
                "Reranker unavailable; falling back to vector-search order for top %d results",
                top_k,
            )
            return results[:top_k]

        results_with_scores = list(zip(results, scores))
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        final_results = [hit for hit, _ in results_with_scores[:top_k]]

        return final_results