from fastapi import Depends, HTTPException
from typing import List, Dict, Any
import asyncio
import logging
import re
import json
from app.features.search.repository import SearchRepository
from app.core.embedding import EmbeddingService
from app.core.vector_database import get_qdrant_client
from qdrant_client import QdrantClient
from app.core.llm_client import call_groq_api
from app.core.reranker_store import get_reranker_service
from .schema import SentimentRequest, SentimentResponse, LLMAnalysis

logger = logging.getLogger(__name__)

def get_search_repository(client: QdrantClient = Depends(get_qdrant_client)) -> SearchRepository:
    return SearchRepository(client=client)

class SentimentService:
    def __init__(self,
        search_repo: SearchRepository = Depends(get_search_repository),
        embedding: EmbeddingService = Depends(EmbeddingService)
    ):
        self.search_repo = search_repo
        self.embedding = embedding
        self.collection_name = "reflection_analysis"

    async def analyze_reflection(self, request: SentimentRequest) -> SentimentResponse:
        """
        Analyze user reflection with sentiment analysis and reranking of similar reflections.
        """
        combined_query = f"{request.what_learned} {request.feelings_after_learning}"
        logger.info(f"Analyzing reflection for query: {combined_query}")
        query_vector = await asyncio.to_thread(self.embedding.generate_vector, combined_query)
        
        initial_results = await asyncio.to_thread(
            self.search_repo.search,
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=20
        )
        
        reranked_results = await asyncio.to_thread(self._rerank_results, combined_query, initial_results, 5)
        
        # Build few-shot examples from Qdrant with proper structure
        context_str = self._build_few_shot_examples(reranked_results)
        used_examples = self._extract_used_examples(reranked_results)
        
        prompt = self._build_reflection_prompt(request, context_str)
        
        try:
            raw_text = await call_groq_api(prompt)
        except Exception as e:
            logger.error(f"AI Analysis Failed: {e}")
            raw_text = ""
        
        final_result = self._validate_and_clean_analysis(raw_text, request)
        reflection_score = 0.0
        sentiment = "Neutral"
        if isinstance(final_result, dict):
            try:
                reflection_score = float(final_result.get("score", 0))
            except (TypeError, ValueError):
                reflection_score = 0.0
            sentiment = final_result.get("sentiment", "Neutral")

        return SentimentResponse(
            sentiment=sentiment,
            reflection_score=reflection_score,
            reranked_results=used_examples
        )

    def _build_few_shot_examples(self, results: list) -> str:
        """
        Build few-shot examples from Qdrant results.
        Extracts learning_reflect, feeling_reflect, score, and sentiment from payload.
        """
        if not results:
            return ""
        
        context_str = ""
        for i, res in enumerate(results):
            payload = res.payload
            learning_reflect = payload.get("learning_reflect", "")
            feeling_reflect = payload.get("feeling_reflect", "")
            score = payload.get("score", 5)
            sentiment = payload.get("sentiment", "Neutral")
            
            context_str += f"[Example {i+1}]\nUser Input:\n- Learning Reflect: {learning_reflect}\n- Feeling Reflect: {feeling_reflect}\nExpected Output:\n- Score: {score}\n- Sentiment: {sentiment}\n\n"
        
        return context_str

    def _extract_used_examples(self, results: list) -> List[str]:
        """Extract learning_reflect texts from results for response."""
        used_examples = []
        for res in results:
            payload = res.payload
            learning_reflect = payload.get("learning_reflect", "")
            if learning_reflect:
                used_examples.append(learning_reflect)
        return used_examples

    def _build_reflection_prompt(self, request: SentimentRequest, context_str: str) -> str:
        """Build the base prompt with few-shot examples for reflection analysis."""
        prompt = f"""
You are an expert learning coach specializing in reflection analysis and progress tracking.
Your task is to analyze the user's learning reflection and provide insightful feedback.

Use the following examples (structure matches reflection.json: learning_reflect, feeling_reflect, score 1-10, sentiment) to understand the required analysis format:
--------------------------------------------------
{context_str}
--------------------------------------------------

User's Reflection Input:
- Learning Reflect: {request.what_learned}
- Feeling Reflect: {request.feelings_after_learning}

Instructions:
1. Analyze the user's learning experience holistically
2. Predict a sentiment label (Positive, Neutral, Negative) consistent with the feeling text
3. Assign a reflection score from 1-10 where 1-3 = negative struggle, 4-6 = mixed/neutral, 7-8 = positive, 9-10 = very positive and confident
4. Identify key strengths and areas for improvement
5. Generate actionable recommendations for continued learning
6. Respond ONLY with a single JSON object with keys: analysis, recommendation, next_steps, score, sentiment

Answer:
"""
        return prompt
    
    def _validate_and_clean_analysis(self, text: str, request: SentimentRequest) -> dict:
        """Validate and clean the LLM output against the strict schema.

        - Extract JSON object from the text
        - Parse it
        - Validate shape and values using Pydantic
        - Normalize minor casing issues for sentiment
        - Fallback to a safe default if anything fails
        """
        try:
            json_match = re.search(r'\{.*\}', text or "", re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response; using fallback.")
                return self._get_fallback_analysis(request)

            raw_obj = json.loads(json_match.group())

            if not isinstance(raw_obj, dict):
                logger.warning("LLM JSON is not an object; using fallback.")
                return self._get_fallback_analysis(request)

            # Best-effort normalization for sentiment casing if present
            sentiment = raw_obj.get("sentiment")
            if isinstance(sentiment, str):
                norm = sentiment.strip().lower()
                if norm in {"positive", "pos"}:
                    raw_obj["sentiment"] = "Positive"
                elif norm in {"neutral", "neu"}:
                    raw_obj["sentiment"] = "Neutral"
                elif norm in {"negative", "neg"}:
                    raw_obj["sentiment"] = "Negative"

            # Validate strictly with Pydantic
            validated = LLMAnalysis.model_validate(raw_obj)
            return validated.model_dump()

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}; using fallback.")
            return self._get_fallback_analysis(request)
        except Exception as e:
            logger.warning(f"LLM output validation failed: {e}; using fallback.")
            return self._get_fallback_analysis(request)

    def _get_fallback_analysis(self, request: SentimentRequest) -> dict:
        """Provide fallback analysis when LLM fails."""
        return {
            "analysis": f"You've made progress in learning {request.what_learned}.",
            "recommendation": "Continue consistent practice and break down complex concepts into smaller parts.",
            "next_steps": "Review the material again, practice with examples, and seek clarification on challenging areas.",
            "score": 7,
            "sentiment": "Neutral"
        }
        
    def _rerank_results(self, query: str, results: list, top_k: int = 5) -> list:
        """Rerank search results using Jina Reranker API."""
        if not results:
            return []
        
        pairs = [
            [query, f"{hit.payload.get('learning_reflect', '')} {hit.payload.get('feeling_reflect', '')}"]
            for hit in results
        ]
        
        reranker = get_reranker_service()
        scores = reranker.predict(pairs)
        results_with_scores = list(zip(results, scores))
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        final_results = [hit for hit, score in results_with_scores[:top_k]]
        
        return final_results

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information and all points for debugging."""
        client = get_qdrant_client()
        collection_info = client.get_collection(collection_name)
        
        # Get all points from collection
        points = client.scroll(
            collection_name=collection_name,
            limit=1000,  # Get up to 1000 points (adjust if needed)
            with_payload=True,
            with_vectors=False
        )
        
        return {
            "collection_name": collection_name,
            "points_count": collection_info.points_count if hasattr(collection_info, 'points_count') else 'N/A',
            "vectors_config": str(collection_info.config.params.vectors) if hasattr(collection_info.config.params, 'vectors') else 'N/A',
            "sample_points": [
                {"id": p.id, "payload": p.payload} 
                for p in points[0]
            ]
        }