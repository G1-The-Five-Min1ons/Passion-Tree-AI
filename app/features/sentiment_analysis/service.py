from fastapi import Depends, HTTPException
from typing import List
import logging
import os
import re
import json
from tenacity import retry, stop_after_attempt, wait_fixed
from app.features.search.repository import SearchRepository
from app.features.sentiment_analysis.service import SentimentAnalysisService
from app.core.embedding import EmbeddingService
from app.core.vector_database import get_qdrant_client
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from groq import AsyncGroq
from .reflection_schema import ReflectionRequest, ReflectionResponse

logger = logging.getLogger(__name__)

def get_search_repository(client: QdrantClient = Depends(get_qdrant_client)) -> SearchRepository:
    return SearchRepository(client=client)

class ReflectionService:
    def __init__(self,
        search_repo: SearchRepository = Depends(get_search_repository),
        embedding: EmbeddingService = Depends(EmbeddingService),
        sentiment_service: SentimentAnalysisService = Depends(SentimentAnalysisService)
    ):
        self.search_repo = search_repo
        self.embedding = embedding
        self.sentiment_service = sentiment_service
        self.collection_name = "learning_reflections"
        
        print("Loading Reranker Model...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)
        print("Reranker Loaded!")

    async def analyze_reflection(self, request: ReflectionRequest) -> ReflectionResponse:
        """
        Analyze user reflection with sentiment analysis and reranking of similar reflections.
        """
        logger.info(f"Analyzing reflection for topic: {request.what_learned}")

        sentiment = self.sentiment_service.analyze(request.feelings_after_learning)
        
        combined_query = f"{request.what_learned} {request.feelings_after_learning}"
        query_vector = self.embedding.generate_vector(combined_query)
        
        initial_results = self.search_repo.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=20
        )
        
        reranked_results = self._rerank_results(combined_query, initial_results, top_k=5)
        
        context_str = ""
        used_examples = []
        for i, res in enumerate(reranked_results):
            reflection_text = res.payload.get("reflection", "")
            analysis = res.payload.get("analysis", "")
            
            used_examples.append(reflection_text)
            context_str += f"[Example {i+1}]\nUser Reflection: {reflection_text}\nAI Analysis: {analysis}\n\n"
        
        prompt = self._build_reflection_prompt(request, context_str)
        
        try:
            raw_analysis = await self._call_groq_api(prompt)
        except Exception as e:
            logger.error(f"AI Analysis Failed: {e}")
            raw_analysis = ""
        
        final_analysis = self._validate_and_clean_analysis(raw_analysis, request)
        
        overall_score = self._calculate_reflection_score(request)
        
        return ReflectionResponse(
            summary=final_analysis,
            sentiment=sentiment,
            overall_score=overall_score,
            reranked_results=used_examples
        )

    def _build_reflection_prompt(self, request: ReflectionRequest, context_str: str) -> str:
        """Build the base prompt with few-shot examples for reflection analysis."""
        prompt = f"""
You are an expert learning coach specializing in reflection analysis and progress tracking.
Your task is to analyze the user's learning reflection and provide insightful feedback.

Use the following examples to understand the required analysis format:
--------------------------------------------------
{context_str}
--------------------------------------------------

User's Reflection Input:
- What Learned: {request.what_learned}
- Mood (1-5): {request.mood}
- Feelings After Learning: {request.feelings_after_learning}
- Progress (1-5): {request.progress}
- Challenge Level (1-5): {request.challenge_level}

Instructions:
1. Analyze the user's learning experience holistically
2. Provide sentiment-aware feedback based on their mood and feelings
3. Identify key strengths and areas for improvement
4. Generate actionable recommendations for continued learning
5. Format output as JSON with keys: analysis, recommendation, next_steps

Answer:
"""
        return prompt

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API with retry logic."""
        try:
            client = AsyncGroq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )

            chat_completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.4,
                max_tokens=1500,
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq API Error: {e}", exc_info=True)
            return "Error: Unable to generate analysis at this moment."
    
    def _validate_and_clean_analysis(self, text: str, request: ReflectionRequest) -> str:
        """Validate and clean the LLM output."""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group())
                return analysis_json
            else:
                logger.warning(f"No JSON found in response: {text}")
                return self._get_fallback_analysis(request)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {text}")
            return self._get_fallback_analysis(request)

    def _get_fallback_analysis(self, request: ReflectionRequest) -> dict:
        """Provide fallback analysis when LLM fails."""
        return {
            "analysis": f"You've made progress in learning {request.what_learned}. With a challenge level of {request.challenge_level}/5 and progress of {request.progress}/5, you're on the right track.",
            "recommendation": "Continue consistent practice and break down complex concepts into smaller parts.",
            "next_steps": "Review the material again, practice with examples, and seek clarification on challenging areas."
        }
    
    def _calculate_reflection_score(self, request: ReflectionRequest) -> float:
        """
        Calculate overall reflection score (0-10) based on:
        - Mood (1-5): Higher mood = better score
        - Progress (1-5): Higher progress = better score
        - Challenge Level (1-5): Balanced challenge contributes to score
        """

        #ตรงนี้เป็นการให้คะแนน reflect เดี๋ยวค่อยมาปรับทีหลังได้ตามความเหมาะสม

        # Normalize mood (1-5 to 0-1)
        mood_score = (request.mood - 1) / 4 * 100
        
        # Normalize progress (1-5 to 0-1)
        progress_score = (request.progress - 1) / 4 * 100
        
        challenge_optimal = abs(request.challenge_level - 3)
        challenge_score = max(0, 100 - (challenge_optimal * 15))
        
        overall_score = (mood_score * 0.3) + (progress_score * 0.4) + (challenge_score * 0.3)
        
        return round(overall_score, 2)
    
    def _rerank_results(self, query: str, results: list, top_k: int = 5) -> list:
        """Rerank search results using CrossEncoder."""
        if not results:
            return []
        
        pairs = [
            [query, f"{hit.payload.get('reflection', '')} {hit.payload.get('analysis', '')}"]
            for hit in results
        ]
        
        scores = self.reranker.predict(pairs)
        results_with_scores = list(zip(results, scores))
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        final_results = [hit for hit, score in results_with_scores[:top_k]]
        
        return final_results
