from fastapi import Depends, HTTPException
from typing import List, Dict, Any
import asyncio
import logging
import re
import json
import os
from tenacity import retry, stop_after_attempt, wait_fixed
from groq import AsyncGroq
from app.features.search.repository import SearchRepository
from app.core.embedding import EmbeddingService
from app.core.vector_database import get_qdrant_client
from qdrant_client import QdrantClient
from app.core.llm_client import call_groq_api
from app.core.reranker_store import get_reranker_service
from .schema import SentimentRequest, SentimentResponse, LLMAnalysis, Advanced, DevelopmentPlan

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
        self.reranker = get_reranker_service()
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

        final_analysis = self._validate_and_clean_analysis(raw_text, request)
        
        # Extract all fields from final_analysis (ใช้ค่าจาก LLM โดยตรง ไม่มี default)
        reflection_score = final_analysis.get("score", 0.0)
        sentiment = final_analysis.get("sentiment", "Neutral")
        summary = final_analysis.get("analysis", "")
        
        # Extract advanced metrics from LLM response
        advanced = None
        advanced_data = final_analysis.get("advanced", {})
        if advanced_data:
            advanced = Advanced(
                primary_emotion=advanced_data.get("primary_emotion"),
                confidence_score=float(advanced_data.get("confidence_score", 0.0)),
                struggle_point=advanced_data.get("struggle_point"),
                learning_disposition=advanced_data.get("learning_disposition"),
                consistency_check=advanced_data.get("consistency_check")
            )
        
        # Extract development plan from LLM response
        development_plan = None
        dev_plan_data = final_analysis.get("development_plan", {})
        if dev_plan_data and "next_steps" in dev_plan_data:
            development_plan = DevelopmentPlan(next_steps=dev_plan_data["next_steps"])

        return SentimentResponse(
            sentiment=sentiment,
            reflection_score=reflection_score,
            summary=summary,
            advanced=advanced,
            development_plan=development_plan,
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
        """Build the base prompt with few-shot examples for reflection analysis in Thai."""
        
        prompt = f"""
You are an expert in learning reflection analysis and tracking learning progress.
Your task is to analyze user's learning reflections and provide insights.

**Important: Users can input data in Thai or English, but you must respond in Thai only.**

Use the following examples (structure matches reflection.json: learning_reflect, feeling_reflect, score 1-10, sentiment) to understand the required analysis pattern:
--------------------------------------------------
{context_str}
--------------------------------------------------

User's reflection data (may be in Thai or English):
- What was learned: {request.what_learned}
- Feelings after learning: {request.feelings_after_learning}

Instructions:
1. Analyze the user's learning experience holistically, regardless of language used
2. Predict sentiment label (Positive, Neutral, Negative) that aligns with the feeling text
3. Assign a reflection score from 1-10 where 1-3 = negative struggle, 4-6 = mixed/neutral, 7-8 = positive, 9-10 = very positive and confident
4. Identify key strengths and areas for improvement
5. Generate actionable recommendations for continued learning
6. **Respond with a single JSON object only, and respond entirely in Thai regardless of the language users use for input** with the following structure:
{{
  "analysis": "Detailed analysis of the learning reflection (in Thai)",
  "recommendation": "Recommendations for improving learning (in Thai)",
  "next_steps": "Suggested next steps for the learner (in Thai)",
  "score": <number between 1-10>,
  "sentiment": "Positive|Neutral|Negative",
  "advanced": {{
    "primary_emotion": "Primary detected emotion in Thai, e.g., Confident, Anxious, Frustrated, Hopeful, Neutral",
    "confidence_score": <number between 0-1, calculated from score/10>,
    "struggle_point": "Main struggle point or challenge (in Thai)",
    "learning_disposition": "Growth Mindset"
  }},
  "development_plan": {{
    "next_steps": ["Step 1 (in Thai)", "Step 2 (in Thai)", "Step 3 (in Thai)", "Step 4 (in Thai)"]
  }}
}}

Response:
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

    def _validate_and_clean_analysis(self, text: str, request: SentimentRequest) -> dict:
        """Validate and clean the LLM output against the strict schema.

        - Extract JSON object from the text
        - Parse it
        - Validate shape and values using Pydantic
        - Normalize minor casing issues for sentiment
        - If parsing fails, raise exception to trigger retry
        """
        json_match = re.search(r'\{.*\}', text or "", re.DOTALL)
        if not json_match:
            logger.error("No JSON found in LLM response")
            raise ValueError("No JSON found in LLM response")

        raw_obj = json.loads(json_match.group())

        if not isinstance(raw_obj, dict):
            logger.error("LLM JSON is not an object")
            raise ValueError("LLM JSON is not an object")

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


    def _get_blank_input_response(self) -> SentimentResponse:
        """Provide response when both inputs are blank."""
        return SentimentResponse(
            sentiment="Neutral",
            reflection_score=0.0,
            summary="No learning data or feelings provided",
            advanced=Advanced(
                primary_emotion="ไม่มีข้อมูล",
                confidence_score=0.0,
                struggle_point="ไม่มีข้อมูล",
                learning_disposition="ไม่มีข้อมูล"
            ),
            development_plan=DevelopmentPlan(
                next_steps=[
                    "Please provide learning information and feelings for accurate analysis",
                    "Set clear learning goals"
                ]
            ),
            reranked_results=[]
        )
        
    def _get_reranker_safe(self):
        model = RerankerModelStore.get_model()
        if model is None:
             logger.warning("Model not ready yet. Force loading (Sync blocking)...")
             RerankerModelStore.load_model()
             model = RerankerModelStore.get_model()
        return model

    def _rerank_results(self, query: str, results: list, top_k: int = 5) -> list:
        """Rerank search results using Jina Reranker API."""
        if not results:
            return []

        pairs = [
            [query, f"{hit.payload.get('learning_reflect', '')} {hit.payload.get('feeling_reflect', '')}"]
            for hit in results
        ]

        scores = self.reranker.predict(pairs)
        results_with_scores = list(zip(results, scores))
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        final_results = [hit for hit, score in results_with_scores[:top_k]]

        return final_results
