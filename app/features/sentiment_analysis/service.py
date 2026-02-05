from fastapi import Depends
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
from .schema import SentimentRequest, SentimentResponse, LLMAnalysis, Advanced, DevelopmentPlan
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

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
        combined_query = f"{request.learning_reflect} {request.mood_reflect}"
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

        # Retry mechanism with feedback loop
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                prompt = self._build_reflection_prompt(request, context_str, previous_error=last_error)
                raw_text = await call_groq_api(prompt)
                final_analysis = self._validate_and_clean_analysis(raw_text, request)
                break  # Success - exit retry loop
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt == max_attempts - 1:
                    # All retries exhausted
                    raise
                # Continue to next attempt with error feedback
                await asyncio.sleep(2)
        else:
            # This shouldn't happen but just in case
            logger.error(f"Failed to get valid analysis after all retries")

            return SentimentResponse(
                summary="Unable to analyze reflection at this moment. Please try again later.",
                sentiment_analysis="Neutral",
                struggle_point="Unable to identify at this moment",
                development_plan=["Please try submitting your reflection again for detailed analysis."],
                ai_confident_score=0.0,
                reflection_score=0.0,
                weighted_reflection_score=0.0
            )
        
        # Extract fields from LLM response (validated by Pydantic)
        reflection_score = final_analysis["score"]
        sentiment = final_analysis["sentiment"]
        summary = final_analysis["analysis"]
        ai_confident_score = final_analysis["ai_confident_score"]
        
        # Extract advanced metrics
        advanced_data = final_analysis["advanced"]
        primary_emotion = advanced_data.get("primary_emotion")
        struggle_point = advanced_data["struggle_point"]
        
        # Extract development plan
        development_plan = final_analysis["development_plan"]["next_steps"]
        
        # Calculate weighted reflection score
        normalized_self_score = self._calculate_weighted_self_score(request)
        weighted_reflection_score = reflection_score * 0.65 + normalized_self_score * 0.35

        return SentimentResponse(
            summary=summary,
            sentiment_analysis=sentiment,
            primary_emotion=primary_emotion,
            struggle_point=struggle_point,
            development_plan=development_plan,
            ai_confident_score=round(ai_confident_score, 2),
            reflection_score=round(reflection_score, 1),
            weighted_reflection_score=round(weighted_reflection_score, 2)
        )

    def _build_few_shot_examples(self, results: list) -> str:
        """
        Build few-shot examples from Qdrant results.
        Extracts summary, sentiment, primary_emotion, struggle_point, and score from payload.
        """
        if not results:
            return ""

        context_str = ""
        for i, res in enumerate(results):
            payload = res.payload
            summary = payload.get("summary", "")
            sentiment = payload.get("sentiment", "Neutral")
            primary_emotion = payload.get("primary_emotion", "")
            struggle_point = payload.get("struggle_point", "")
            development_plan = payload.get("development_plan", "")
            score = payload.get("reflection_score", 5)

            context_str += f"""[Example {i+1}]
Summary: {summary}
Sentiment: {sentiment}
Primary Emotion: {primary_emotion}
Struggle Point: {struggle_point}
Development Plan: {development_plan}
Reflection Score: {score}

"""

        return context_str

    def _calculate_weighted_self_score(self, request: SentimentRequest) -> float:
        """Calculate weighted self-score: progress 50%, feel 35%, challenge 15%"""
        avg_self_score = (request.progress_score * 0.5 +
                        request.feel_score * 0.35 +
                        request.challenge_score * 0.15)
        return avg_self_score * 2  # Scale from 0-5 to 0-10

    def _build_reflection_prompt(self, request: SentimentRequest, context_str: str, previous_error: str = None) -> str:
        """Build the base prompt with few-shot examples for reflection analysis in Thai."""
        
        # Add error feedback section if there was a previous error
        error_feedback = ""
        if previous_error:
            error_feedback = f"""
**IMPORTANT - PREVIOUS ATTEMPT FAILED:**
Your previous response had the following error: {previous_error}

Please fix this error and ensure your response follows the exact JSON structure specified below.
Common issues to avoid:
- Missing required fields
- Incorrect field types (e.g., string instead of number)
- Invalid sentiment values (must be exactly: "Positive", "Neutral", or "Negative")
- Malformed JSON syntax

"""
        
        prompt = f"""
You are an expert in learning reflection analysis and tracking learning progress.
Your task is to analyze user's learning reflections and provide comprehensive insights.

**Important: Users can input data in Thai or English, but you must respond in Thai only.**

{error_feedback}Use the following examples to understand the required analysis pattern:
--------------------------------------------------
{context_str}
--------------------------------------------------

User's reflection data (may be in Thai or English):
- Learning reflection: {request.learning_reflect}
- Mood reflection: {request.mood_reflect}
- Self-assessed feel score: {request.feel_score}/5
- Self-assessed progress score: {request.progress_score}/5
- Self-assessed challenge score: {request.challenge_score}/5

Instructions:
1. Analyze the user's learning experience holistically, considering both text and numeric self-assessments
2. Predict sentiment label (Positive, Neutral, Negative) that aligns with the overall reflection
3. Assign a reflection score from 0-10 based on text quality and depth of reflection:
   - 0-3 = superficial or negative reflection with limited insight
   - 4-6 = moderate reflection with some self-awareness
   - 7-8 = good reflection showing clear understanding
   - 9-10 = exceptional reflection with deep metacognition
4. Assess your AI confidence score (0-1) in the analysis based on:
   - Clarity and completeness of the user's reflection text
   - Consistency between mood_reflect and numeric scores
   - Sufficient information to make accurate assessment
   - Low confidence (0.3-0.5): vague, incomplete, or inconsistent reflection
   - Medium confidence (0.5-0.7): adequate but some gaps
   - High confidence (0.7-1.0): clear, detailed, consistent reflection
5. Identify the primary emotion and main struggle point
6. Generate a detailed, actionable development plan with specific, in-depth steps
   - Each step should be comprehensive and detailed (2-4 sentences per step)
   - Include specific tools, techniques, or resources
   - Provide concrete examples and actionable advice
   - Focus on practical implementation details
7. **Respond with a single JSON object only, and respond entirely in Thai regardless of input language** with this exact structure:
{{
  "analysis": "Detailed summary analyzing the student's reflection, combining what they learned and how they feel (in Thai, 2-3 sentences)",
  "recommendation": "Brief recommendations for improving learning (in Thai)",
  "next_steps": "Overview of suggested next steps (in Thai)",
  "score": <number between 0-10>,
  "sentiment": "Positive|Neutral|Negative",
  "ai_confident_score": <number between 0-1, your confidence in this analysis>,
  "advanced": {{
    "primary_emotion": "Primary detected emotion in Thai, e.g., Confident, Anxious, Frustrated, Hopeful, Curious, Overwhelmed, Determined",
    "confidence_score": <number between 0-1, calculated from score/10>,
    "struggle_point": "Main struggle point or challenge identified from the reflection (in Thai)",
    "learning_disposition": "Growth Mindset"
  }},
  "development_plan": {{
    "next_steps": [
      "Step 1: [2-4 sentences with in-depth details, including specific tools, techniques, and clear examples]",
      "Step 2: [2-4 sentences with in-depth details, including workflow steps and specific best practices]",
      "Step 3: [2-4 sentences with in-depth details, including resources and practical methods that can be implemented]",
      "Step 4: [2-4 sentences with in-depth details, including practice approaches and progress tracking methods]"
    ]
  }}
}}

Response:
"""
        return prompt


    def _validate_and_clean_analysis(self, text: str, request: SentimentRequest) -> dict:
        """Validate and clean the LLM output against the strict schema.

        - Extract JSON object from the text
        - Parse it
        - Validate shape and values using Pydantic
        - Normalize minor casing issues for sentiment
        - If parsing fails, raise exception with clear error message for feedback
        """
        json_match = re.search(r'\{.*\}', text or "", re.DOTALL)
        if not json_match:
            logger.error("No JSON found in LLM response")
            raise ValueError("No JSON object found in response. Please provide a valid JSON object.")

        try:
            raw_obj = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise ValueError(f"Invalid JSON syntax: {str(e)}. Please check your JSON formatting.")

        if not isinstance(raw_obj, dict):
            logger.error("LLM JSON is not an object")
            raise ValueError("Response must be a JSON object (dictionary), not an array or primitive.")

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
        try:
            validated = LLMAnalysis.model_validate(raw_obj)
            return validated.model_dump()
        except Exception as e:
            logger.error(f"Pydantic validation error: {e}")
            raise ValueError(f"Schema validation failed: {str(e)}. Please ensure all required fields are present with correct types.")

    def _rerank_results(self, query: str, results: list, top_k: int = 5) -> list:
        """Rerank search results using Jina Reranker API."""
        if not results:
            return []

        pairs = [
            [query, f"{hit.payload.get('summary', '')} {hit.payload.get('struggle_point', '')}"]
            for hit in results
        ]

        scores = self.reranker.predict(pairs)
        results_with_scores = list(zip(results, scores))
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        final_results = [hit for hit, score in results_with_scores[:top_k]]

        return final_results
