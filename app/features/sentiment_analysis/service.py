from fastapi import Depends
from typing import Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_fixed
import logging
import os
import re
from groq import AsyncGroq
from .schemas import ReflectionAnalysisRequest, ReflectionAnalysisResponse

logger = logging.getLogger(__name__)

class ReflectionAnalysisService:
    """Service for analyzing learning reflections and generating reflection scores"""
    
    def __init__(self):
        self.rag_context = self._get_rag_context()
        self.system_prompt = self._get_system_prompt()
    
    def _get_rag_context(self) -> str:
        """Get RAG context with evaluation guidelines"""
        return """Reflection Scoring Rubric:
1–2: Extremely vague reflection, minimal effort, very low progress
3–4: Weak reflection, limited understanding, low progress
5–6: Basic reflection, some understanding, moderate progress
7–8: Clear learning, meaningful reflection, strong progress
9–10: Deep understanding, strong insight, high progress despite challenge

Additional rules:
    - Progress score <= 2 limits reflection score to max 4
    - Progress score <= 4 allows reflection score up to 10
    - Challenge score >= 4 boosts score only if progress >= 3"""
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for reflection evaluation"""
        return """You are a learning reflection evaluation assistant.

Your task is to evaluate a user's learning reflection and produce:
1. A single overall reflection score (1–10)
2. A short scoring conclusion (1 sentence)
3. Sentiment classification of the emotional reflection

You will be given:
- Evaluation guidelines (RAG context)
- A user reflection containing:
  • What the user learned (text)
  • How the user felt after learning (text)
  • Learning progress score (1–5)
  • Challenge level score (1–5)

Evaluation dimensions:
- Learning clarity and specificity
- Depth of understanding
- Emotional reflection quality (sentiment analysis)
- Learning progress score
- Relationship between challenge and progress

Sentiment analysis rules:
- Analyze only the "how the user felt" text.
- Classify sentiment as positive, neutral, or negative.
- Positive sentiment supports higher scores.
- Negative sentiment lowers the score unless progress is high.

Scoring rules:
- Output score must be an integer from 1 to 10.
- Progress score is the strongest factor.
- High challenge increases score only if progress ≥ 3.
- Vague reflections cap the score at 5.
- Do not invent information.

Conclusion rules:
- Exactly 1 sentence.
- Neutral and professional tone.
- Summarize WHY the score was assigned.
- Do not give advice or instructions.

Output MUST be JSON only:
{
  "reflection_score": <1-10>,
  "scoring_conclusion": "<exactly 1 sentence explaining the score>",
  "sentiment": "<positive|neutral|negative>"
}"""
    
    async def analyze_reflection(self, request: ReflectionAnalysisRequest) -> ReflectionAnalysisResponse:
        """Analyze user's learning reflection and return score and conclusion"""
        logger.info(f"Analyzing reflection with progress_score={request.learning_progress_score}, challenge_score={request.challenge_level_score}")
        
        # Build the prompt with user's reflection data
        user_prompt = f"""Evaluate the following learning reflection:

What the user learned:
"{request.what_learned}"

How the user felt:
"{request.how_felt}"

Learning progress score: {request.learning_progress_score}/5
Challenge level score: {request.challenge_level_score}/5

Evaluation guidelines:
{self.rag_context}"""
        
        try:
            raw_response = await self._call_groq_api(user_prompt)
            logger.debug(f"Raw response from Groq: {raw_response}")
            
            parsed_data = self._parse_evaluation_response(raw_response)

            sentiment = await self._analyze_sentiment(request.how_felt)
            
            return ReflectionAnalysisResponse(
                reflection_score=parsed_data["reflection_score"],
                scoring_conclusion=parsed_data["scoring_conclusion"],
                sentiment_classification=sentiment,
                used_context=[self.rag_context]
            )
            
        except Exception as e:
            logger.error(f"Reflection analysis failed: {e}", exc_info=True)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def _call_groq_api(self, user_prompt: str) -> str:
        """Call Groq API for reflection evaluation"""
        try:
            client = AsyncGroq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            
            chat_completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=512,
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API Error: {e}", exc_info=True)
            raise
    
    def _parse_evaluation_response(self, response: str) -> dict:
        """Parse the AI-generated evaluation response"""
        try:
            # Extract JSON from the response
            json_match = re.search(r'\{[^{}]*"reflection_score"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                logger.error(f"Could not find JSON in response: {response}")
                raise ValueError("Invalid response format from AI")
            
            import json
            parsed = json.loads(json_match.group())
            
            # Validate required fields
            if "reflection_score" not in parsed or "scoring_conclusion" not in parsed:
                raise ValueError("Missing required fields in AI response")
            
            # Ensure reflection_score is an integer between 1-10
            score = int(parsed["reflection_score"])
            if not 1 <= score <= 10:
                logger.warning(f"Score out of range: {score}, clamping to 1-10")
                score = max(1, min(10, score))
            
            return {
                "reflection_score": score,
                "scoring_conclusion": str(parsed.get("scoring_conclusion", "")).strip()
            }
            
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}", exc_info=True)
            raise
    
    async def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of the 'how_felt' text"""
        try:
            client = AsyncGroq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            
            sentiment_prompt = f"""Analyze the sentiment of the following text and respond with ONLY one word: positive, neutral, or negative.

Text: "{text}"

Response:"""
            
            chat_completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": sentiment_prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=10,
            )
            
            sentiment = chat_completion.choices[0].message.content.strip().lower()
            
            # Validate sentiment is one of the expected values
            if sentiment not in ["positive", "neutral", "negative"]:
                logger.warning(f"Unexpected sentiment value: {sentiment}, defaulting to neutral")
                sentiment = "neutral"
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "neutral"