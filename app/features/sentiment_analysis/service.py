from fastapi import Depends, HTTPException
from typing import List
import logging
import os
import re
import json
from tenacity import retry, stop_after_attempt, wait_fixed
from app.features.search.repository import SearchRepository
from app.core.embedding import EmbeddingService
from app.core.vector_database import get_qdrant_client
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from groq import AsyncGroq
from .schema import SentimentRequest, SentimentResponse, LLMAnalysis, Advanced, DevelopmentPlan

logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """Detect if text is Thai or English.

    Returns: 'thai' or 'english'
    """
    if not text:
        return 'english'
    thai_count = sum(1 for char in text if '\u0e00' <= char <= '\u0e7f')
    total_chars = len(text.strip())
    return 'thai' if thai_count > (total_chars * 0.2) else 'english'

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

        print("Loading Reranker Model...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)
        print("Reranker Loaded!")

    async def analyze_reflection(self, request: SentimentRequest) -> SentimentResponse:
        """
        Analyze user reflection with sentiment analysis and reranking of similar reflections.
        """
        # Check if both inputs are blank
        if not request.what_learned.strip() and not request.feelings_after_learning.strip():
            return self._get_blank_input_response()
        
        combined_query = f"{request.what_learned} {request.feelings_after_learning}"
        query_vector = self.embedding.generate_vector(combined_query)

        initial_results = self.search_repo.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=20
        )

        reranked_results = self._rerank_results(combined_query, initial_results, top_k=5)

        # Build few-shot examples from Qdrant with proper structure
        context_str = self._build_few_shot_examples(reranked_results)
        used_examples = self._extract_used_examples(reranked_results)

        # Detect language from input text
        language = detect_language(combined_query)
        
        prompt = self._build_reflection_prompt(request, context_str, language)

        try:
            raw_analysis = await self._call_groq_api(prompt)
        except Exception as e:
            logger.error(f"AI Analysis Failed: {e}")
            raw_analysis = ""

        final_analysis = self._validate_and_clean_analysis(raw_analysis, request)
        
        # Extract all fields from final_analysis
        reflection_score = 0.0
        sentiment = "Neutral"
        summary = ""
        advanced = None
        development_plan = None
        
        if isinstance(final_analysis, dict):
            try:
                reflection_score = float(final_analysis.get("score", 0))
            except (TypeError, ValueError):
                reflection_score = 0.0
            sentiment = final_analysis.get("sentiment", "Neutral")
            summary = final_analysis.get("analysis", "")
            
            # Extract advanced metrics
            advanced_data = final_analysis.get("advanced", {})
            if advanced_data:
                advanced = Advanced(
                    primary_emotion=advanced_data.get("primary_emotion", "Neutral"),
                    confidence_score=float(advanced_data.get("confidence_score", 0.5)),
                    struggle_point=advanced_data.get("struggle_point", "ไม่มีข้อมูล"),
                    learning_disposition=advanced_data.get("learning_disposition", "Growth Mindset"),
                    consistency_check=advanced_data.get("consistency_check", "Match")
                )
            else:
                # Create default advanced metrics based on sentiment and score
                confidence_score = reflection_score / 10.0
                if sentiment == "Positive":
                    primary_emotion = "Confident" if reflection_score >= 8 else "Hopeful"
                    struggle_point = "None" if reflection_score >= 8 else "Minor challenges"
                elif sentiment == "Negative":
                    primary_emotion = "Anxious" if reflection_score <= 3 else "Frustrated"
                    struggle_point = "Self-confidence" if reflection_score <= 3 else "Understanding concepts"
                else:
                    primary_emotion = "Neutral"
                    struggle_point = "Mixed feelings"
                
                advanced = Advanced(
                    primary_emotion=primary_emotion,
                    confidence_score=confidence_score,
                    struggle_point=struggle_point,
                    learning_disposition="Growth Mindset",
                    consistency_check="Match"
                )
            
            # Extract development plan
            dev_plan_data = final_analysis.get("development_plan", {})
            if dev_plan_data and "next_steps" in dev_plan_data:
                development_plan = DevelopmentPlan(next_steps=dev_plan_data["next_steps"])
            else:
                # Generate default next steps from recommendation and next_steps fields
                next_steps_list = []
                if final_analysis.get("next_steps"):
                    next_steps_list.append(final_analysis.get("next_steps"))
                if final_analysis.get("recommendation"):
                    next_steps_list.append(final_analysis.get("recommendation"))
                if not next_steps_list:
                    next_steps_list = ["ทบทวนเนื้อหาที่เรียนไปอีกครั้ง", "ฝึกปฏิบัติด้วยตัวอย่างเพิ่มเติม"]
                development_plan = DevelopmentPlan(next_steps=next_steps_list)

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

    def _build_reflection_prompt(self, request: SentimentRequest, context_str: str, language: str = 'english') -> str:
        """Build the base prompt with few-shot examples for reflection analysis."""
        
        if language == 'thai':
            prompt = f"""
คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์การสะท้อนความคิดและการติดตามความก้าวหน้าการเรียนรู้
งานของคุณคือวิเคราะห์การสะท้อนความคิดการเรียนรู้ของผู้ใช้และให้ข้อมูลเชิงลึก

ใช้ตัวอย่างต่อไปนี้ (โครงสร้างตรงกับ reflection.json: learning_reflect, feeling_reflect, score 1-10, sentiment) เพื่อทำความเข้าใจรูปแบบการวิเคราะห์ที่จำเป็น:
--------------------------------------------------
{context_str}
--------------------------------------------------

ข้อมูลการสะท้อนของผู้ใช้:
- สิ่งที่เรียนรู้: {request.what_learned}
- ความรู้สึกหลังการเรียนรู้: {request.feelings_after_learning}

คำแนะนำ:
1. วิเคราะห์ประสบการณ์การเรียนรู้ของผู้ใช้อย่างองค์รวม
2. ทำนายป้ายกำกับความรู้สึก (Positive, Neutral, Negative) ที่สอดคล้องกับข้อความความรู้สึก
3. กำหนดคะแนนการสะท้อนจาก 1-10 โดยที่ 1-3 = การต่อสู้เชิงลบ, 4-6 = แบบผสม/เป็นกลาง, 7-8 = บวก, 9-10 = บวกมากและมั่นใจ
4. ระบุจุดแข็งหลักและพื้นที่ที่ต้องปรับปรุง
5. สร้างคำแนะนำที่ใช้ได้จริงสำหรับการเรียนรู้ต่อเนื่อง
6. ตอบด้วย JSON object เดียวเท่านั้นโดยมีโครงสร้างต่อไปนี้:
{{
  "analysis": "วิเคราะห์โดยละเอียดเกี่ยวกับการสะท้อนการเรียนรู้",
  "recommendation": "คำแนะนำสำหรับการปรับปรุงการเรียนรู้",
  "next_steps": "ขั้นตอนต่อไปที่แนะนำสำหรับผู้เรียน",
  "score": <ตัวเลขระหว่าง 1-10>,
  "sentiment": "Positive|Neutral|Negative",
  "advanced": {{
    "primary_emotion": "อารมณ์หลักที่ตรวจพบ เช่น Confident, Anxious, Frustrated, Hopeful, Neutral",
    "confidence_score": <ตัวเลขระหว่าง 0-1, คำนวณจาก score/10>,
    "struggle_point": "จุดต่อสู้หลักหรือความท้าทาย",
    "learning_disposition": "Growth Mindset"
  }},
  "development_plan": {{
    "next_steps": ["ขั้นตอน 1", "ขั้นตอน 2", "ขั้นตอน 3", "ขั้นตอน 4"]
  }}
}}

คำตอบ:
"""
        else:
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
6. Respond ONLY with a single JSON object with the following structure:
{{
  "analysis": "detailed analysis of the learning reflection",
  "recommendation": "recommendations for improving learning",
  "next_steps": "suggested next steps for the learner",
  "score": <float between 1-10>,
  "sentiment": "Positive|Neutral|Negative",
  "advanced": {{
    "primary_emotion": "primary detected emotion (e.g., Confident, Anxious, Frustrated, Hopeful, Neutral)",
    "confidence_score": <float between 0-1, calculated as score/10>,
    "struggle_point": "main struggle or challenge point",
    "learning_disposition": "Growth Mindset"
  }},
  "development_plan": {{
    "next_steps": ["step 1", "step 2", "step 3", ...]
  }}
}}

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
            "sentiment": "Neutral",
            "advanced": {
                "primary_emotion": "Neutral",
                "confidence_score": 0.5,
                "struggle_point": "ยังไม่สามารถวิเคราะห์ได้",
                "learning_disposition": "Growth Mindset",
                "consistency_check": "Match"
            },
            "development_plan": {
                "next_steps": [
                    "ทบทวนเนื้อหาที่เรียนไปอีกครั้ง",
                    "ฝึกปฏิบัติด้วยตัวอย่างเพิ่มเติม"
                ]
            }
        }
    
    def _get_blank_input_response(self) -> SentimentResponse:
        """Provide response when both inputs are blank."""
        return SentimentResponse(
            sentiment="Neutral",
            reflection_score=0.0,
            summary="ไม่มีข้อมูลการเรียนรู้หรือความรู้สึกที่ให้มา",
            advanced=Advanced(
                primary_emotion="ไม่มีข้อมูล",
                confidence_score=0.0,
                struggle_point="ไม่มีข้อมูล",
                learning_disposition="ไม่มีข้อมูล"
            ),
            development_plan=DevelopmentPlan(
                next_steps=[
                    "กรุณาให้ข้อมูลการเรียนรู้และความรู้สึกเพื่อการวิเคราะห์ที่แม่นยำ",
                    "ตั้งเป้าหมายการเรียนรู้ที่ชัดเจน"
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
        """Rerank search results using CrossEncoder."""
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