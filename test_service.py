import sys
import json
import asyncio
import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

from app.features.search.repository import SearchRepository
from app.core.embedding import EmbeddingService
from app.core.vector_database import get_qdrant_client, create_collection_if_not_exists

COLLECTION_NAME = "reflection_analysis"
VECTOR_SIZE = 384

async def ingest_data():
    print(f"Starting ingestion into '{COLLECTION_NAME}'...")

    client = get_qdrant_client()
    repo = SearchRepository(client)
    embedding_service = EmbeddingService()

    try:
        create_collection_if_not_exists(COLLECTION_NAME, VECTOR_SIZE)
        print(f"Collection '{COLLECTION_NAME}' is ready.")
    except Exception as e:
        print(f"Collection check skipped or failed: {e}")

    file_path = os.path.join(
        "..",
        "Passion-Tree-Infrastructure",
        "API Docs",
        "Mock-Data",
        "reflection.json",
    )
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} items. Processing vectors...")

    for idx, item in enumerate(data):
        question_obj = item.get("question", {})
        
        # Extract learning and feeling from question object
        if isinstance(question_obj, dict):
            learning = question_obj.get("learning_reflect", "")
            feeling = question_obj.get("feeling_reflect", "")
        else:
            learning = ""
            feeling = ""
        
        question_text = f"{learning} {feeling}".strip()
        
        answer_obj = item.get("answer", {})
        reflection_score = answer_obj.get("reflection_score", 0)
        sentiment = answer_obj.get("sentiment", "")
        summary = answer_obj.get("summary", "")
        advanced = answer_obj.get("advanced", {})
        development_plan = answer_obj.get("development_plan", {})

        vector = embedding_service.generate_vector(question_text)

        payload = {
            "learning_reflect": learning,
            "feeling_reflect": feeling,
            "reflection_score": reflection_score,
            "sentiment": sentiment,
            "summary": summary,
            "advanced": advanced,
            "development_plan": development_plan,
            "original_id": idx
        }

        repo.upsert_point(
            collection_name=COLLECTION_NAME,
            point_id=idx,
            vector=vector,
            payload=payload,
        )

        if idx % 5 == 0:
            print(f"   - Processed {idx+1}/{len(data)} items...")

    print("Ingestion Complete! Data is ready for RAG.")


if __name__ == "__main__":
    asyncio.run(ingest_data())