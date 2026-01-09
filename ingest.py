import json
import asyncio
import os
from app.features.search.repository import SearchRepository
from app.core.embedding import EmbeddingService
from app.core.vector_database import get_qdrant_client, create_collection_if_not_exists

COLLECTION_NAME = "learning_paths_nodes"
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
        "learn-path-node.json",
    )
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} items. Processing vectors...")

    for idx, item in enumerate(data):
        question = item.get("Question", "")
        answer = item.get("Answer", "")

        vector = embedding_service.generate_vector(question)

        payload = {"question": question, "answer": answer, "original_id": idx}

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
