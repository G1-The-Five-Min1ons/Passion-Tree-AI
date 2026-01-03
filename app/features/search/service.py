from typing import List, Optional, Dict, Any
from fastapi import Depends
from app.features.search.repository import SearchRepository
from app.core.embedding import EmbeddingService
from app.features.search.schemas import SearchResponse
from app.core.vector_database import get_qdrant_client, create_collection_if_not_exists
from qdrant_client import QdrantClient
import logging

logger = logging.getLogger(__name__)

def get_search_repository(client: QdrantClient = Depends(get_qdrant_client)) -> SearchRepository:
    return SearchRepository(client=client)

def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

class SearchService:
    def __init__(
        self,
        repository: SearchRepository = Depends(get_search_repository),
        embedding: EmbeddingService = Depends(get_embedding_service)
    ):
        self.repository = repository
        self.embedding = embedding

    async def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        resource_type: str = "learning_paths" # ตั้ง Default เป็นชื่อ collection หลัก
    ) -> SearchResponse:
        try:
            logger.info(f"Searching in collection: {resource_type} with query: {query}")
            # แปลง Input Text เป็น Vector (ต้องได้ 384 dims ตาม Qdrant)
            vector = self.embedding.generate_vector(query)
            logger.info(f"Generated vector with {len(vector)} dimensions")
            
            # เรียกใช้ search แบบ Generic โดยส่งชื่อ collection เข้าไปตรงๆ
            results = self.repository.search(
                collection_name=resource_type, 
                query_vector=vector, 
                top_k=top_k, 
                filters=filters
            )
            
            logger.info(f"Search returned {len(results)} results")
            return SearchResponse(query=query, total=len(results), results=results)
        except Exception as e:
            logger.error(f"Search Error: {e}", exc_info=True)
            return SearchResponse(query=query, total=0, results=[])

    async def sync_upsert(self, collection_name: str, path_id: int, title: str, description: str, metadata: dict):
        # สร้าง Vector จาก Title + Description
        vector = self.embedding.get_path_vector(title, description)
        
        # รวม title, description เข้ากับ metadata
        payload = {
            "title": title,
            "description": description,
            **metadata  # เพิ่ม metadata อื่นๆ เช่น category_id, difficulty
        }
        
        self.repository.upsert_point(
            collection_name=collection_name,
            point_id=path_id,
            vector=vector,
            payload=payload
        )

    async def sync_delete(self, collection_name: str, path_id: int):
        self.repository.delete_point(collection_name, path_id)

    def initialize_collections(self, collection_name: str = "learning_paths", vector_size: int = 384):
        """Initialize required Qdrant collections."""
        create_collection_if_not_exists(collection_name, vector_size)
        logger.info(f"Collection '{collection_name}' initialized successfully")

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information and sample points for debugging."""
        client = get_qdrant_client()
        collection_info = client.get_collection(collection_name)
        
        # Get some sample points
        points = client.scroll(
            collection_name=collection_name,
            limit=10,
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
            ],
            "total_scrolled": len(points[0])
        }