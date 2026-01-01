from typing import List, Optional, Dict, Any
from fastapi import Depends
from app.features.search.repository import SearchRepository
from app.core.embedding import EmbeddingService
from app.features.search.schemas import SearchResponse

class SearchService:
    def __init__(
        self,
        repository: SearchRepository = Depends(),
        embedding: EmbeddingService = Depends()
    ):
        self.repository = repository
        self.embedding = embedding

    async def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        resource_type: Optional[str] = None
    ) -> SearchResponse:
        try:
            vector = self.embedding.generate_vector(query)
            # Select collection or logic based on resource_type
            if resource_type == "learning_path" or resource_type is None:
                results = self.repository.search_learning_paths(vector, top_k, filters)
            elif resource_type == "reflection_tree":
                # Example: implement search for reflection_tree (add method in repository)
                results = self.repository.search_reflection_tree(vector, top_k, filters)
            else:
                # Default: fallback or raise error
                results = []
            return SearchResponse(query=query, total=len(results), results=results)
        except Exception as e:
            print(f"Search Error: {e}")
            return SearchResponse(query=query, total=0, results=[])

    async def sync_upsert(self, path_id: int, title: str, description: str, category_id: int):
        """รับข้อมูลจาก Go มาลง Qdrant (รวมถึงหมวดหมู่เพื่อใช้ Filter)"""
        vector = self.embedding.get_path_vector(title, description)
        
        payload = {
            "title": title, 
            "description": description, 
            "category_id": category_id
        }
        self.repository.upsert_path(path_id, vector, payload)

    async def sync_delete(self, path_id: int):
        self.repository.delete_path(path_id)