from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastapi import Depends
from typing import List, Optional

from app.core.vector_database import get_qdrant_client
from app.features.search.schemas import SearchResult, LearningPathPayload

class SearchRepository:
    def __init__(self, client: QdrantClient = Depends(get_qdrant_client)):
        self.client = client
        self.collection_name = "learning_paths"

    def search_learning_paths(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        return [
            SearchResult(
                id=hit.id,
                score=hit.score,
                payload=LearningPathPayload(
                    title=hit.payload.get("title", ""),
                    description=hit.payload.get("description", "")
                )
            ) for hit in search_results
        ]

    def upsert_path(self, path_id: int, vector: List[float], payload: dict):
        """for Create and Update"""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=path_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def delete_path(self, path_id: int):
        """for deleting data when the main SQL is deleted"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[path_id])
        )