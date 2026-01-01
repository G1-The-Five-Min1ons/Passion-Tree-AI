from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastapi import Depends
from typing import List, Optional, Dict, Any
from app.core.vector_database import get_qdrant_client
from app.features.search.schemas import SearchResult

class SearchRepository:
    def __init__(self, client: QdrantClient = Depends(get_qdrant_client), collection_name: str = "learning_paths"):
        self.client = client
        self.collection_name = collection_name

    def search_collection(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: str = "learning_paths"
    ) -> List[SearchResult]:
        query_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            query_filter = models.Filter(must=must_conditions)

        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        )

        return [
            SearchResult(
                id=hit.id,
                score=hit.score,
                payload=hit.payload
            ) for hit in search_results
        ]

    def search_learning_paths(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        return self.search_collection(query_vector, top_k, filters, collection_name="learning_paths")

    def search_reflection_tree(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        return self.search_collection(query_vector, top_k, filters, collection_name="reflection_tree")

    def upsert_path(self, path_id: int, vector: List[float], payload: dict):
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
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[path_id])
        )