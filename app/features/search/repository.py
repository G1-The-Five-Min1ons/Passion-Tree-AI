from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastapi import Depends
from typing import List, Optional, Dict, Any, Union
from app.features.search.schemas import SearchResult

class SearchRepository:
    def __init__(self, client: QdrantClient):
        self.client = client

    def _build_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        if not filters:
            return None

        must_conditions = []
        for key, value in filters.items():
            if isinstance(value, dict):
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        range=models.Range(**value)
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(must=must_conditions)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        with_payload: bool = True
    ) -> List[SearchResult]:
        query_filter = self._build_filters(filters)

        search_results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=with_payload
        )

        return [
            SearchResult(
                id=hit.id,
                score=hit.score,
                payload=hit.payload
            ) for hit in search_results.points
        ]

    def upsert_point(
        self, 
        collection_name: str, 
        point_id: Union[int, str], 
        vector: List[float], 
        payload: dict
    ):
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def delete_point(self, collection_name: str, point_id: Union[int, str]):
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=[point_id])
        )
    