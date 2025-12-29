# Orchestrator

from typing import List
from fastapi import Depends
from app.features.search.repository import SearchRepository
from app.features.search.embedding import EmbeddingService
from app.features.search.schemas import SearchResponse


class SearchService:
    def __init__(
        self,
        repository: SearchRepository = Depends(),
        embedding: EmbeddingService = Depends()
    ):
        self.repository = repository
        self.embedding = embedding

    async def search(self, query: str, top_k: int) -> SearchResponse:
        try:
            vector = self.embedding.generate_vector(query)
            results = self.repository.search_learning_paths(vector, top_k)
            return SearchResponse(query=query, total=len(results), results=results)
        except Exception as e:
            # เพิ่มการจัดการ Error พื้นฐาน
            return SearchResponse(query=query, total=0, results=[])

    async def sync_upsert(self, path_id: int, title: str, description: str):
        """รับข้อมูลจาก Go มาทำ Embedding และบันทึก/แก้ไข"""
        # รวมข้อความเพื่อความแม่นยำในการค้นหา (Context)
        combined_text = f"{title}: {description}"
        vector = self.embedding.generate_vector(combined_text)

        payload = {"title": title, "description": description}
        self.repository.upsert_path(path_id, vector, payload)

    async def sync_delete(self, path_id: int):
        """ลบข้อมูลออกจาก Vector DB ตามคำสั่งจาก Go"""
        self.repository.delete_path(path_id)
