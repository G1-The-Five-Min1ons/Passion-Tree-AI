import httpx
from typing import Any, Dict, Optional
from fastapi import HTTPException

class BaseClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def _send_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=self.headers
                )
                
                # ตรวจสอบ Error เบื้องต้น (HTTP 4xx, 5xx)
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                # จัดการ Error จาก Server (เช่น API Key ผิด, โควตาเต็ม)
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"API Error: {e.response.text}"
                )
            except httpx.RequestError as e:
                # จัดการ Error ทางเทคนิค (เช่น เน็ตหลุด, Timeout)
                raise HTTPException(
                    status_code=503,
                    detail=f"Network error: {str(e)}"
                )