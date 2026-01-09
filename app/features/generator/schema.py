from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    topic: str = Field(..., example="อยากเรียนเขียนโปรแกรมภาษา Go เริ่มยังไงดี", description="User's learning goal")

class GenerateResponse(BaseModel):
    topic: str
    result: str
    used_context: list[str] = Field(default=[], description="Examples used for generation (Debugging)")