"""
src/models/request.py
Pydantic models for API requests.
"""
from pydantic import BaseModel, Field
from typing import Optional


class QuestionRequest(BaseModel):
    """Request model for asking a legal question."""
    question: str = Field(..., description="The legal question to ask.")
    user_id: Optional[str] = Field(None, description="The unique identifier of the user.")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Người lao động có được nghỉ tết âm lịch bao nhiêu ngày?",
                "user_id": "user_123"
            }
        }
