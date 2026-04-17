"""
src/models/response.py
Pydantic models for API responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any


class Citation(BaseModel):
    """Citation model for referencing legal documents."""
    text: str = Field(..., description="The segment of text being cited.")
    source: str = Field(..., description="The name or reference of the document source.")
    position: int = Field(..., description="The position in the answer where the citation occurs.")


class WebResult(BaseModel):
    """Web search result model."""
    url: str
    title: str
    content: str
    source_type: str = "web"


class AnswerResponse(BaseModel):
    """Response model for a legal question."""
    question: str
    answer: str
    citations: List[Citation] = []
    web_results: List[WebResult] = []
    confidence: float = 0.0
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    generation_attempt: int = 1
    processing_time_ms: int = 0
    error: Optional[str] = None


class StreamEvent(BaseModel):
    """Container for SSE events sent during graph execution."""
    event: str  # 'status', 'data', 'error', 'end'
    node: Optional[str] = None
    message: Optional[str] = None
    data: Optional[Any] = None
