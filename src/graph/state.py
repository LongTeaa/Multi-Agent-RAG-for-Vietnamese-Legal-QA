"""
src/graph/state.py
Lightweight shared state for the LangGraph workflow.

Phase 4 keeps large payloads (document text, web snippets, debug history) out
of GraphState. Agents pass compact ids through the graph and fetch full context
from src.graph.runtime_store when needed.
"""
from typing import Literal, Optional, TypedDict
from uuid import uuid4


class Citation(TypedDict, total=False):
    """Citation source in the generated answer."""
    text: str
    source: str
    position: int
    url: str
    source_id: str
    citation_id: str
    metadata: dict


class Document(TypedDict):
    """Retrieved legal document payload kept for backward compatibility only."""
    content: str
    metadata: dict
    score: float


class GraphState(TypedDict, total=False):
    """Compact state shared by all LangGraph agents."""

    # Input and tracing
    request_id: str
    trace_id: str
    question: str
    user_id: Optional[str]

    # Router
    intent: Optional[Literal["legal_query", "procedural", "out_of_scope", "general_chat"]]
    intent_confidence: Optional[float]

    # Retrieval coordination
    query_filters: Optional[dict]
    query_preferences: Optional[dict]
    retrieved_chunk_ids: Optional[list[str]]
    retrieved_scores: Optional[dict[str, float]]
    selected_context_ids: Optional[list[str]]

    # Grader
    grader_verdict: Optional[Literal["yes", "no"]]
    grader_score: Optional[float]

    # Web search coordination
    web_result_ids: Optional[list[str]]

    # Generator
    answer: Optional[str]
    citation_ids: Optional[list[str]]
    confidence: Optional[float]

    # Hallucination grader
    hallucination_verdict: Optional[Literal["pass", "fail"]]
    hallucinations: Optional[str]

    # Control
    generation_attempt: int
    hallucination_retry_count: int
    error: Optional[str]

    # Legacy payload fields. New workflow code should avoid returning these in
    # node updates; API mappers may still read them for old saved states/tests.
    documents: Optional[list[Document]]
    web_results: Optional[list[dict]]
    citations: Optional[list[Citation]]


def create_initial_state(question: str, user_id: str = "") -> GraphState:
    """Create a compact initial state for one request."""
    request_id = f"req_{uuid4().hex}"
    return GraphState(
        request_id=request_id,
        trace_id=request_id,
        question=question,
        user_id=user_id or None,
        intent=None,
        intent_confidence=None,
        query_filters=None,
        query_preferences=None,
        retrieved_chunk_ids=None,
        retrieved_scores=None,
        selected_context_ids=None,
        grader_verdict=None,
        grader_score=None,
        web_result_ids=None,
        answer=None,
        citation_ids=None,
        confidence=None,
        hallucination_verdict=None,
        hallucinations=None,
        generation_attempt=0,
        hallucination_retry_count=0,
        error=None,
    )

