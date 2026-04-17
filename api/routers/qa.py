"""
api/routers/qa.py
Router for legal QA endpoints, including streaming support.
"""
import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from src.graph.graph import app as graph_app
from src.graph.state import create_initial_state
from src.models.request import QuestionRequest
from src.models.response import AnswerResponse, StreamEvent, Citation, WebResult
from src.utils.logger import logger

router = APIRouter(prefix="/qa", tags=["QA"])

# Mapping internal node names to user-friendly status messages
NODE_MAPPING = {
    "router": "Đang phân tích ý định câu hỏi...",
    "retriever": "Đang tìm kiếm trong cơ sở dữ liệu luật...",
    "grader": "Đang kiểm tra độ phù hợp của tài liệu...",
    "web_searcher": "Dữ liệu nội bộ chưa đủ, đang mở rộng tìm kiếm Google...",
    "generator": "Đang tổng hợp câu trả lời và trích dẫn...",
    "hallucination_grader": "Đang kiểm tra tính chính xác và ảo giác..."
}


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Synchronous endpoint to ask a question and get the final answer.
    """
    start_time = time.time()
    try:
        initial_state = create_initial_state(request.question, request.user_id)
        
        # Invoke the graph synchronously
        final_state = await asyncio.to_thread(graph_app.invoke, initial_state)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return _map_state_to_response(final_state, processing_time)
        
    except Exception as e:
        logger.error(f"Error in ask_question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream")
async def stream_question(question: str = "", user_id: str = ""):
    """
    Streaming endpoint using SSE to provide real-time updates.
    """
    if not question or not question.strip():
        async def error_gen():
            yield {
                "event": "error", 
                "data": json.dumps({"message": "Vui lòng nhập câu hỏi pháp lý."})
            }
        return EventSourceResponse(error_gen())

    return EventSourceResponse(
        generate_graph_events(question, user_id),
        send_timeout=300,
        ping=20
    )


async def generate_graph_events(question: str, user_id: str):
    """Internal generator for graph events."""
    start_time = time.time()
    initial_state = create_initial_state(question, user_id)
    full_state = initial_state.copy()
    
    try:
        yield {"event": "status", "data": json.dumps({"node": "start", "message": "Khởi tạo hệ thống..."})}

        # astream yields events as the graph executes
        async for event in graph_app.astream(initial_state, stream_mode="updates"):
            if not isinstance(event, dict):
                logger.warning(f"Unexpected event type from graph: {type(event)}")
                continue

            for node_name, updates in event.items():
                # Update our local full_state tracker safely
                if isinstance(updates, dict):
                    full_state.update(updates)
                else:
                    logger.warning(f"Node '{node_name}' returned non-dict updates: {type(updates)}")
                
                # Send status update
                if node_name in NODE_MAPPING:
                    yield {
                        "event": "status",
                        "data": json.dumps({
                            "node": node_name,
                            "message": NODE_MAPPING.get(node_name, "Đang xử lý...")
                        })
                    }
                
                # Check for explicit error returned by a node
                if isinstance(updates, dict) and updates.get("error"):
                    logger.error(f"Node '{node_name}' reported error: {updates.get('error')}")

        # Execution finished, send final answer
        processing_time = int((time.time() - start_time) * 1000)
        response = _map_state_to_response(full_state, processing_time)
        
        yield {
            "event": "final_answer",
            "data": response.model_dump_json() if hasattr(response, "model_dump_json") else response.json()
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error("Error in graph streaming: {}\n{}", str(e), error_trace)

        yield {
            "event": "error",
            "data": json.dumps({
                "message": "Có lỗi xảy ra trong quá trình xử lý.", 
                "detail": str(e),
                "type": type(e).__name__,
                "trace": error_trace[:500]
            })
        }

    finally:
        yield {"event": "end", "data": "Finish"}


def _map_state_to_response(state: dict, processing_time: int) -> AnswerResponse:
    """Helper to convert GraphState dict to AnswerResponse model."""
    # 1. Đảm bảo answer và error luôn là string (không được để None)
    answer = state.get("answer") or "Xin lỗi, tôi không thể tìm thấy câu trả lời phù hợp."
    
    # Ép kiểu error về string rỗng nếu là None để tránh Pydantic crash
    error_val = state.get("error")
    if error_val is None:
        error_val = ""
    else:
        error_val = str(error_val)

    # 2. Xử lý confidence an toàn
    try:
        confidence = float(state.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    
    return AnswerResponse(
        question=state.get("question", ""),
        answer=answer,
        citations=[Citation(**c) for c in state.get("citations", [])] if state.get("citations") else [],
        web_results=[WebResult(**w) for w in state.get("web_results", [])] if state.get("web_results") else [],
        confidence=confidence,
        intent=state.get("intent"),
        intent_confidence=state.get("intent_confidence"),
        generation_attempt=state.get("generation_attempt", 1),
        processing_time_ms=processing_time,
        error=error_val
    )
