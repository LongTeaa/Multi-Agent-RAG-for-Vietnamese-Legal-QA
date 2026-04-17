"""
src/graph/edges.py
Định nghĩa logic điều kiện (conditional edges) để định tuyến giữa các node trong LangGraph.
"""
from typing import Literal
from langgraph.graph import END
from src.graph.state import GraphState
from src.utils.logger import logger
from src.config import MAX_RETRIES


def decide_to_retrieve(state: GraphState) -> Literal["retriever", END]:
    """
    Dựa trên intent từ router để quyết định có đi tiếp đến retriever hay dừng lại.
    
    Args:
        state: Trạng thái hiện tại của graph
        
    Returns:
        Tên node tiếp theo hoặc END
    """
    intent = state.get("intent")
    
    if intent in ["legal_query", "procedural"]:
        logger.info(f"[EDGE] Intent matches RAG flow: '{intent}'. Routing to retriever.")
        return "retriever"
    
    logger.info(f"[EDGE] Intent '{intent}' does not require RAG. Routing to END.")
    # router_node đã trả về intent, generator_node sẽ không chạy
    # Ta có thể thêm một node "simple_chat" nếu muốn, nhưng ở đây theo design là exit.
    return END


def grade_documents(state: GraphState) -> Literal["web_searcher", "generator"]:
    """
    Dựa trên kết quả đánh giá của grader để chọn generator hay web_searcher (fallback).
    
    Args:
        state: Trạng thái hiện tại của graph
        
    Returns:
        "generator" nếu tài liệu đủ, ngược lại "web_searcher"
    """
    verdict = state.get("grader_verdict")
    
    if verdict == "yes":
        logger.info("[EDGE] Documents are relevant. Routing to generator.")
        return "generator"
    
    logger.info("[EDGE] Documents are NOT relevant. Routing to web_searcher fallback.")
    return "web_searcher"


def check_hallucination(state: GraphState) -> Literal["generator", END]:
    """
    Kiểm tra kết quả của hallucination grader hoặc lỗi từ generator.
    Nếu fail hoặc gặp lỗi parse và chưa quá số lần retry -> generate lại.
    Nếu pass hoặc đã quá số lần retry -> kết thúc.
    """
    verdict = state.get("hallucination_verdict")
    attempt = state.get("generation_attempt", 0)
    error = state.get("error", "")
    
    # 1. Nếu có lỗi ParseError từ generator -> Ưu tiên retry
    if error and "ParseError" in str(error):
        if attempt < MAX_RETRIES:
            logger.warning(f"[EDGE] Generator encountered ParseError (attempt {attempt}/{MAX_RETRIES}). Retrying generation...")
            return "generator"
        else:
            logger.error(f"[EDGE] Max retries reached for ParseError ({MAX_RETRIES}). Routing to END.")
            return END

    # 2. Nếu grader cho qua -> END
    if verdict == "pass":
        logger.info("[EDGE] No hallucinations detected. Routing to END.")
        return END
    
    # 3. Nếu grader báo lỗi ảo giác -> Kiểm tra số lần retry
    if attempt < MAX_RETRIES:
        logger.warning(f"[EDGE] Hallucinations detected (attempt {attempt}/{MAX_RETRIES}). Retrying generation...")
        return "generator"
    
    logger.error(f"[EDGE] Hallucinations detected but max retries reached ({MAX_RETRIES}). Routing to END.")
    return END

