"""
src/graph/graph.py
Xây dựng và compile LangGraph cho hệ thống Multi-Agent RAG.
"""
import inspect
import time
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from src.graph.state import GraphState
from src.agents.router import router_node
from src.agents.retriever import retriever_node
from src.agents.grader import grader_node
from src.agents.web_searcher import web_searcher_node
from src.agents.generator import generator_node
from src.agents.hallucination_grader import hallucination_grader_node
from src.graph.edges import decide_to_retrieve, grade_documents, check_hallucination
from src.graph.runtime_store import record_agent_event


def _summarize_input(state: GraphState) -> str:
    question = (state.get("question") or "")[:120]
    filters = state.get("query_filters") or {}
    return f"question={question!r}; filters={filters}"


def _summarize_output(updates: dict[str, Any]) -> str:
    compact_keys = [
        "intent",
        "intent_confidence",
        "grader_verdict",
        "grader_score",
        "retrieved_chunk_ids",
        "selected_context_ids",
        "web_result_ids",
        "citation_ids",
        "confidence",
        "generation_attempt",
        "hallucination_verdict",
        "hallucination_retry_count",
    ]
    summary = {key: updates.get(key) for key in compact_keys if key in updates}
    if "answer" in updates:
        summary["answer_chars"] = len(updates.get("answer") or "")
    return str(summary)


def _compact_chunk_ids(updates: dict[str, Any]) -> list[str]:
    return list(updates.get("selected_context_ids") or updates.get("retrieved_chunk_ids") or [])


def _audit_node(name: str, node: Callable) -> Callable:
    async def async_wrapper(state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            updates = await node(state)
            return _record(name, state, updates, started)
        except Exception as exc:
            _record(name, state, {"error": str(exc)}, started)
            raise

    def sync_wrapper(state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            updates = node(state)
            return _record(name, state, updates, started)
        except Exception as exc:
            _record(name, state, {"error": str(exc)}, started)
            raise

    return async_wrapper if inspect.iscoroutinefunction(node) else sync_wrapper


def _record(name: str, state: GraphState, updates: Any, started: float) -> Any:
    if isinstance(updates, dict):
        trace_id = state.get("trace_id") or state.get("request_id") or ""
        latency_ms = int((time.perf_counter() - started) * 1000)
        record_agent_event(
            trace_id=trace_id,
            agent=name,
            input_summary=_summarize_input(state),
            output_summary=_summarize_output(updates),
            chunk_ids=_compact_chunk_ids(updates),
            scores=updates.get("retrieved_scores") or {},
            latency_ms=latency_ms,
            error=updates.get("error"),
        )
    return updates


def build_graph():
    """
    Khởi tạo và cấu hình workflow LangGraph.
    
    Returns:
        Compiled LangGraph application
    """
    # 1. Khởi tạo StateGraph với GraphState definition
    workflow = StateGraph(GraphState)
    
    # 2. Thêm các Agents (Nodes)
    workflow.add_node("router", _audit_node("router", router_node))
    workflow.add_node("retriever", _audit_node("retriever", retriever_node))
    workflow.add_node("grader", _audit_node("grader", grader_node))
    workflow.add_node("web_searcher", _audit_node("web_searcher", web_searcher_node))
    workflow.add_node("generator", _audit_node("generator", generator_node))
    workflow.add_node("hallucination_grader", _audit_node("hallucination_grader", hallucination_grader_node))
    
    # 3. Kết nối các Nodes bằng Edges
    
    # Bắt đầu tại Router
    workflow.set_entry_point("router")
    
    # Phân luồng từ Router: Đi tìm kiếm hoặc Kết thúc (out_of_scope/chat)
    workflow.add_conditional_edges(
        "router",
        decide_to_retrieve,
        {
            "retriever": "retriever",
            END: END
        }
    )
    
    # Sau khi retrieve -> đi chấm điểm tài liệu (Grader)
    workflow.add_edge("retriever", "grader")
    
    # Phân luồng từ Grader: Đi generator ngay hoặc đi Web Search nếu thiếu tin
    workflow.add_conditional_edges(
        "grader",
        grade_documents,
        {
            "generator": "generator",
            "web_searcher": "web_searcher"
        }
    )
    
    # Web search xong -> đi Generator
    workflow.add_edge("web_searcher", "generator")
    
    # Sinh câu trả lời xong -> Kiểm tra ảo giác
    workflow.add_edge("generator", "hallucination_grader")
    
    # Phân luồng từ Hallucination Grader: Retry generator hoặc Kết thúc
    workflow.add_conditional_edges(
        "hallucination_grader",
        check_hallucination,
        {
            "generator": "generator",
            END: END
        }
    )
    
    # 4. Compile graph
    # Checkpoint (Memory) có thể thêm ở PHASE sau nếu cần lưu session
    app = workflow.compile()
    
    return app


# Instance graph toàn cục để import ở các module khác (VD: API)
app = build_graph()
