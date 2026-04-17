"""
src/graph/graph.py
Xây dựng và compile LangGraph cho hệ thống Multi-Agent RAG.
"""
from langgraph.graph import StateGraph, END
from src.graph.state import GraphState
from src.agents.router import router_node
from src.agents.retriever import retriever_node
from src.agents.grader import grader_node
from src.agents.web_searcher import web_searcher_node
from src.agents.generator import generator_node
from src.agents.hallucination_grader import hallucination_grader_node
from src.graph.edges import decide_to_retrieve, grade_documents, check_hallucination


def build_graph():
    """
    Khởi tạo và cấu hình workflow LangGraph.
    
    Returns:
        Compiled LangGraph application
    """
    # 1. Khởi tạo StateGraph với GraphState definition
    workflow = StateGraph(GraphState)
    
    # 2. Thêm các Agents (Nodes)
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("web_searcher", web_searcher_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("hallucination_grader", hallucination_grader_node)
    
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
