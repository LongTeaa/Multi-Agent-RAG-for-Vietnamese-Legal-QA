"""
Agents module - 6 LLM-powered agents cho Multi-Agent RAG hệ thống pháp lý tiếng Việt.

Các agents:
1. Router: Phân loại intent câu hỏi
2. Retriever: Tìm kiếm hybrid (BM25 + Vector)
3. Grader: CRAG relevance check
4. Web Searcher: Tìm kiếm web fallback
5. Generator: Sinh câu trả lời với trích dẫn
6. Hallucination Grader: Kiểm tra ảo giác
"""

from .retriever import retriever_node
from .router import router_node
from .grader import grader_node
from .web_searcher import web_searcher_node
from .generator import generator_node
from .hallucination_grader import hallucination_grader_node

__all__ = [
    "retriever_node",
    "router_node",
    "grader_node",
    "web_searcher_node",
    "generator_node",
    "hallucination_grader_node",
]
