"""
src/graph/state.py
Định nghĩa GraphState – trạng thái chia sẻ giữa tất cả các agent trong LangGraph.
"""
from typing import TypedDict, List, Optional, Literal


class Citation(TypedDict):
    """Trích dẫn nguồn trong câu trả lời."""
    text: str       # Đoạn trích dẫn (ví dụ: "Theo Điều 105, Khoản 1, BLLĐ 2019")
    source: str     # Tên tài liệu (ví dụ: "Bộ luật Lao động 2019 - Điều 105, Khoản 1")
    position: int   # Vị trí xuất hiện trong answer (0-indexed)


class Document(TypedDict):
    """Tài liệu pháp lý được truy xuất từ Qdrant."""
    content: str
    metadata: dict  # so_hieu_van_ban, ten_van_ban, dieu, khoang, ...
    score: float    # Relevance score từ Hybrid Search


class GraphState(TypedDict, total=False):
    """
    Trạng thái chia sẻ toàn bộ LangGraph.

    Mỗi agent đọc state → xử lý → trả về state đã cập nhật.
    """
    # ── INPUT ────────────────────────────────────────────────────
    question: str
    user_id: Optional[str]

    # ── ROUTER AGENT ─────────────────────────────────────────────
    intent: Optional[Literal["legal_query", "procedural", "out_of_scope", "general_chat"]]
    intent_confidence: Optional[float]

    # ── RETRIEVER ────────────────────────────────────────────────
    documents: Optional[List[Document]]

    # ── GRADER AGENT (CRAG) ──────────────────────────────────────
    grader_verdict: Optional[Literal["yes", "no"]]
    grader_score: Optional[float]

    # ── WEB SEARCHER AGENT ───────────────────────────────────────
    web_results: Optional[List[dict]]   # [{content: str, url: str}, ...]

    # ── GENERATOR AGENT ──────────────────────────────────────────
    answer: Optional[str]
    citations: Optional[List[Citation]]
    confidence: Optional[float]

    # ── HALLUCINATION GRADER AGENT ───────────────────────────────
    hallucination_verdict: Optional[Literal["pass", "fail"]]
    hallucinations: Optional[str]       # Mô tả ảo giác phát hiện được

    # ── CONTROL ──────────────────────────────────────────────────
    generation_attempt: int             # Đếm số lần generate (max = MAX_RETRIES)
    error: Optional[str]                # Thông báo lỗi nếu có


def create_initial_state(question: str, user_id: str = "") -> GraphState:
    """
    Tạo state khởi tạo cho một request mới.

    Args:
        question: Câu hỏi pháp lý của người dùng
        user_id:  Identifier của user (tùy chọn)

    Returns:
        GraphState với tất cả optional fields = None
    """
    return GraphState(
        question=question,
        user_id=user_id or None,
        intent=None,
        intent_confidence=None,
        documents=None,
        grader_verdict=None,
        grader_score=None,
        web_results=None,
        answer=None,
        citations=None,
        confidence=None,
        hallucination_verdict=None,
        hallucinations=None,
        generation_attempt=0,
        error=None,
    )
