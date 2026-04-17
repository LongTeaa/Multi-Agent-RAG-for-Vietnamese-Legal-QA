"""
Step 10: Grader Agent - Corrective RAG (CRAG) Relevance Check

Chức năng: Đánh giá mức độ liên quan và tính đầy đủ của tài liệu
truy xuất được so với câu hỏi. Trả về verdict: "yes" (đủ) hoặc "no" (cần web search).
"""

from typing import Dict, Any, List
from src.utils.llm_factory import get_model_with_fallback, parse_json_response
from src.utils.logger import logger
from src.graph.state import GraphState


GRADER_PROMPT = """
Bạn là một chuyên gia pháp lý đánh giá tính liên quan của tài liệu.

Câu hỏi: {question}

Tài liệu truy xuất:
{documents_text}

Hãy đánh giá:
1. Tài liệu này có chứa thông tin cần thiết để trả lời câu hỏi không?
2. Thông tin có đầy đủ để cung cấp câu trả lời chính xác?
3. Có cần tìm kiếm thêm từ các nguồn khác không?

Trả về JSON hợp lệ theo đúng format sau (không thêm comment):
{{
  "relevance_score": <số_thực_0_đến_1>,
  "verdict": "<yes_hoặc_no>",
  "reasoning": "<lý_giải_ngắn_gọn_về_tính_liên_quan>"
}}

Luu ý: verdict là \"no\" khi:
- Tài liệu không liên quan đến câu hỏi
- Văn bản có trạng_thái \"Hết hiệu lực\" hoặc ban hành trước năm 2010 và đã có văn bản thay thế
- Thiếu chi tiết càn thiết để trả lời đầy đủ
"""


def _format_documents(documents: List[Dict]) -> str:
    """
    Format danh sách documents thành text để đưa vào prompt grader.
    
    Args:
        documents: List[Dict] từ retriever output
        
    Returns:
        Formatted text string
    """
    if not documents:
        return "Không có tài liệu nào được tìm thấy."
    
    text_parts = []
    for i, doc in enumerate(documents, 1):
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        score = doc.get("score", 0.0)
        
        # Format: [Tài liệu 1] Tên Luật - Điều X - Điểm liên quan: Y%
        law_name = metadata.get("ten_van_ban", "Unknown")
        article = metadata.get("dieu", "")
        source_info = f"{law_name}"
        if article:
            source_info += f" - {article}"
        
        text_parts.append(f"[Tài liệu {i}] {source_info} (Điểm liên quan: {score:.2%})")
        text_parts.append(content[:500])  # Lấy 500 ký tự đầu tiên
        text_parts.append("-" * 50)
    
    return "\n".join(text_parts)


def grader_node(state: GraphState) -> Dict[str, Any]:
    """
    Grader node: Đánh giá tính liên quan của documents truy xuất được.
    
    Input từ state:
    - question: str (câu hỏi)
    - documents: List[Dict] (từ retriever)
    
    Output cập nhật state:
    - grader_verdict: str ("yes" | "no")
    - grader_score: float (0.0 - 1.0)
    """
    try:
        question = state.get("question", "")
        documents = state.get("documents", [])
        
        if not documents:
            logger.warning("No documents to grade")
            return {
                "grader_verdict": "no",
                "grader_score": 0.0,
                "error": None
            }
        
        logger.info(f"Grading {len(documents)} documents for relevance...")
        
        # 1. Format documents
        documents_text = _format_documents(documents)
        
        # 2. Prepare prompt
        prompt = GRADER_PROMPT.format(
            question=question,
            documents_text=documents_text
        )
        
        # 3. Call LLM (uses distributed project keys based on purpose)
        llm = get_model_with_fallback(purpose="grader")
        response = llm.invoke(prompt)
        
        # 4. Parse JSON response
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", part.get("content", str(part))))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)

        try:
            result = parse_json_response(content)
        except Exception as parse_err:
            logger.error(f"Failed to parse LLM response in grader: {parse_err}")
            # Trả về mặc định nếu parse fail
            return {
                "grader_verdict": "no",
                "grader_score": 0.0,
                "error": str(parse_err)
            }
        
        # 5. Extract verdict and score
        verdict = result.get("verdict", "no")
        raw_score = result.get("relevance_score", 0.0)
        # LLM có thể trả về string như "0.95" thay vì số → safe cast
        try:
            score = float(raw_score)
        except (ValueError, TypeError):
            score = 0.0
        reasoning = result.get("reasoning", "")
        
        # Validate verdict
        if verdict not in ["yes", "no"]:
            logger.warning(f"Invalid verdict '{verdict}', defaulting to 'no'")
            verdict = "no"
        
        logger.info(f"Grader verdict: '{verdict}' with score {score:.2%}")
        logger.debug(f"Reasoning: {reasoning}")
        
        return {
            "grader_verdict": verdict,
            "grader_score": score,
            "error": None
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error("Error in grader_node: {}", error_msg[:500], exc_info=True)

        
        # Check quota/rate limit
        if "quota" in error_msg.lower() and "429" in error_msg:
             logger.error("!!! QUOTA EXHAUSTED !!! Bạn đã hết hạn mức sử dụng trong ngày (RPD).")
        
        return {
            "grader_verdict": "no",
            "grader_score": 0.0,
            "error": error_msg
        }

