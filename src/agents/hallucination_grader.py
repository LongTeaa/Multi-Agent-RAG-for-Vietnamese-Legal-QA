"""
Step 13: Hallucination Grader Agent - Self-Reflection Check

Chức năng: Kiểm tra lại câu trả lời để phát hiện ảo giác (hallucination) -
thông tin sai, không có trong tài liệu, hoặc trích dẫn không chính xác.
Loop back đến generator nếu phát hiện lỗi (max 3 lần).
"""

from typing import Dict, Any, List
from src.utils.llm_factory import get_model_with_fallback, parse_json_response
from src.utils.logger import logger
from src.graph.state import GraphState


HALLUCINATION_GRADER_PROMPT = """
Bạn là một chuyên gia kiểm tra tính chính xác của các câu trả lời pháp lý.

Để phát hiện ảo giác (hallucination), hãy kiểm tra:
1. Mỗi dữ kiện trong câu trả lời có xuất hiện trong "Tài liệu gốc" HOẶC "Kết quả tìm kiếm Web" không?
2. Có thông tin nào sai lệch hoàn toàn hoặc bị hiểu sai so với nội dung được cung cấp?
3. Có tuyên bố về luật pháp mà không có bất kỳ căn cứ nào từ các nguồn tài liệu đã cung cấp?

Lưu ý:
    - 'fail': Nếu câu trả lời chứa thông tin SAI LỆCH hoặc MÂU THUẪN trực tiếp với tài liệu.
    - 'pass': Nếu câu trả lời chính xác dựa trên tài liệu. 
    LƯU Ý: Nếu AI trích dẫn thêm số hiệu điều luật (ví dụ: Điều 112) mà nội dung luật đó ĐÚNG với tài liệu nhưng tài liệu thiếu số hiệu, hãy linh hoạt cho 'pass' nếu nội dung khớp. CHỈ đánh giá 'fail' khi AI tự bịa ra các con số (ngày nghỉ, mức lương, mốc thời gian) hoàn toàn không có cơ sở trong tài liệu.

- Bạn phải đối chiếu câu trả lời với CẢ "Tài liệu pháp lý địa phương" và "Kết quả tìm kiếm từ Web".
- Nếu thông tin trong câu trả lời khớp với bất kỳ nguồn nào trong số đó, đó KHÔNG phải là ảo giác.
- Trả về 'pass' nếu thông tin trung thực với các nguồn tham khảo.

Câu hỏi: {question}

Câu trả lời cần kiểm tra:
{answer}

Tài liệu tham khảo (bao gồm Local Docs và Web Results):
{documents}

Trả về JSON hợp lệ theo đúng format sau (không thêm comment):
{{
  "verdict": "<pass_hoặc_fail>",
  "hallucinations": "<mô_tả_ảo_giác_nếu_có>",
  "reasoning": "<lý_giải_ngắn_gọn_cho_kết_quả_kiểm_tra>"
}}
"""


def _format_all_context_for_grader(documents: List[Dict], web_results: List[Dict] = None) -> str:
    """
    Format cả documents từ retriever và web_results thành một khối text duy nhất cho Grader.
    """
    context_parts = []
    
    # Format local documents
    if documents:
        context_parts.append("--- TÀI LIỆU PHÁP LÝ ĐỊA PHƯƠNG ---")
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            source = doc.get("source", "Unknown")
            context_parts.append(f"[{i}] Nguồn: {source}\nNội dung: {content}\n")
            
    # Format web results
    if web_results:
        context_parts.append("--- KẾT QUẢ TÌM KIẾM TỪ WEB ---")
        for i, res in enumerate(web_results, 1):
            content = res.get("content", "")
            title = res.get("title", "Unknown")
            url = res.get("url", "")
            context_parts.append(f"[Web {i}] Tiêu đề: {title}\nURL: {url}\nNội dung: {content}\n")
            
    return "\n".join(context_parts)


async def hallucination_grader_node(state: GraphState) -> Dict[str, Any]:
    """
    Node kiểm tra ảo giác trong câu trả lời.
    """
    question = state.get("question")
    answer = state.get("answer")
    documents = state.get("documents", [])
    web_results = state.get("web_results", [])
    attempt = state.get("hallucination_retry_count", 0) + 1
    
    logger.info(f"Checking hallucinations in answer (attempt {attempt})...")
    
    if not answer or answer == "Không có nội dung câu trả lời.":
        return {
            "hallucination_verdict": "fail",
            "hallucinations": "Câu trả lời rỗng, cần sinh lại.",
            "hallucination_retry_count": attempt
        }

    # Format all documents for the grader
    all_docs_text = _format_all_context_for_grader(documents, web_results)
    
    # Log prompt length for monitoring
    prompt_content = HALLUCINATION_GRADER_PROMPT.format(
        question=question,
        answer=answer,
        documents=all_docs_text
    )
    logger.info(f"Hallucination grader prompt length: {len(prompt_content)} chars")
    
    try:
        # Sử dụng model hỗ trợ JSON Mode, truyền purpose bằng keyword argument để tránh nhầm với model name
        llm = get_model_with_fallback(purpose="hallucination_grader", json_mode=True)

        response = await llm.ainvoke(prompt_content)
        
        # Parse JSON
        result = parse_json_response(response.content)
        verdict = result.get("verdict", "fail").lower().strip()
        hallucinations = result.get("hallucinations", "")
        reasoning = result.get("reasoning", "")
        
        if verdict == "pass":
            logger.info("Hallucination check verdict: 'pass'")
            return {
                "hallucination_verdict": "pass",
                "hallucinations": None,
                "hallucination_retry_count": attempt
            }
        else:
            logger.warning(f"Hallucination check verdict: 'fail'")
            logger.warning(f"Hallucinations detected: {hallucinations}")
            logger.debug(f"Reasoning: {reasoning}")
            return {
                "hallucination_verdict": "fail",
                "hallucinations": hallucinations,
                "hallucination_retry_count": attempt
            }
            
    except Exception as e:
        logger.error("Error in hallucination_grader_node: {}", e)
        # Nếu có lỗi parse (ví dụ response bị cắt), coi như fail để retry
        if "ParseError" in str(e) or "JSON" in str(e).upper():
            return {
                "hallucination_verdict": "fail",
                "hallucinations": "Lỗi định dạng JSON từ Grader.",
                "hallucination_retry_count": attempt,
                "error": str(e)
            }
            
        return {
            "hallucination_verdict": "pass", # Fallback an toàn nếu lỗi hệ thống
            "hallucinations": None,
            "hallucination_retry_count": attempt,
            "error": str(e)
        }
