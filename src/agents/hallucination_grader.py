"""
Step 13: Hallucination Grader Agent - Self-Reflection Check

Chức năng: Kiểm tra lại câu trả lời để phát hiện ảo giác (hallucination) -
thông tin sai, không có trong tài liệu, hoặc trích dẫn không chính xác.
Loop back đến generator nếu phát hiện lỗi (max 3 lần).
"""

import re
from typing import Dict, Any, List
from src.utils.llm_factory import get_model_with_fallback, parse_json_response
from src.utils.logger import logger
from src.graph.runtime_store import get_citations, get_documents, get_web_results
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
            metadata = doc.get("metadata", {})
            source = metadata.get("ten_van_ban") or metadata.get("doc_id") or doc.get("source", "Unknown")
            page_start = metadata.get("page_start", "")
            page_end = metadata.get("page_end", "")
            page_label = str(page_start) if page_start else ""
            if page_label and page_end and page_end != page_start:
                page_label += f"-{page_end}"
            citation_id = f"S{i}"
            context_parts.append(
                f"[{citation_id}] Nguồn: {source}\n"
                f"Điều/Khoản: {metadata.get('dieu', '')} {metadata.get('khoang', '')}\n"
                f"Trang: {page_label}\n"
                f"URL: {metadata.get('source_url', '')}\n"
                f"Nội dung: {content}\n"
            )
            
    # Format web results
    if web_results:
        context_parts.append("--- KẾT QUẢ TÌM KIẾM TỪ WEB ---")
        for i, res in enumerate(web_results, 1):
            content = res.get("content", "")
            title = res.get("title", "Unknown")
            url = res.get("url", "")
            context_parts.append(f"[Web {i}] Tiêu đề: {title}\nURL: {url}\nNội dung: {content}\n")
            
    return "\n".join(context_parts)


def _extract_source_ids(text: str) -> set[str]:
    return {
        match.group(1).replace(" ", "")
        for match in re.finditer(r"\[(S\d+|Web\s+\d+)\]", text or "", flags=re.IGNORECASE)
    }


def _valid_source_ids(documents: List[Dict], web_results: List[Dict] = None) -> set[str]:
    valid = {f"S{i}" for i, _ in enumerate(documents or [], 1)}
    valid.update({f"Web{i}" for i, _ in enumerate(web_results or [], 1)})
    return valid


def _numeric_facts(text: str) -> set[str]:
    normalized = (text or "").replace(",", ".")
    return {
        match.group(0).strip(".")
        for match in re.finditer(r"\b\d+(?:\.\d+)?%?\b", normalized)
    }


def _rule_based_hallucination_check(
    answer: str,
    documents: List[Dict],
    web_results: List[Dict] = None,
    citations: List[Dict] = None,
) -> str:
    """Return a failure reason for deterministic citation/fact issues."""
    valid_ids = _valid_source_ids(documents, web_results)
    cited_ids = _extract_source_ids(answer)

    for citation in citations or []:
        if isinstance(citation, dict):
            cited_ids.update(_extract_source_ids(str(citation.get("source", ""))))
            cited_ids.update(_extract_source_ids(str(citation.get("text", ""))))

    invalid_ids = cited_ids - valid_ids
    if invalid_ids:
        return f"Câu trả lời trích dẫn source id không tồn tại: {sorted(invalid_ids)}"

    if documents and not cited_ids:
        return "Câu trả lời không có citation source id [S...] từ tài liệu đã truy xuất."

    context_text = "\n".join(
        doc.get("content", "") + "\n" + " ".join(str(value) for value in doc.get("metadata", {}).values())
        for doc in documents or []
    )
    context_text += "\n" + "\n".join(res.get("content", "") for res in web_results or [])
    answer_numbers = _numeric_facts(answer)
    context_numbers = _numeric_facts(context_text)
    allowed_numbers = {re.sub(r"\D", "", source_id) for source_id in cited_ids}
    unsupported_numbers = answer_numbers - context_numbers - allowed_numbers
    if unsupported_numbers:
        return f"Câu trả lời có số liệu không xuất hiện trong context: {sorted(unsupported_numbers)}"

    missing_url_citations = [
        citation
        for citation in citations or []
        if isinstance(citation, dict)
        and _extract_source_ids(str(citation.get("source", "")) + str(citation.get("text", "")))
        and not citation.get("url")
    ]
    if missing_url_citations:
        return "Citation có source id nhưng thiếu URL nguồn."

    return ""


async def hallucination_grader_node(state: GraphState) -> Dict[str, Any]:
    """
    Node kiểm tra ảo giác trong câu trả lời.
    """
    question = state.get("question")
    answer = state.get("answer")
    documents = get_documents(state)
    web_results = get_web_results(state)
    citations = get_citations(state)
    attempt = state.get("hallucination_retry_count", 0) + 1
    
    logger.info(f"Checking hallucinations in answer (attempt {attempt})...")
    
    if not answer or answer == "Không có nội dung câu trả lời.":
        return {
            "hallucination_verdict": "fail",
            "hallucinations": "Câu trả lời rỗng, cần sinh lại.",
            "hallucination_retry_count": attempt
        }

    rule_failure = _rule_based_hallucination_check(answer, documents, web_results, citations)
    if rule_failure:
        logger.warning(f"Rule-based hallucination check failed: {rule_failure}")
        return {
            "hallucination_verdict": "fail",
            "hallucinations": rule_failure,
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
