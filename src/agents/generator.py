"""
Step 12: Generator Agent - Answer Generation with Citations

Chức năng: Tổng hợp và sinh câu trả lời dựa trên tài liệu đã được kiểm chứng,
với trích dẫn nguồn rõ ràng. Bắt buộc có citation format: "Theo Điều X, Khoản Y, [Tên Luật] năm N..."
"""

from typing import Dict, Any, List
import json
import re
from src.utils.llm_factory import get_model_with_fallback, parse_json_response
from src.utils.logger import logger
from src.graph.state import GraphState


GENERATOR_PROMPT = """
Bạn là một luật sư pháp lý Việt Nam có kinh nghiệm. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên tài liệu pháp luật.

QUAN TRỌNG:
1. Trình bày câu trả lời bằng định dạng Markdown để dễ đọc:
   - Sử dụng **danh sách có dấu chấm (bullet points)** hoặc **danh sách đánh số** cho các ý chính.
   - Sử dụng **in đậm (bold)** cho tên luật, số hiệu văn bản hoặc các cụm từ quan trọng.
   - Sử dụng **xuống dòng (line breaks)** hợp lý giữa các đoạn để không bị quá dày đặc.
2. Trả lời dựa trên **TẤT CẢ** tài liệu tham khảo được cung cấp (bao gồm cả Tài liệu pháp lý và Kết quả tìm kiếm web).
3. Nếu "Tài liệu pháp lý" không có thông tin, hãy sử dụng thông tin từ "Kết quả tìm kiếm web" để trả lời, nhưng cần lưu ý rõ: *"Dựa trên thông tin tìm kiếm được trên internet..."*
4. BẮT BUỘC trích dẫn rõ ràng: "Theo Điều X, Khoản Y, [Tên Luật] năm N: ..." hoặc *"Theo thông tin từ nguồn [Tên trang web]: ..."*
5. BẮT BUỘC: Trình bày CHI TIẾT các nội dung như mức lương, điều kiện, mốc thời gian... ngay trong trường "answer". KHÔNG ĐƯỢC chỉ trả lời chung chung rồi để người dùng tự xem citations.
6. BẮT BUỘC: Bạn CHỈ ĐƯỢC PHÉP trả về DUY NHẤT một khối JSON hợp lệ. KHÔNG ĐƯỢC thêm bất kỳ văn bản giải thích, lời chào hay bình luận nào bên ngoài khối JSON.

    Văn phong: Trang trọng, chuyên nghiệp, súc tích.
    YÊU CẦU QUAN TRỌNG: 
    - CHỈ SỬ DỤNG thông tin có trong tài liệu tham khảo. KHÔNG đưa thêm kiến thức bên ngoài (ví dụ: số hiệu điều luật, ngày tháng cụ thể) nếu chúng không xuất hiện rõ ràng trong tài liệu, để tránh bị đánh dấu là ảo giác (hallucination).
    - Tổng hợp thông tin từ nhiều nguồn để tránh lặp lại cùng một ý. 
    - Trả lời trực tiếp, đi thẳng vào vấn đề, tránh diễn giải quá dài dòng.
    - Nếu thông tin giữa Tài liệu pháp lý và Web giống nhau, hãy gộp lại và trích dẫn cả hai.
    Confidence: Số thực từ 0.0 đến 1.0.

    Câu hỏi: {question}

    Tài liệu tham khảo:
    {context}

    {feedback}

    JSON Output Format:
    {{
      "answer": "<câu_trả_lời_định_dạng_Markdown_kèm_trích_dẫn>",
      "citations": [
        {{
          "text": "<đoạn_văn_trích_dẫn>",
          "source": "<tên_văn_bản_hoặc_website>",
          "position": <số_nguyên>,
          "url": "<link_nếu_có>"
        }}
      ],
      "confidence": <số_thực_0_đến_1>
    }}
    """


def _format_documents_and_web(
    documents: List[Dict],
    web_results: List[Dict] = None
) -> str:
    """
    Format documents từ retriever + web_results thành context text.
    
    Args:
        documents: List[Dict] từ retriever
        web_results: List[Dict] từ web_searcher (optional)
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    # 1. Add local legal documents
    if documents:
        context_parts.append("=== TÀI LIỆU PHÁP LY ===")
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            law_name = metadata.get("ten_van_ban", "")
            dieu = metadata.get("dieu", "")
            khoang = metadata.get("khoang", "")
            
            source_info = law_name
            if dieu:
                source_info += f" - {dieu}"
            if khoang:
                source_info += f" {khoang}"
            
            context_parts.append(f"\n[Tài liệu {i}] {source_info}")
            context_parts.append(content)
            context_parts.append("-" * 50)
    
    # 2. Add web results if available
    if web_results:
        context_parts.append("\n=== KẾT QUẢ TÌM KIẾM WEB ===")
        for i, result in enumerate(web_results, 1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            if len(content) > 4000:
                content = content[:4000] + "... (truncated)"
            
            context_parts.append(f"\n[Web {i}] {title}")
            context_parts.append(f"URL: {url}")
            context_parts.append(content)
            context_parts.append("-" * 50)

    
    return "\n".join(context_parts)


def _extract_citations(answer: str, documents: List[Dict]) -> List[Dict]:
    """
    Extract citations từ answer text.
    Tìm các pattern như "Theo Điều X, Khoản Y, [Tên Luật] năm N"
    
    Args:
        answer: Generated answer text
        documents: Original documents list
        
    Returns:
        List[Dict] with keys: text, source, position
    """
    if not answer or not isinstance(answer, str):
        return []

    citations = []
    
    # Pattern để tìm citation: "Theo Điều X, Khoản Y, [Luật/Nghị định/...] năm N"
    # Hoặc: "Điều X, Luật Y năm Z"
    citation_patterns = [
        r'Theo\s+(?:Điều\s+\d+(?:,\s*Khoản\s+\d+)?)[^.!?]*(?:Luật|Nghị định|Thông tư|Nghị quyết)\s+[^.!?]*?\s+năm\s+\d{4}',
        r'Điều\s+\d+(?:,\s*Khoản\s+\d+)?[^.!?]*(?:Luật|Nghị định|Thông tư)\s+[^.!?]*?\s+năm\s+\d{4}',
    ]
    
    # Pattern 3: Web citation "Theo nguồn [Tên web]" hoặc "[Tên Web] cho biết..."
    citation_patterns.extend([
        r'Theo\s+nguồn\s+[^.!?]+',
        r'Nguồn:\s+[^.!?]+',
        r'\[Web\s+\d+\]'
    ])
    
    position = 0
    for pattern in citation_patterns:
        for match in re.finditer(pattern, answer):
            citation_text = match.group(0)
            
            # Kiểm tra xem đây là web citation hay luật
            is_web = any(term in citation_text.lower() for term in ["nguồn", "web", "http", "www"])
            
            citations.append({
                "text": citation_text,
                "source": citation_text if not is_web else "Nguồn Internet",
                "position": position,
                "url": "" # Fallback if regex can't find direct URL
            })
            position += 1
    
    return citations


def generator_node(state: GraphState) -> Dict[str, Any]:
    """
    Generator node: Sinh câu trả lời với trích dẫn dựa trên documents.
    
    Input từ state:
    - question: str
    - documents: List[Dict] (từ retriever)
    - web_results: List[Dict] (từ web_searcher, optional)
    
    Output cập nhật state:
    - answer: str (câu trả lời pháp lý trang trọng)
    - citations: List[Dict] (trích dẫn: text, source, position)
    - confidence: float (0.0 - 1.0)
    """
    try:
        question = state.get("question", "")
        documents = state.get("documents", [])
        web_results = state.get("web_results", [])
        
        if not question:
            logger.error("No question provided to generator")
            return {
                "answer": "Không có câu hỏi để trả lời.",
                "citations": [],
                "confidence": 0.0,
                "error": "Missing question",
                "hallucination_verdict": None
            }
        
        if not documents and not web_results:
            logger.warning("No documents or web results available")
            return {
                "answer": "Không tìm thấy tài liệu liên quan để trả lời câu hỏi này.",
                "citations": [],
                "confidence": 0.0,
                "error": "No documents found",
                "hallucination_verdict": None
            }
        
        logger.info("Generating answer...")
        
        # 2. Build prompt with optional feedback if it's a retry
        context = _format_documents_and_web(documents, web_results)
        
        feedback_text = ""
        hallucination_verdict = state.get("hallucination_verdict")
        hallucination_desc = state.get("hallucinations")
        error = state.get("error")

        
        if hallucination_verdict == "fail" and hallucination_desc:
            feedback_text = f"\nLƯU Ý: Câu trả lời trước đó của bạn đã bị từ chối vì lý do ảo giác: {hallucination_desc}. Vui lòng sửa lỗi này, bám sát tài liệu tham khảo và đảm bảo trích dẫn chính xác.\n"
        elif error and "ParseError" in str(error):
            feedback_text = "\nLƯU Ý: Câu trả lời trước đó của bạn bị lỗi định dạng JSON (có thể do bị cắt cụt). Vui lòng đảm bảo trả về một khối JSON HOÀN CHỈNH và đúng cấu trúc.\n"


        prompt = GENERATOR_PROMPT.format(
            question=question,
            context=context,
            feedback=feedback_text
        )
        logger.info("Generator prompt length: {} chars", len(prompt))

        
        # 3. Call LLM
        llm = get_model_with_fallback(purpose="generator", json_mode=True)
        response = llm.invoke(prompt)
        
        # 4. Parse JSON response
        answer_content = response.content
        logger.debug(f"Generator raw response type: {type(answer_content)}")
        logger.info("Generator raw response (first 200 chars): {}", str(answer_content)[:200])
        
        if isinstance(answer_content, list):
            # Xử lý trường hợp LangChain trả về list of dicts/strings
            logger.warning(f"Generator returned list content: {answer_content}")
            answer_content = "\n".join([str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in answer_content])

        try:
            # parse_json_response đã có sẵn logic xử lý markdown code fence và tìm block JSON
            result = parse_json_response(answer_content)
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict from parse_json_response, got {type(result)}")
        except Exception as parse_err:
            logger.error(f"Failed to parse JSON in generator: {parse_err}")
            return {
                "answer": f"Lỗi định dạng câu trả lời từ AI: {str(parse_err)[:100]}",
                "citations": [],
                "confidence": 0.0,
                "generation_attempt": state.get("generation_attempt", 0) + 1,
                "error": f"ParseError: {str(parse_err)}",
                "hallucination_verdict": None 
            }
        
        # 5. Extract answer, citations, confidence
        logger.info("Parsed result keys: {}", list(result.keys()))
        answer = result.get("answer")
        if not answer:
             logger.warning("Field 'answer' is missing or empty in parsed result!")
             # Thử tìm các field tương tự nếu LLM đặt tên sai
             answer = result.get("response") or result.get("content") or "Không có nội dung câu trả lời."
        
        raw_confidence = result.get("confidence", 0.5)

        # LLM đôi khi trả về string như "Không có thông tin" thay vì số → dùng safe cast
        try:
            confidence = float(raw_confidence)
        except (ValueError, TypeError):
            confidence = 0.5
        
        # Try to use citations from LLM, fallback to extraction
        try:
            citations = result.get("citations")
            if not isinstance(citations, list) or not citations:
                citations = _extract_citations(answer, documents)
        except Exception as e:
            logger.warning(f"Error extracting citations: {e}")
            citations = _extract_citations(answer, documents)
        
        logger.info(f"Generated answer with {len(citations)} citations")
        logger.debug(f"Confidence: {confidence}")
        
        # Lấy attempt hiện tại và cộng thêm 1
        current_attempt = state.get("generation_attempt", 0)
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "generation_attempt": current_attempt + 1,
            "error": None,
            "hallucination_verdict": None
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error("Error in generator_node: {}", error_msg[:500], exc_info=True)

        
        current_attempt = state.get("generation_attempt", 0)
        
        return {
            "answer": f"Lỗi khi sinh câu trả lời: {error_msg[:100]}",
            "citations": [],
            "confidence": 0.0,
            "generation_attempt": current_attempt + 1,
            "error": f"GeneratorError: {error_msg[:100]}",
            "hallucination_verdict": None
        }

