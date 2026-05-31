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
from src.graph.runtime_store import get_documents, get_web_results, put_citations
from src.graph.state import GraphState


GENERATOR_PROMPT = """
Bạn là trợ lý pháp lý Việt Nam. Trả lời câu hỏi chỉ dựa trên phần TÀI LIỆU THAM KHẢO bên dưới.

Quy tắc trả lời:
- Trả lời trực tiếp, súc tích, bằng Markdown dễ đọc.
- Chỉ nêu dữ kiện có trong tài liệu tham khảo; nếu thiếu căn cứ, nói rõ là chưa đủ dữ liệu.
- Mỗi kết luận pháp lý quan trọng phải có source id như [S1], [S2] hoặc [Web 1].
- Với văn bản luật, nêu tên văn bản, Điều/Khoản nếu có, trang nếu có.
- Với bảng/phụ lục, nêu bảng/phụ lục, trang và giá trị liên quan nếu có.
- Nếu chỉ dùng kết quả web, nói rõ đó là thông tin từ internet.
- Trả về duy nhất một JSON hợp lệ, không thêm chữ ngoài JSON.

Câu hỏi:
{question}

TÀI LIỆU THAM KHẢO:
{context}

{feedback}

JSON schema:
{{
  "answer": "<câu trả lời Markdown có citation [S...] hoặc [Web ...]>",
  "citations": [
    {{
      "text": "<cụm hoặc câu trong answer có citation>",
      "source": "<source id, ví dụ [S1]>",
      "position": <số nguyên>,
      "url": "<link nếu có>"
    }}
  ],
  "confidence": <số thực từ 0 đến 1>
}}
"""


def _page_label(metadata: Dict[str, Any]) -> str:
    page_start = metadata.get("page_start", "")
    page_end = metadata.get("page_end", "")
    if not page_start:
        return ""
    page_text = str(page_start)
    if page_end and page_end != page_start:
        page_text += f"-{page_end}"
    return page_text


def _title_from_slug(slug: str) -> str:
    words = [word for word in re.split(r"[-_\s]+", slug or "") if word and not word.isdigit()]
    if not words:
        return ""

    prefix = "Luật"
    if words[:2] == ["bo", "luat"]:
        prefix = "Bộ luật"
        words = words[2:]
    elif words[0] == "luat":
        words = words[1:]

    stop_words = {
        "so", "qh", "qh12", "qh13", "qh14", "qh15", "pdf", "html", "d1",
        "2024", "2025", "2026", "2019", "2007", "2012", "2014",
    }
    meaningful = [word for word in words if word not in stop_words and not re.fullmatch(r"\d+", word)]
    if not meaningful:
        return ""
    return f"{prefix} {' '.join(meaningful).capitalize()}"


def _clean_law_name(metadata: Dict[str, Any]) -> str:
    law_name = metadata.get("ten_van_ban", "") or metadata.get("doc_id", "")
    doc_id = metadata.get("doc_id", "")
    source_url = metadata.get("source_url", "")
    source_path = metadata.get("source_path", "")

    known_titles = {
        "45_2019_qh14": "Bộ luật Lao động",
        "luat_109_2025_qh15_pdf": "Luật Thuế thu nhập cá nhân",
        "luat_116_2025_qh15_pdf": "Luật An ninh mạng",
    }
    if doc_id in known_titles:
        return known_titles[doc_id]

    url_match = re.search(r"/([^/?#]+?)(?:-\d{4})?-so-", source_url or "")
    if url_match:
        title = _title_from_slug(url_match.group(1))
        if title:
            return title

    raw = re.sub(r"\.(pdf|docx?|html?)$", "", str(source_path or law_name), flags=re.IGNORECASE)
    raw = raw.split("/")[-1].split("\\")[-1]
    raw = re.sub(r"\bPDF\b", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("_", " ").replace("-", " ")
    raw = " ".join(raw.split())
    if raw.lower().startswith("luat "):
        return "Luật " + raw[5:]
    return raw or law_name


def _source_label(source_id: str, metadata: Dict[str, Any], include_source_id: bool = True) -> str:
    law_name = _clean_law_name(metadata)
    law_number = metadata.get("so_hieu_van_ban", "")
    dieu = metadata.get("dieu", "")
    khoang = metadata.get("khoang", "")
    level = metadata.get("level", "")
    table_id = metadata.get("table_id", "")
    page_text = _page_label(metadata)

    parts = [f"[{source_id}]"] if include_source_id else []
    parts.append(law_name)
    if law_number:
        parts.append(f"số {law_number}")
    if level == "table":
        parts.append(f"bảng {table_id}" if table_id else "bảng/phụ lục")
    else:
        if dieu:
            parts.append(dieu)
        if khoang:
            parts.append(khoang)
    if page_text:
        parts.append(f"trang {page_text}")
    return ", ".join(part for part in parts if part)


def _source_map_from_documents(documents: List[Dict]) -> Dict[str, Dict[str, Any]]:
    source_map: Dict[str, Dict[str, Any]] = {}
    for i, doc in enumerate(documents or [], 1):
        metadata = doc.get("metadata", {})
        source_id = f"S{i}"
        source_map[source_id] = {
            "source_id": source_id,
            "label": _source_label(source_id, metadata, include_source_id=False),
            "url": metadata.get("source_url", ""),
            "metadata": metadata,
        }
    return source_map


def _source_map_from_web(web_results: List[Dict] = None) -> Dict[str, Dict[str, Any]]:
    source_map: Dict[str, Dict[str, Any]] = {}
    for i, result in enumerate(web_results or [], 1):
        source_id = f"Web{i}"
        title = result.get("title", "") or "Nguồn Internet"
        url = result.get("url", "")
        source_map[source_id] = {
            "source_id": source_id,
            "label": f"{title} ({url})" if url else title,
            "url": url,
            "metadata": {"source_type": "web", "title": title},
        }
    return source_map


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
            source_id = f"S{i}"
            
            law_name = metadata.get("ten_van_ban", "")
            dieu = metadata.get("dieu", "")
            khoang = metadata.get("khoang", "")
            chunk_id = metadata.get("chunk_id", "")
            status = metadata.get("trang_thai", "")
            source_url = metadata.get("source_url", "")
            page_start = metadata.get("page_start", "")
            page_end = metadata.get("page_end", "")
            level = metadata.get("level", "")
            table_id = metadata.get("table_id", "")
            
            source_info = law_name
            if dieu:
                source_info += f" - {dieu}"
            if khoang:
                source_info += f" {khoang}"
            
            context_parts.append(f"\n[{source_id}] {source_info}")
            context_parts.append(f"Citation source: {_source_label(source_id, metadata)}")
            if chunk_id:
                context_parts.append(f"Chunk ID: {chunk_id}")
            if level:
                context_parts.append(f"Cấp: {level}")
            if table_id:
                context_parts.append(f"Table ID: {table_id}")
            if status:
                context_parts.append(f"Trạng thái: {status}")
            if page_start:
                page_text = str(page_start)
                if page_end and page_end != page_start:
                    page_text += f"-{page_end}"
                context_parts.append(f"Trang: {page_text}")
            if source_url:
                context_parts.append(f"URL nguồn: {source_url}")
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


def _canonical_source_id(source_id: str) -> str:
    source_id = (source_id or "").strip()
    if re.fullmatch(r"Web\s+\d+", source_id, flags=re.IGNORECASE):
        return re.sub(r"\s+", "", source_id.title())
    if re.fullmatch(r"S\d+", source_id, flags=re.IGNORECASE):
        return source_id.upper()
    return source_id


def _extract_source_id(text: str) -> str:
    match = re.search(r"\[(S\d+|Web\s+\d+)\]", text or "", flags=re.IGNORECASE)
    return _canonical_source_id(match.group(1)) if match else ""


def _extract_citation_marker(text: str) -> tuple[str, str]:
    match = re.search(r"\[((?:S\d+|Web\s+\d+)[^\]]*)\]", text or "", flags=re.IGNORECASE)
    if not match:
        return "", ""
    marker = " ".join(match.group(1).split())
    source_match = re.match(r"(S\d+|Web\s+\d+)", marker, flags=re.IGNORECASE)
    source_id = _canonical_source_id(source_match.group(1) if source_match else "")
    return marker, source_id


def _normalize_citations(
    citations: Any,
    answer: str,
    documents: List[Dict],
    web_results: List[Dict] = None,
) -> List[Dict]:
    source_map = _source_map_from_documents(documents)
    source_map.update(_source_map_from_web(web_results))
    normalized: List[Dict] = []

    if isinstance(citations, list):
        for index, citation in enumerate(citations):
            if not isinstance(citation, dict):
                continue
            text = str(citation.get("text") or citation.get("source") or "").strip()
            source = str(citation.get("source") or "").strip()
            marker, marker_source_id = _extract_citation_marker(text)
            if not marker:
                marker, marker_source_id = _extract_citation_marker(source)
            source_id = marker_source_id or _extract_source_id(text) or _extract_source_id(source)
            source_info = source_map.get(source_id)
            fallback_text = f"Nguồn {source_id}" if source_id else "Nguồn tham khảo"
            normalized.append({
                "text": text or source or fallback_text,
                "source": source_info["label"] if source_info else source,
                "position": int(citation.get("position", index) or index),
                "url": source_info["url"] if source_info else str(citation.get("url") or ""),
                "source_id": source_id,
                "marker": marker,
                "metadata": source_info["metadata"] if source_info else {},
            })

    existing_source_ids = {item.get("source_id") for item in normalized if item.get("source_id")}
    for source_id, source_info in source_map.items():
        if source_id in existing_source_ids:
            continue
        if re.search(rf"\[{re.escape(source_id)}\]", answer or "", flags=re.IGNORECASE):
            normalized.append({
                "text": f"Citation {source_id}",
                "source": source_info["label"],
                "position": len(normalized),
                "url": source_info["url"],
                "source_id": source_id,
                "marker": source_id,
                "metadata": source_info["metadata"],
            })

    if not normalized:
        normalized = _extract_citations(answer, documents)

    return normalized


def _display_citations(
    answer: str,
    citations: List[Dict],
    documents: List[Dict],
    web_results: List[Dict] = None,
) -> tuple[str, List[Dict]]:
    """Rewrite internal [Sx] citations to user-facing [n] citations."""
    source_map = _source_map_from_documents(documents)
    source_map.update(_source_map_from_web(web_results))
    marker_to_display: Dict[str, int] = {}
    source_to_markers: Dict[str, List[str]] = {}

    def remember_marker(marker: str) -> int:
        source_id = _extract_citation_marker(f"[{marker}]")[1]
        if marker not in marker_to_display:
            marker_to_display[marker] = len(marker_to_display) + 1
            if source_id:
                source_to_markers.setdefault(source_id, []).append(marker)
        return marker_to_display[marker]

    def replace_marker(match: re.Match) -> str:
        marker = " ".join(match.group(1).split())
        return f"[{remember_marker(marker)}]"

    display_answer = re.sub(
        r"\[((?:S\d+|Web\s+\d+)[^\]]*)\]",
        replace_marker,
        answer or "",
        flags=re.IGNORECASE,
    )

    display_citations: List[Dict] = []
    used_display_ids: set[int] = set()
    for citation in citations or []:
        if not isinstance(citation, dict):
            continue
        marker = citation.get("marker") or _extract_citation_marker(str(citation.get("text", "")))[0]
        source_id = citation.get("source_id") or _extract_citation_marker(f"[{marker}]")[1]
        if not marker and source_id:
            marker = source_to_markers.get(source_id, [source_id])[0]
        display_id = remember_marker(marker) if marker else len(marker_to_display) + 1
        if display_id in used_display_ids:
            continue
        used_display_ids.add(display_id)
        source_info = source_map.get(source_id, {})
        display_citations.append({
            **citation,
            "display_id": display_id,
            "source_id": source_id,
            "source": source_info.get("label") or citation.get("source", ""),
            "url": source_info.get("url") or citation.get("url", ""),
        })

    for marker, display_id in marker_to_display.items():
        if display_id in used_display_ids:
            continue
        source_id = _extract_citation_marker(f"[{marker}]")[1]
        source_info = source_map.get(source_id, {})
        display_citations.append({
            "text": f"Citation [{display_id}]",
            "source": source_info.get("label", ""),
            "position": len(display_citations),
            "url": source_info.get("url", ""),
            "source_id": source_id,
            "display_id": display_id,
            "metadata": source_info.get("metadata", {}),
        })

    display_citations.sort(key=lambda item: item.get("display_id") or 0)
    return display_answer, display_citations


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
        trace_id = state.get("trace_id") or state.get("request_id") or ""
        documents = get_documents(state)
        web_results = get_web_results(state)
        
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
            citations = _normalize_citations(result.get("citations"), answer, documents, web_results)
        except Exception as e:
            logger.warning(f"Error extracting citations: {e}")
            citations = _extract_citations(answer, documents)
        answer, citations = _display_citations(answer, citations, documents, web_results)
        
        citation_ids = put_citations(trace_id, citations)
        logger.info(f"Generated answer with {len(citations)} citations")
        logger.debug(f"Confidence: {confidence}")
        
        # Lấy attempt hiện tại và cộng thêm 1
        current_attempt = state.get("generation_attempt", 0)
        
        return {
            "answer": answer,
            "citation_ids": citation_ids,
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

