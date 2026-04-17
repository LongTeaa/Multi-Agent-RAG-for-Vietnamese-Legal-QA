"""
src/utils/llm_factory.py
Khởi tạo LLM instance dựa theo config.
Hỗ trợ: Gemini.

Public API:
    get_llm()             → BaseChatModel (singleton, cached)
    parse_json_response() → dict  (parse JSON an toàn từ LLM output)
"""
import re
import json
from functools import lru_cache
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableWithFallbacks

from src.config import (
    LLM_MODEL, LLM_TEMPERATURE,
    GEMINI_API_KEY, GEMINI_API_KEYS,
    LLM_MAX_RETRIES, LLM_REQUEST_TIMEOUT,
    FALLBACK_MODEL
)
from src.utils.logger import logger


@lru_cache(maxsize=20)
def get_llm(
    model_name: Optional[str] = None, 
    max_retries: Optional[int] = None,
    api_key: Optional[str] = None,
    json_mode: bool = False
) -> BaseChatModel:
    """
    Trả về Gemini LLM instance (cached theo model_name, retries, và api_key).
    """
    target_model = model_name or LLM_MODEL
    retries = max_retries if max_retries is not None else LLM_MAX_RETRIES
    
    # Chỉ hỗ trợ Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI
    actual_key = api_key or GEMINI_API_KEY
    if not actual_key:
        raise ValueError("Chưa cấu hình GEMINI_API_KEY")
        
    logger.debug(f"[LLM] Khởi tạo Gemini: {target_model} (retries={retries}, json_mode={json_mode})")
    
    kwargs = {
        "model": target_model,
        "google_api_key": actual_key,
        "temperature": LLM_TEMPERATURE,
        "max_retries": retries,
        "timeout": LLM_REQUEST_TIMEOUT,
        "max_output_tokens": 4096,
        "convert_system_message_to_human": True,
    }
    
    if json_mode:
        # LangChain Google GenAI mới nhất ưu tiên tham số top-level
        kwargs["response_mime_type"] = "application/json"
        
    return ChatGoogleGenerativeAI(**kwargs)



def get_model_with_fallback(primary_model: Optional[str] = None, purpose: str = "default", json_mode: bool = False) -> Any:
    """
    Tạo một runnable Gemini có khả năng tự động xoay vòng Key (Key Rotation).
    """
    target_primary = primary_model or LLM_MODEL
    
    if not GEMINI_API_KEYS:
        raise ValueError("Không tìm thấy GEMINI_API_KEY nào trong .env")
    
    # 1. Chuẩn bị danh sách keys
    keys = list(GEMINI_API_KEYS)
    
    # 2. Logic xoay vòng index dựa trên mục đích (Phân phối tải)
    if purpose != "default" and len(keys) > 1:
        import hashlib
        # Hash tên agent để lấy offset ổn định
        offset = int(hashlib.md5(purpose.encode()).hexdigest(), 16) % len(keys)
        # Đảo thứ tự keys: đưa key tại index 'offset' lên đầu
        keys = keys[offset:] + keys[:offset]
        logger.info(f"[LLM] Khởi tạo '{purpose}' sử dụng Project Key #{offset} làm primary (json_mode={json_mode})")
    
    # 3. Tạo danh sách các LLM instances với các keys đã sắp xếp
    gemini_models = []
    for i, key in enumerate(keys):
        # Key đầu tiên dùng retries=1 để failover sang key khác nhanh hơn nếu bị 429
        retries = 1 if i == 0 else 2
        gemini_models.append(get_llm(target_primary, max_retries=retries, api_key=key, json_mode=json_mode))

    
    main_llm = gemini_models[0]
    fallback_list = gemini_models[1:]
    
    # 4. Nếu còn GEMINI_API_KEYS khác, thiết lập fallback chaining
    if fallback_list:
        # 503 Service Unavailable, 429 Rate Limit, 500 Internal Error, 502 Bad Gateway
        # Có thể thêm các lỗi cụ thể từ Google API nếu LangChain không bắt hết
        return main_llm.with_fallbacks(fallback_list)
    
    return main_llm


def parse_json_response(text: str) -> dict[str, Any]:
    """
    Parse JSON từ LLM response một cách an toàn, hỗ trợ sửa lỗi JSON bị cắt cụt.
    """
    if not text or not text.strip():
        raise ValueError("LLM trả về chuỗi rỗng")

    # Bước 1: Tiền xử lý - Loại bỏ markdown code fence
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()

    def try_parse(json_str: str) -> Optional[dict]:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Thử sửa lỗi JSON bị cắt cụt bằng cách đóng dần các ngoặc
            for suffix in ["\"}", "}", "\"]}", "]}", "\"]}]}", "}]}", "\"]", "]", "\""]:
                try:
                    return json.loads(json_str + suffix)
                except:
                    continue
            return None

    # Bước 2: Thử parse toàn bộ chuỗi đã làm sạch
    result = try_parse(cleaned)
    if result and isinstance(result, dict) and "answer" in result:
        return result

    # Bước 3: Tìm tất cả các block {} khả thi
    # Dùng regex tìm tất cả các block từ dấu { đến dấu } cuối cùng có thể
    matches = re.findall(r"(\{[\s\S]*\})", text)
    if not matches:
        # Thử tìm block bị hở ở cuối (cắt cụt)
        matches = re.findall(r"(\{[\s\S]*)", text)

    potential_results = []
    for match in matches:
        parsed = try_parse(match)
        if parsed and isinstance(parsed, dict):
            # Ưu tiên block có chứa 'answer'
            if "answer" in parsed:
                return parsed
            potential_results.append(parsed)

    if potential_results:
        return potential_results[0]

    raise ValueError(
        f"Không thể parse JSON từ LLM response.\n"
        f"Preview (300 ký tự đầu): {text[:300]}"
    )

