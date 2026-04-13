"""
src/utils/llm_factory.py
Khởi tạo LLM instance dựa theo config.
Hỗ trợ: Gemini (default), Claude, OpenAI.

Public API:
    get_llm()             → BaseChatModel (singleton, cached)
    parse_json_response() → dict  (parse JSON an toàn từ LLM output)
"""
import re
import json
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.config import (
    LLM_MODEL, LLM_TEMPERATURE,
    GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
)
from src.utils.logger import logger


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Trả về LLM instance (singleton – chỉ khởi tạo 1 lần nhờ lru_cache).
    Tự detect provider dựa vào tiền tố của LLM_MODEL:
      - ``gemini-*``  → Google Generative AI (ChatGoogleGenerativeAI)
      - ``claude-*``  → Anthropic (ChatAnthropic)
      - ``gpt-*``     → OpenAI (ChatOpenAI)

    Raises:
        ValueError: Nếu API key tương ứng chưa được set, hoặc model không được hỗ trợ.
    """
    model_lower = LLM_MODEL.lower()
    logger.info(f"[LLM] Khởi tạo model: {LLM_MODEL} (temperature={LLM_TEMPERATURE})")

    if model_lower.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY chưa được set trong .env. "
                "Lấy key tại https://aistudio.google.com/app/apikey"
            )
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=LLM_TEMPERATURE,
            convert_system_message_to_human=True,   # Gemini không có system role
        )

    elif model_lower.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY chưa được set trong .env. "
                "Lấy key tại https://console.anthropic.com/"
            )
        llm = ChatAnthropic(
            model=LLM_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=LLM_TEMPERATURE,
        )

    elif model_lower.startswith("gpt"):
        from langchain_openai import ChatOpenAI
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY chưa được set trong .env. "
                "Lấy key tại https://platform.openai.com/api-keys"
            )
        llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=LLM_TEMPERATURE,
        )

    else:
        raise ValueError(
            f"Model '{LLM_MODEL}' không được hỗ trợ. "
            "Dùng một trong: gemini-2.0-flash, gemini-1.5-pro, "
            "claude-3-5-sonnet-20241022, gpt-4o"
        )

    logger.info(f"[LLM] Model khởi tạo thành công: {LLM_MODEL}")
    return llm


def parse_json_response(text: str) -> dict[str, Any]:
    """
    Parse JSON từ LLM response một cách an toàn.

    LLM thường bọc JSON trong markdown code block (```json ... ```) hoặc
    thêm text giới thiệu trước JSON. Hàm này xử lý cả hai trường hợp.

    Args:
        text: Raw string từ LLM (``response.content``)

    Returns:
        dict parsed từ JSON

    Raises:
        ValueError: Nếu không tìm thấy JSON hợp lệ trong ``text``

    Example::

        raw = '```json\\n{\"intent\": \"legal_query\", \"confidence\": 0.95}\\n```'
        result = parse_json_response(raw)
        # → {"intent": "legal_query", "confidence": 0.95}
    """
    if not text or not text.strip():
        raise ValueError("LLM trả về chuỗi rỗng")

    # Bước 1: Loại bỏ markdown code fence (```json ... ``` hoặc ``` ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()

    # Bước 2: Thử parse trực tiếp
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Bước 3: Tìm JSON object đầu tiên trong text (bỏ qua prose trước/sau)
    match = re.search(r"\{[\s\S]+\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Không thể parse JSON từ LLM response.\n"
        f"Preview (300 ký tự đầu): {text[:300]}"
    )
