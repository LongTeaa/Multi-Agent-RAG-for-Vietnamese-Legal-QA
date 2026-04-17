"""
Step 9: Router Agent - Intent Classification

Chức năng: Phân loại loại câu hỏi của người dùng để quyết định routing
xử lý phù hợp (legal_query, procedural, out_of_scope, general_chat).
"""

from typing import Dict, Any
import json
from src.utils.llm_factory import get_model_with_fallback, parse_json_response
from src.utils.logger import logger
from src.graph.state import GraphState


ROUTER_PROMPT = """
Bạn là một classifier chuyên phân loại câu hỏi về lĩnh vực pháp lý Việt Nam.

Phân loại câu hỏi dưới đây vào một trong 4 loại:
1. legal_query: Hỏi về luật pháp, điều khoản cụ thể, quy định, tra cứu văn bản pháp luật
2. procedural: Hỏi về quy trình hành chính, bước thực hiện của một thủ tục
3. out_of_scope: Không liên quan đến pháp lý (ví dụ: hỏi công nghệ, thể thao, giải trí)
4. general_chat: Chào hỏi, xã giao thông thường

Vi đụ:
- "Thời giờ làm việc tối đa là bao nhiêu theo Bộ luật Lao động?" → legal_query
- "Điều kiện kết hôn theo Luật Hôn nhân và Gia đình là gì?" → legal_query
- "Thủ tục đăng ký kết hôn gồm những bước nào?" → procedural
- "Cách đăng ký tạm trú tại TP.HCM?" → procedural
- "ChatGPT là gì?" → out_of_scope
- "Giá vàng hôm nay là bao nhiêu?" → out_of_scope
- "Xin chào bạn" → general_chat
- "Bạn có khỏe không?" → general_chat

Câu hỏi: {question}

Trả về JSON hợp lệ theo đúng format sau (không thêm comment):
{{
  "intent": "<loại_intent>",
  "confidence": <số_thực_0_đến_1>,
  "reasoning": "<lý_do_phân_loại_ngắn_gọn>"
}}
"""


def router_node(state: GraphState) -> Dict[str, Any]:
    """
    Router node: Phân loại intent của câu hỏi người dùng.
    
    Input từ state:
    - question: str (câu hỏi tiếng Việt)
    
    Output cập nhật state:
    - intent: str (legal_query | procedural | out_of_scope | general_chat)
    - intent_confidence: float (0.0 - 1.0)
    """
    try:
        question = state.get("question", "")
        
        if not question:
            logger.error("No question provided to router")
            return {
                "intent": "general_chat",
                "intent_confidence": 0.0
            }
        
        logger.info(f"Classifying intent for: {question[:100]}...")
        
        # 1. Prepare prompt
        prompt = ROUTER_PROMPT.format(question=question)
        
        # 2. Call LLM (uses distributed project keys based on purpose)
        llm = get_model_with_fallback(purpose="router")
        
        # Xử lý trường hợp Gemini trả về list content (như đã fix ở generator)
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", part.get("content", str(part))))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)

        # 3. Parse JSON response
        try:
            result = parse_json_response(content)
        except Exception as parse_err:
            logger.error(f"Failed to parse JSON in router: {parse_err}")
            return {
                "intent": "general_chat",
                "intent_confidence": 0.0,
                "error": str(parse_err)
            }
        
        # 4. Extract intent and confidence
        intent = result.get("intent", "general_chat")
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")
        
        logger.info(f"Router classified as '{intent}' with confidence {confidence}")
        logger.debug(f"Reasoning: {reasoning}")
        
        # Validate intent values
        valid_intents = ["legal_query", "procedural", "out_of_scope", "general_chat"]
        if intent not in valid_intents:
            logger.warning(f"Invalid intent '{intent}', defaulting to 'general_chat'")
            intent = "general_chat"
        
        return {
            "intent": intent,
            "intent_confidence": float(confidence),
            "error": None
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error("Router failed: {}", error_msg[:500])

        
        # Check nếu là quota exceeded (không phải RPM)
        if "quota" in error_msg.lower() and "429" in error_msg:
            logger.error("!!! QUOTA EXHAUSTED !!! Bạn đã hết hạn mức sử dụng trong ngày (RPD).")
        elif "429" in error_msg:
            logger.warning("Rate limit hit (RPM) remains after retries.")
        
        return {
            "intent": "general_chat",
            "intent_confidence": 0.0,
            "error": f"Router error: {error_msg[:100]}"
        }
