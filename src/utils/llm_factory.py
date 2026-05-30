"""
src/utils/llm_factory.py
Gemini LLM factory with key-rotation observability.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from functools import lru_cache
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from src.config import (
    GEMINI_API_KEY,
    GEMINI_API_KEYS,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
)
from src.utils.logger import logger


@lru_cache(maxsize=20)
def get_llm(
    model_name: Optional[str] = None,
    max_retries: Optional[int] = None,
    api_key: Optional[str] = None,
    json_mode: bool = False,
) -> BaseChatModel:
    """Return a cached Gemini chat model instance."""
    target_model = model_name or LLM_MODEL
    retries = max_retries if max_retries is not None else LLM_MAX_RETRIES

    from langchain_google_genai import ChatGoogleGenerativeAI

    actual_key = api_key or GEMINI_API_KEY
    if not actual_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    logger.debug(
        "[LLM] init Gemini model={} retries={} json_mode={}",
        target_model,
        retries,
        json_mode,
    )

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
        kwargs["response_mime_type"] = "application/json"

    return ChatGoogleGenerativeAI(**kwargs)


def _key_order_for_purpose(purpose: str, total_keys: int) -> list[int]:
    """Return a stable key-index order for an agent purpose."""
    if total_keys <= 0:
        return []
    if purpose != "default" and total_keys > 1:
        offset = int(hashlib.md5(purpose.encode()).hexdigest(), 16) % total_keys
        return list(range(offset, total_keys)) + list(range(0, offset))
    return list(range(total_keys))


def _error_kind(error: Exception) -> str:
    message = str(error).lower()
    if "429" in message or "quota" in message or "rate limit" in message or "ratelimit" in message:
        return "quota/rate_limit"
    if "timeout" in message or "deadline" in message:
        return "timeout"
    if "503" in message or "502" in message or "500" in message or "unavailable" in message:
        return "server_error"
    return type(error).__name__


class ObservedFallbackChain:
    """
    Lightweight fallback runnable that logs which safe key index is used.

    Raw API keys are never logged. Agents only rely on invoke/ainvoke, so this
    wrapper keeps the call surface small and explicit.
    """

    def __init__(
        self,
        *,
        purpose: str,
        model_name: str,
        json_mode: bool,
        candidates: list[tuple[int, Any]],
    ) -> None:
        self.purpose = purpose
        self.model_name = model_name
        self.json_mode = json_mode
        self.candidates = candidates
        self.key_order_indices = [key_index for key_index, _ in candidates]

        primary_key_index = self.key_order_indices[0] if self.key_order_indices else None
        fallback_key_indices = self.key_order_indices[1:]
        logger.info(
            "[LLM] agent={} model={} primary_key_index={} fallback_key_indices={} json_mode={}",
            purpose,
            model_name,
            primary_key_index,
            fallback_key_indices,
            json_mode,
        )

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for key_index, runnable in self.candidates:
            try:
                result = runnable.invoke(*args, **kwargs)
                logger.info("[LLM] agent={} key_index={} succeeded", self.purpose, key_index)
                return result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[LLM] agent={} key_index={} failed reason={}",
                    self.purpose,
                    key_index,
                    _error_kind(exc),
                )

        if last_error is not None:
            logger.error("[LLM] agent={} all_keys_failed", self.purpose)
            raise last_error
        raise ValueError("No LLM candidates configured")

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for key_index, runnable in self.candidates:
            try:
                if hasattr(runnable, "ainvoke"):
                    result = runnable.ainvoke(*args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                else:
                    result = runnable.invoke(*args, **kwargs)
                logger.info("[LLM] agent={} key_index={} succeeded", self.purpose, key_index)
                return result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[LLM] agent={} key_index={} failed reason={}",
                    self.purpose,
                    key_index,
                    _error_kind(exc),
                )

        if last_error is not None:
            logger.error("[LLM] agent={} all_keys_failed", self.purpose)
            raise last_error
        raise ValueError("No LLM candidates configured")


def get_model_with_fallback(
    primary_model: Optional[str] = None,
    purpose: str = "default",
    json_mode: bool = False,
) -> ObservedFallbackChain:
    """
    Create a Gemini runnable with deterministic key distribution and fallback.

    The primary key is chosen by hashing the agent purpose. If that key fails
    because of quota/rate limits or other transient errors, the wrapper tries
    the remaining configured keys in order and logs only their indices.
    """
    target_primary = primary_model or LLM_MODEL

    if not GEMINI_API_KEYS:
        raise ValueError("No GEMINI_API_KEY values found in .env")

    key_order = _key_order_for_purpose(purpose, len(GEMINI_API_KEYS))
    candidates: list[tuple[int, Any]] = []

    for position, key_index in enumerate(key_order):
        key = GEMINI_API_KEYS[key_index]
        retries = 1 if position == 0 else 2
        llm = get_llm(target_primary, max_retries=retries, api_key=key, json_mode=json_mode)
        candidates.append((key_index, llm))

    return ObservedFallbackChain(
        purpose=purpose,
        model_name=target_primary,
        json_mode=json_mode,
        candidates=candidates,
    )


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON safely from an LLM response."""
    if not text or not text.strip():
        raise ValueError("LLM returned an empty response")

    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()

    def try_parse(json_str: str) -> Optional[dict]:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            for suffix in ["\"}", "}", "\"]}", "]}", "\"]}]}", "}]}", "\"]", "]", "\""]:
                try:
                    return json.loads(json_str + suffix)
                except Exception:
                    continue
            return None

    result = try_parse(cleaned)
    if result and isinstance(result, dict) and "answer" in result:
        return result

    matches = re.findall(r"(\{[\s\S]*\})", text)
    if not matches:
        matches = re.findall(r"(\{[\s\S]*)", text)

    potential_results = []
    for match in matches:
        parsed = try_parse(match)
        if parsed and isinstance(parsed, dict):
            if "answer" in parsed:
                return parsed
            potential_results.append(parsed)

    if potential_results:
        return potential_results[0]

    raise ValueError(
        "Could not parse JSON from LLM response.\n"
        f"Preview (first 300 chars): {text[:300]}"
    )

