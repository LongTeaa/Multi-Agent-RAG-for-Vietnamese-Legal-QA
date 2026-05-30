"""
Runtime context store for LangGraph requests.

The graph state should stay compact enough for retries, streaming updates, and
checkpointing. Large payloads such as retrieved documents and web snippets live
here and are fetched by trace_id plus compact ids.
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import models

from src.config import COLLECTION_NAME, ROOT_DIR
from src.data_pipeline.indexer import get_qdrant_client
from src.utils.logger import logger


MAX_CONTEXTS = 256
TTL_SECONDS = 60 * 60
_contexts: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _now() -> float:
    return time.time()


def _ensure_context(trace_id: str) -> dict[str, Any]:
    _prune_contexts()
    context = _contexts.get(trace_id)
    if context is None:
        context = {
            "created_at": _now(),
            "documents": {},
            "web_results": {},
            "citations": {},
            "events": [],
        }
        _contexts[trace_id] = context
    context["last_accessed_at"] = _now()
    _contexts.move_to_end(trace_id)
    return context


def _prune_contexts() -> None:
    cutoff = _now() - TTL_SECONDS
    expired = [
        trace_id
        for trace_id, context in _contexts.items()
        if context.get("last_accessed_at", context.get("created_at", 0)) < cutoff
    ]
    for trace_id in expired:
        _contexts.pop(trace_id, None)

    while len(_contexts) > MAX_CONTEXTS:
        _contexts.popitem(last=False)


def _payload_to_document(payload: dict[str, Any], score: float = 0.0) -> dict[str, Any]:
    metadata = {key: value for key, value in payload.items() if key != "chunk_text"}
    return {
        "content": payload.get("chunk_text", ""),
        "metadata": metadata,
        "score": float(score or 0.0),
    }


def put_documents(trace_id: str, documents: list[dict[str, Any]]) -> list[str]:
    context = _ensure_context(trace_id)
    ids: list[str] = []
    for index, doc in enumerate(documents or [], 1):
        metadata = doc.get("metadata", {})
        chunk_id = metadata.get("chunk_id") or f"doc_{index}"
        ids.append(chunk_id)
        context["documents"][chunk_id] = doc
    return ids


def get_documents(state: dict[str, Any], ids_key: str = "selected_context_ids") -> list[dict[str, Any]]:
    trace_id = state.get("trace_id") or state.get("request_id")
    ids = state.get(ids_key) or state.get("retrieved_chunk_ids") or []

    if not trace_id:
        return state.get("documents") or []

    context = _ensure_context(trace_id)
    documents = [context["documents"][chunk_id] for chunk_id in ids if chunk_id in context["documents"]]
    missing_ids = [chunk_id for chunk_id in ids if chunk_id not in context["documents"]]

    if missing_ids:
        fetched = fetch_documents_by_chunk_ids(missing_ids)
        put_documents(trace_id, fetched)
        fetched_by_id = {
            doc.get("metadata", {}).get("chunk_id"): doc
            for doc in fetched
        }
        documents.extend(fetched_by_id[chunk_id] for chunk_id in missing_ids if chunk_id in fetched_by_id)

    if documents:
        return documents
    return state.get("documents") or []


def fetch_documents_by_chunk_ids(chunk_ids: list[str]) -> list[dict[str, Any]]:
    if not chunk_ids:
        return []
    try:
        client = get_qdrant_client()
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_id",
                        match=models.MatchAny(any=list(dict.fromkeys(chunk_ids))),
                    )
                ]
            ),
            limit=len(set(chunk_ids)),
            with_payload=True,
            with_vectors=False,
        )
        by_id = {
            (record.payload or {}).get("chunk_id"): _payload_to_document(record.payload or {})
            for record in records
        }
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]
    except Exception as exc:
        logger.warning(f"[RUNTIME_STORE] Could not fetch documents by id: {exc}")
        return []


def put_web_results(trace_id: str, web_results: list[dict[str, Any]]) -> list[str]:
    context = _ensure_context(trace_id)
    ids: list[str] = []
    for index, result in enumerate(web_results or [], 1):
        result_id = result.get("result_id") or f"web_{index}"
        ids.append(result_id)
        context["web_results"][result_id] = {**result, "result_id": result_id}
    return ids


def get_web_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    trace_id = state.get("trace_id") or state.get("request_id")
    ids = state.get("web_result_ids") or []
    if not trace_id:
        return state.get("web_results") or []
    context = _ensure_context(trace_id)
    web_results = [context["web_results"][result_id] for result_id in ids if result_id in context["web_results"]]
    return web_results or state.get("web_results") or []


def put_citations(trace_id: str, citations: list[dict[str, Any]]) -> list[str]:
    context = _ensure_context(trace_id)
    ids: list[str] = []
    for index, citation in enumerate(citations or [], 1):
        citation_id = citation.get("citation_id") or f"citation_{index}"
        ids.append(citation_id)
        context["citations"][citation_id] = {**citation, "citation_id": citation_id}
    return ids


def get_citations(state: dict[str, Any]) -> list[dict[str, Any]]:
    trace_id = state.get("trace_id") or state.get("request_id")
    ids = state.get("citation_ids") or []
    if not trace_id:
        return state.get("citations") or []
    context = _ensure_context(trace_id)
    citations = [context["citations"][citation_id] for citation_id in ids if citation_id in context["citations"]]
    return citations or state.get("citations") or []


def record_agent_event(
    *,
    trace_id: str,
    agent: str,
    input_summary: str,
    output_summary: str,
    chunk_ids: list[str] | None = None,
    scores: dict[str, float] | None = None,
    latency_ms: int = 0,
    token_usage: dict[str, int] | None = None,
    error: str | None = None,
) -> None:
    if not trace_id:
        return

    context = _ensure_context(trace_id)
    event = {
        "trace_id": trace_id,
        "step": len(context["events"]) + 1,
        "agent": agent,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "chunk_ids": chunk_ids or [],
        "scores": scores or {},
        "latency_ms": latency_ms,
        "token_usage": token_usage or {"input": 0, "output": 0},
        "error": error,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    context["events"].append(event)
    _write_audit_event(event)


def _write_audit_event(event: dict[str, Any]) -> None:
    try:
        logs_dir = Path(ROOT_DIR) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"agent_audit_{datetime.utcnow():%Y-%m-%d}.jsonl"
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning(f"[RUNTIME_STORE] Could not write audit event: {exc}")

