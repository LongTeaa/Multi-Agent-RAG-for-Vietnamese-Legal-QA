"""
Retriever Agent - hybrid search, metadata filters, optional rerank, context expansion.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List

from qdrant_client import models

from src.config import (
    COLLECTION_NAME,
    RERANK_ENABLED,
    RERANK_FINAL_K,
    RERANK_MODEL,
    RERANK_TOP_N,
    RETRIEVAL_DEFAULT_STATUS,
    RETRIEVAL_EXPAND_CONTEXT,
    RETRIEVAL_PREFETCH_MULTIPLIER,
    TOP_K_DOCUMENTS,
)
from src.data_pipeline.indexer import get_qdrant_client
from src.graph.runtime_store import put_documents
from src.graph.state import GraphState
from src.utils.embedding import embed_query, make_sparse_vector_input
from src.utils.logger import logger


LAW_NUMBER_PATTERN = re.compile(r"\b(\d{1,4})\s*[/\-_]\s*((?:19|20)\d{2})\s*[/\-_]\s*([A-ZĐ0-9]+)\b", re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"\b(?:điều|dieu)\s+(\d+[a-z]?)\b", re.IGNORECASE)
CLAUSE_PATTERN = re.compile(r"\b(?:khoản|khoan)\s+(\d+)\b", re.IGNORECASE)
POINT_PATTERN = re.compile(r"\b(?:điểm|diem)\s+([a-zđ])\b", re.IGNORECASE)
HISTORICAL_TERMS = ("hết hiệu lực", "trước đây", "tại thời điểm", "năm 20", "năm 19", "quá khứ")
TABLE_TERMS = ("bảng", "biểu", "danh mục", "mức giá", "giá đất", "đất ở", "đơn giá")
APPENDIX_TERMS = ("phụ lục", "phu luc", "appendix")
LAND_PRICE_TERMS = ("giá đất", "đất ở", "tên đường", "đường ")


def _match_value(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _match_any(key: str, values: list[str] | list[int]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


def _parse_query_filters(question: str) -> dict[str, Any]:
    """Extract conservative legal filters from the user query."""
    filters: dict[str, Any] = {}
    normalized = question.lower()

    law_match = LAW_NUMBER_PATTERN.search(question)
    if law_match:
        filters["so_hieu_van_ban"] = f"{law_match.group(1)}/{law_match.group(2)}/{law_match.group(3).upper()}"

    article_match = ARTICLE_PATTERN.search(question)
    if article_match:
        article_number = re.search(r"\d+", article_match.group(1))
        if article_number:
            filters["article_number"] = int(article_number.group(0))

    clause_match = CLAUSE_PATTERN.search(question)
    if clause_match and "article_number" in filters:
        filters["clause_number"] = int(clause_match.group(1))

    point_match = POINT_PATTERN.search(question)
    if point_match and "clause_number" in filters:
        filters["point_label"] = point_match.group(1).lower()

    if RETRIEVAL_DEFAULT_STATUS and not any(term in normalized for term in HISTORICAL_TERMS):
        filters["trang_thai"] = RETRIEVAL_DEFAULT_STATUS

    return filters


def _parse_query_preferences(question: str) -> dict[str, Any]:
    """Detect soft retrieval preferences that should not become hard filters."""
    normalized = question.lower()
    preferred_levels: list[str] = []
    preferred_doc_id_keywords: list[str] = []

    wants_table = any(term in normalized for term in TABLE_TERMS)
    wants_appendix = any(term in normalized for term in APPENDIX_TERMS)
    wants_land_price = any(term in normalized for term in LAND_PRICE_TERMS)

    if wants_table or wants_land_price:
        preferred_levels.append("table")

    if wants_appendix or wants_land_price:
        preferred_doc_id_keywords.append("phu_luc")

    return {
        "preferred_levels": preferred_levels,
        "preferred_doc_id_keywords": preferred_doc_id_keywords,
        "wants_table": wants_table,
        "wants_appendix": wants_appendix,
        "wants_land_price": wants_land_price,
    }


def _build_qdrant_filter(
    filters: dict[str, Any],
    extra_conditions: list[models.FieldCondition] | None = None,
) -> models.Filter | None:
    conditions: list[models.FieldCondition] = []
    for key, value in filters.items():
        if value in (None, "", []):
            continue
        conditions.append(_match_value(key, value))
    if extra_conditions:
        conditions.extend(extra_conditions)
    return models.Filter(must=conditions) if conditions else None


def _payload_to_document(payload: dict[str, Any], score: float = 0.0, expanded: bool = False) -> dict[str, Any]:
    metadata_keys = [
        "chunk_id",
        "doc_id",
        "so_hieu_van_ban",
        "ten_van_ban",
        "loai_van_ban",
        "chuong",
        "dieu",
        "khoang",
        "nam_ban_hanh",
        "trang_thai",
        "co_quan_ban_hanh",
        "ngay_hieu_luc",
        "source_url",
        "page_start",
        "page_end",
        "level",
        "parent_id",
        "parent_article_id",
        "prev_chunk_id",
        "next_chunk_id",
        "article_number",
        "clause_number",
        "point_label",
        "table_id",
    ]
    metadata = {key: payload.get(key, "") for key in metadata_keys}
    metadata["expanded_context"] = expanded
    return {
        "content": payload.get("chunk_text", ""),
        "metadata": metadata,
        "score": float(score or 0.0),
    }


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    logger.info(f"[RETRIEVER] Loading reranker: {RERANK_MODEL}")
    return CrossEncoder(RERANK_MODEL)


def _rerank_documents(question: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not RERANK_ENABLED or not documents:
        return documents[:TOP_K_DOCUMENTS]

    candidates = documents[: max(RERANK_TOP_N, TOP_K_DOCUMENTS)]
    try:
        reranker = _get_reranker()
        pairs = [(question, doc.get("content", "")) for doc in candidates]
        scores = reranker.predict(pairs)
        for doc, score in zip(candidates, scores):
            doc["score"] = float(score)
            doc["metadata"]["reranked"] = True
        candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        final_k = max(1, RERANK_FINAL_K or TOP_K_DOCUMENTS)
        logger.info(f"[RETRIEVER] Reranked {len(candidates)} candidates -> top {final_k}")
        return candidates[:final_k]
    except Exception as exc:
        logger.warning(f"[RETRIEVER] Rerank failed, using fused ranking: {exc}")
        return documents[:TOP_K_DOCUMENTS]


def _apply_metadata_boost(
    documents: list[dict[str, Any]],
    preferences: dict[str, Any],
) -> list[dict[str, Any]]:
    """Boost table/appendix candidates when the query explicitly asks for them."""
    preferred_levels = set(preferences.get("preferred_levels", []))
    doc_id_keywords = preferences.get("preferred_doc_id_keywords", [])

    boosted: list[dict[str, Any]] = []
    for doc in documents:
        metadata = doc.get("metadata", {})
        score = float(doc.get("score", 0.0))
        boost_reasons: list[str] = []

        if metadata.get("level") in preferred_levels:
            score += 0.60
            boost_reasons.append(f"level:{metadata.get('level')}")

        doc_id = str(metadata.get("doc_id", "")).lower()
        if any(keyword in doc_id for keyword in doc_id_keywords):
            score += 0.25
            boost_reasons.append("appendix_doc")

        if boost_reasons:
            doc = {**doc, "metadata": {**metadata, "boost_reasons": boost_reasons}, "score": score}
        boosted.append(doc)

    return sorted(boosted, key=lambda item: item.get("score", 0.0), reverse=True)


def _merge_documents(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for doc in group:
            chunk_id = doc.get("metadata", {}).get("chunk_id")
            if chunk_id and chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)
            merged.append(doc)
    return merged


def _query_hybrid(
    *,
    client,
    query_vector: list[float],
    sparse_vector: Any,
    query_filter: models.Filter | None,
    limit: int,
    prefetch_limit: int,
) -> list[dict[str, Any]]:
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_vector, using="default", limit=prefetch_limit),
            models.Prefetch(
                query=sparse_vector,
                using="bm25",
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [
        _payload_to_document(result.payload or {}, result.score or 0.0)
        for result in response.points
    ]


def _preferred_level_candidates(
    *,
    client,
    query_vector: list[float],
    sparse_vector: Any,
    query_filters: dict[str, Any],
    preferences: dict[str, Any],
    prefetch_limit: int,
) -> list[dict[str, Any]]:
    preferred_levels = preferences.get("preferred_levels", [])
    if not preferred_levels:
        return []

    candidates: list[dict[str, Any]] = []
    for level in preferred_levels:
        level_filter = _build_qdrant_filter(query_filters, extra_conditions=[_match_value("level", level)])
        candidates.extend(
            _query_hybrid(
                client=client,
                query_vector=query_vector,
                sparse_vector=sparse_vector,
                query_filter=level_filter,
                limit=TOP_K_DOCUMENTS,
                prefetch_limit=prefetch_limit,
            )
        )
    return candidates


def _scroll_by_filter(client, scroll_filter: models.Filter, limit: int = 12) -> list[dict[str, Any]]:
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=scroll_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [_payload_to_document(record.payload or {}, expanded=True) for record in records]


def _expand_context(client, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not RETRIEVAL_EXPAND_CONTEXT or not documents:
        return documents

    seen = {doc["metadata"].get("chunk_id") for doc in documents if doc.get("metadata")}
    expanded = list(documents)

    neighbor_ids: list[str] = []
    parent_article_ids: list[str] = []
    for doc in documents:
        metadata = doc.get("metadata", {})
        for key in ("prev_chunk_id", "next_chunk_id"):
            value = metadata.get(key)
            if value and value not in seen:
                neighbor_ids.append(value)
        parent_article_id = metadata.get("parent_article_id")
        if parent_article_id:
            parent_article_ids.append(parent_article_id)

    try:
        if neighbor_ids:
            for doc in _scroll_by_filter(client, models.Filter(must=[_match_any("chunk_id", list(set(neighbor_ids)))]), limit=len(set(neighbor_ids))):
                chunk_id = doc["metadata"].get("chunk_id")
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    expanded.append(doc)

        if parent_article_ids:
            parent_filter = models.Filter(
                must=[
                    _match_any("parent_article_id", list(set(parent_article_ids))),
                    _match_value("level", "article"),
                ]
            )
            for doc in _scroll_by_filter(client, parent_filter, limit=len(set(parent_article_ids))):
                chunk_id = doc["metadata"].get("chunk_id")
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    expanded.append(doc)
    except Exception as exc:
        logger.warning(f"[RETRIEVER] Context expansion failed: {exc}")
        return documents

    if len(expanded) > len(documents):
        logger.info(f"[RETRIEVER] Expanded context {len(documents)} -> {len(expanded)} chunks")
    return expanded


def retriever_node(state: GraphState) -> Dict[str, Any]:
    """
    Run Qdrant hybrid retrieval with metadata filters.

    Returns compact ids/scores only. Full documents are stored outside
    GraphState and fetched by later agents via trace_id.
    """
    try:
        question = state.get("question", "")
        trace_id = state.get("trace_id") or state.get("request_id") or ""
        if not question:
            logger.error("No question provided to retriever")
            return {"retrieved_chunk_ids": [], "selected_context_ids": [], "error": None}

        logger.info(f"Retrieving documents for question: {question[:100]}...")
        query_vector = embed_query(question)
        sparse_vector = make_sparse_vector_input(question)
        client = get_qdrant_client()

        query_filters = _parse_query_filters(question)
        query_preferences = _parse_query_preferences(question)
        qdrant_filter = _build_qdrant_filter(query_filters)
        prefetch_limit = max(TOP_K_DOCUMENTS * RETRIEVAL_PREFETCH_MULTIPLIER, TOP_K_DOCUMENTS)
        fusion_limit = max(RERANK_TOP_N if RERANK_ENABLED else TOP_K_DOCUMENTS, TOP_K_DOCUMENTS)

        fused_documents = _query_hybrid(
            client=client,
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            query_filter=qdrant_filter,
            limit=fusion_limit,
            prefetch_limit=prefetch_limit,
        )
        preferred_documents = _preferred_level_candidates(
            client=client,
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            query_filters=query_filters,
            preferences=query_preferences,
            prefetch_limit=prefetch_limit,
        )
        fused_documents = _merge_documents(preferred_documents, fused_documents)
        fused_documents = _apply_metadata_boost(fused_documents, query_preferences)
        logger.info(
            f"[RETRIEVER] Retrieved {len(fused_documents)} fused candidates "
            f"with filters={query_filters or '{}'} preferences={query_preferences or '{}'}"
        )

        selected_documents = _rerank_documents(question, fused_documents)
        documents = _expand_context(client, selected_documents)
        put_documents(trace_id, documents)

        retrieved_chunk_ids = [
            doc["metadata"].get("chunk_id", "")
            for doc in selected_documents
            if doc.get("metadata", {}).get("chunk_id")
        ]
        selected_context_ids = [
            doc["metadata"].get("chunk_id", "")
            for doc in documents
            if doc.get("metadata", {}).get("chunk_id")
        ]
        retrieved_scores = {
            doc["metadata"].get("chunk_id", ""): doc.get("score", 0.0)
            for doc in selected_documents
            if doc.get("metadata", {}).get("chunk_id")
        }

        logger.info(f"Successfully formatted {len(documents)} documents")
        return {
            "query_filters": query_filters,
            "query_preferences": query_preferences,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_scores": retrieved_scores,
            "selected_context_ids": selected_context_ids,
            "error": None,
        }

    except Exception as e:
        logger.error("Error in retriever_node: {}", e, exc_info=True)
        return {
            "retrieved_chunk_ids": [],
            "selected_context_ids": [],
            "retrieved_scores": {},
            "error": f"Retrieval failed: {str(e)}",
        }
