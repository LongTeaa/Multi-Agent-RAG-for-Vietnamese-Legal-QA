"""
src/data_pipeline/qdrant_simple.py
Lightweight Qdrant wrapper using `requests` library to bypass httpx encoding issues on Windows.
This is a fallback when regular qdrant-client fails due to encoding problems.
"""
import json
import time
import uuid
from typing import List, Optional, Dict, Any

import requests
from loguru import logger

from src.config import QDRANT_HOST, QDRANT_PORT, SPARSE_RETRIEVAL_MODE, VECTOR_SIZE, COLLECTION_NAME


class QdrantSimpleClient:
    """Simple Qdrant client using requests (avoids httpx encoding issues)."""

    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()

    def create_collection(self, collection_name: str, vector_size: int = VECTOR_SIZE) -> bool:
        """Create collection if not exists."""
        try:
            # Check if exists
            url = f"{self.base_url}/collections/{collection_name}"
            resp = self.session.get(url)
            if resp.status_code == 200:
                logger.info(f"[QDRANT_SIMPLE] Collection '{collection_name}' already exists")
                self.create_payload_indexes(collection_name)
                return True

            # Create collection - PUT to /collections/{collection_name}
            create_url = f"{self.base_url}/collections/{collection_name}"
            payload = {
                "vectors": {
                    "default": {  # Named vector "default" for dense
                        "size": vector_size,
                        "distance": "Cosine",
                        "hnsw_config": {
                            "m": 16,
                            "ef_construct": 200,
                        },
                    }
                },
                "sparse_vectors": {
                    "bm25": {  # Named sparse vector "bm25"
                        "index": {
                            "on_disk": True
                        },
                        **({"modifier": "idf"} if SPARSE_RETRIEVAL_MODE == "qdrant_bm25" else {}),
                    }
                }
            }
            last_error = None
            for attempt in range(1, 4):
                try:
                    resp = self.session.put(create_url, json=payload, timeout=30)
                    resp.raise_for_status()
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"[QDRANT_SIMPLE] Create collection attempt {attempt}/3 failed: {e}")
                    time.sleep(1.5 * attempt)
            if last_error:
                raise last_error
            logger.info(f"[QDRANT_SIMPLE] Created collection '{collection_name}'")
            self.create_payload_indexes(collection_name)
            return True

        except Exception as e:
            logger.error(f"[QDRANT_SIMPLE] Failed to create collection: {e}")
            return False

    def create_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes used by filters and citations."""
        payload_fields = {
            "trang_thai": "keyword",
            "nam_ban_hanh": "integer",
            "loai_van_ban": "keyword",
            "so_hieu_van_ban": "keyword",
            "co_quan_ban_hanh": "keyword",
            "doc_id": "keyword",
            "source_url": "keyword",
            "page_start": "integer",
            "page_end": "integer",
            "level": "keyword",
            "parent_id": "keyword",
            "parent_article_id": "keyword",
            "article_number": "integer",
            "clause_number": "integer",
            "point_label": "keyword",
            "table_id": "keyword",
        }
        for field_name, field_schema in payload_fields.items():
            try:
                url = f"{self.base_url}/collections/{collection_name}/index"
                resp = self.session.put(
                    url,
                    json={"field_name": field_name, "field_schema": field_schema},
                    params={"wait": "true"},
                )
                if resp.status_code >= 400:
                    logger.debug(f"[QDRANT_SIMPLE] Payload index response for {field_name}: {resp.text[:300]}")
                resp.raise_for_status()
            except Exception as e:
                logger.debug(f"[QDRANT_SIMPLE] Payload index may already exist for {field_name}: {e}")

    def upsert_points(
        self, collection_name: str, points: List[Dict], wait: bool = True
    ) -> int:
        """Upsert points to collection."""
        try:
            url = f"{self.base_url}/collections/{collection_name}/points"

            # Convert to Qdrant point format
            qdrant_points = []
            for point in points:
                qdrant_points.append({
                    "id": point.get("id", int(uuid.uuid4().int % (2**63 - 1))),
                    "vector": point["vector"],
                    "payload": point.get("payload", {}),
                })

            payload = {
                "points": qdrant_points,
                "wait": wait,
            }

            resp = self.session.put(url, json=payload)
            if resp.status_code >= 400:
                logger.error(f"[QDRANT_SIMPLE] Upsert response: {resp.text[:1000]}")
            resp.raise_for_status()

            logger.info(f"[QDRANT_SIMPLE] Upserted {len(qdrant_points)} points")
            return len(qdrant_points)

        except Exception as e:
            logger.error(f"[QDRANT_SIMPLE] Failed to upsert points: {e}")
            return 0

    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get collection info."""
        try:
            url = f"{self.base_url}/collections/{collection_name}"
            resp = self.session.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"[QDRANT_SIMPLE] Failed to get collection info: {e}")
            return None

    def delete_collection(self, collection_name: str) -> bool:
        """Delete collection."""
        try:
            url = f"{self.base_url}/collections/{collection_name}"
            resp = self.session.delete(url)
            resp.raise_for_status()
            logger.info(f"[QDRANT_SIMPLE] Deleted collection '{collection_name}'")
            time.sleep(1.0)
            return True
        except Exception as e:
            logger.error(f"[QDRANT_SIMPLE] Failed to delete collection: {e}")
            return False


def upsert_chunks_simple(
    chunks: List[dict],
    collection_name: str = COLLECTION_NAME,
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
    recreate: bool = False,
) -> int:
    """
    Simple upsert without embedding (use pre-computed vectors from chunks).

    Args:
        chunks: List of dict with 'content', 'vector' (embedding), and metadata
        collection_name: Target collection
        host: Qdrant host
        port: Qdrant port
        recreate: Delete and recreate collection

    Returns:
        Number of upserted points
    """
    client = QdrantSimpleClient(host, port)
    from src.utils.embedding import make_sparse_vector_payload_json

    # Delete if recreate
    if recreate:
        client.delete_collection(collection_name)

    # Create collection
    if not client.create_collection(collection_name):
        logger.error("[QDRANT_SIMPLE] Failed to create collection")
        return 0

    # Prepare points
    points = []
    for chunk in chunks:
        if "vector" not in chunk:
            logger.warning("[QDRANT_SIMPLE] Chunk missing 'vector' field, skipping")
            continue

        point_id = int(uuid.uuid4().int % (2**63 - 1))
        payload = {
            k: v for k, v in chunk.items()
            if k not in ["vector", "chunk_index", "content"]
        }
        payload["chunk_text"] = chunk.get("content", "")

        sparse_vec = make_sparse_vector_payload_json(chunk.get("content", ""))

        points.append({
            "id": point_id,
            "vector": {
                "default": chunk["vector"],
                "bm25": sparse_vec
            },
            "payload": payload,
        })

    if not points:
        logger.error("[QDRANT_SIMPLE] No valid points to upsert")
        return 0

    # Upsert
    upserted = client.upsert_points(collection_name, points, wait=True)

    # Log stats
    if upserted > 0:
        info = client.get_collection_info(collection_name)
        if info:
            logger.info(f"[QDRANT_SIMPLE] Collection stats: {info.get('result', {}).get('points_count', 'N/A')} points")

    return upserted
