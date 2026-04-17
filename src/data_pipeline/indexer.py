"""
src/data_pipeline/indexer.py
Tạo Qdrant collection, embedding vectors, và upsert chunks vào Qdrant.

Features:
- Khởi tạo collection "vietnamese_legal_chunks" với dense + sparse vectors
- Batch upsert chunks (embedding + Qdrant upsert)
- Caching embed kết quả để tránh re-compute
"""
from typing import List, Optional
from dataclasses import dataclass
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    HnswConfigDiff,
    SparseVectorParams,
    SparseIndexParams,
)
from tqdm import tqdm

from src.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    VECTOR_SIZE,
)
from src.utils.logger import logger
from src.utils.embedding import embed_texts, embed_query, tokenize_for_bm25, generate_sparse_vector


@dataclass
class IndexConfig:
    """Cấu hình cho Qdrant indexing."""
    collection_name: str = COLLECTION_NAME
    vector_size: int = VECTOR_SIZE
    batch_size: int = 8   # Giảm từ 32 → 8 để tiết kiệm RAM
    hnsw_m: int = 16      # HNSW m parameter
    hnsw_ef_construct: int = 200  # HNSW ef_construct
    recreate: bool = False  # Xóa collection cũ trước khi tạo


def get_qdrant_client(
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
    api_key: Optional[str] = QDRANT_API_KEY,
) -> QdrantClient:
    """
    Khởi tạo Qdrant client (singleton-like). Xử lý encoding issues trên Windows.

    Args:
        host: Qdrant server host
        port: Qdrant server port
        api_key: API key (nếu cần auth)

    Returns:
        QdrantClient instance
    """
    import os
    import sys
    import io
    import warnings
    
    # FIX 1: Buộc UTF-8 environment variable
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    
    # FIX 2: Khôi phục stdout/stderr với UTF-8 nếu cần
    if sys.platform == "win32":
        try:
            if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != 'utf-8':
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', newline='')
            if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding.lower() != 'utf-8':
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', newline='')
        except Exception:
            pass  # Bỏ qua nếu không thể khôi phục
    
    # FIX 3: Patch httpx để sử dụng UTF-8 mặc định
    try:
        import httpx
        httpx.AsyncClient.__init__.__defaults__ = (
            None,  # auth
            None,  # cookies
            None,  # headers
            None,  # params
            None,  # timeout
            None,  # follow_redirects
            None,  # allow_redirects
            None,  # limits
            None,  # proxy
            None,  # proxies
            None,  # mounts
            None,  # trust_env
            None,  # event_hooks
            "utf-8",  # ← Buộc UTF-8 cho HTTP client
        )
    except Exception:
        pass  # Nếu không thể patch, continue
    
    # FIX 4: Suppress encoding warnings
    warnings.filterwarnings('ignore', message='.*ascii.*')
    warnings.filterwarnings('ignore', message='.*codec.*')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    url = f"http://{host}:{port}"
    api_key = api_key if api_key else None
    
    try:
        # FIX 5: Suppress httpx/urllib3 logging để tránh verbose output
        import logging
        for module_name in ['httpx', 'urllib3', 'qdrant_client']:
            logging.getLogger(module_name).setLevel(logging.CRITICAL)
        
        client = QdrantClient(url=url, api_key=api_key, timeout=30.0)
        logger.info(f"[INDEXER] Connected to Qdrant http://{host}:{port}")
        return client

    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"[INDEXER] Qdrant connection failed: {error_msg}")
        raise


def create_collection(
    client: QdrantClient,
    config: IndexConfig = IndexConfig(),
) -> None:
    """
    Tạo Qdrant collection với dense vectors + sparse BM25 indexing.

    Args:
        client: QdrantClient instance
        config: IndexConfig với cấu hình collection

    Raises:
        Exception: Nếu fail
    """
    try:
        # Kiểm tra xem collection đã tồn tại chưa
        collections = client.get_collections()
        existing_names = [c.name for c in collections.collections]

        if config.collection_name in existing_names:
            if config.recreate:
                logger.info(f"[INDEXER] Deleting existing collection: {config.collection_name}")
                client.delete_collection(config.collection_name)
            else:
                logger.info(f"[INDEXER] Collection '{config.collection_name}' already exists")
                return

        # Tạo collection mới
        logger.info(
            f"[INDEXER] Creating collection '{config.collection_name}' "
            f"(vectors={config.vector_size}, hnsw_m={config.hnsw_m})"
        )

        client.recreate_collection(
            collection_name=config.collection_name,
            vectors_config={
                "default": VectorParams(
                    size=config.vector_size,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=config.hnsw_m,
                        ef_construct=config.hnsw_ef_construct,
                    ),
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    index=SparseIndexParams(on_disk=True)  # Bật on_disk để tối ưu hiệu năng
                )
            },
        )

        # Tạo payload indices để filter nhanh
        payload_fields = {
            "trang_thai": "keyword",
            "nam_ban_hanh": "integer",
            "loai_van_ban": "keyword",
            "so_hieu_van_ban": "keyword",
            "co_quan_ban_hanh": "keyword",
        }

        for field_name, field_type in payload_fields.items():
            try:
                client.create_payload_index(
                    config.collection_name,
                    field_name,
                    field_type,
                )
                logger.debug(f"[INDEXER] Created payload index: {field_name}")
            except Exception as e:
                logger.debug(f"[INDEXER] Index {field_name} may already exist: {e}")

        logger.info(f"[INDEXER] ✓ Collection '{config.collection_name}' created successfully")

    except Exception as e:
        logger.error(f"[INDEXER] ✗ Failed to create collection: {e}")
        raise


def upsert_chunks(
    client: QdrantClient,
    chunks: List[dict],
    config: IndexConfig = IndexConfig(),
) -> int:
    """
    Embedding batch chunks và upsert vào Qdrant.

    Args:
        client: QdrantClient instance
        chunks: List of dict chunk objects (từ LegalChunk.to_dict())
        config: IndexConfig

    Returns:
        Số chunks được upsert thành công
    """
    if not chunks:
        logger.warning("[INDEXER] No chunks to upsert")
        return 0

    logger.info(f"[INDEXER] Upserting {len(chunks)} chunks to '{config.collection_name}'...")

    points = []
    chunk_contents = [c["content"] for c in chunks]

    # Embedding batch
    logger.info(f"[INDEXER] Embedding {len(chunk_contents)} chunks...")
    embeddings_batch = embed_texts(chunk_contents)

    if len(embeddings_batch) != len(chunks):
        logger.error(f"[INDEXER] Embedding count mismatch: {len(embeddings_batch)} vs {len(chunks)}")
        return 0

    # Tạo Point objects
    for i, (chunk, embedding) in enumerate(tqdm(zip(chunks, embeddings_batch), total=len(chunks), desc="Creating points")):
        point_id = int(uuid.uuid4().int % (2**63 - 1))  # Tạo UUID ngẫu nhiên

        # Payload: metadata + content
        payload = {
            k: v for k, v in chunk.items()
            if k not in ["content", "chunk_index"]  # Bỏ content khỏi payload (có thể lớn)
        }
        payload["chunk_text"] = chunk["content"]  # Store content với tên payload

        # Tạo sparse vector thật sự (hashing + TF)
        sparse_vec = generate_sparse_vector(chunk["content"])

        point = PointStruct(
            id=point_id,
            vector={
                "default": embedding,
                "bm25": sparse_vec
            },
            payload=payload,
        )
        points.append(point)

    # Batch upsert
    logger.info(f"[INDEXER] Upserting {len(points)} points to Qdrant...")
    try:
        client.upsert(
            collection_name=config.collection_name,
            points=points,
            wait=True,
        )
        logger.info(f"[INDEXER] ✓ Upserted {len(points)} points successfully")
        return len(points)

    except Exception as e:
        logger.error(f"[INDEXER] ✗ Failed to upsert: {e}")
        raise


def get_collection_stats(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """
    Lấy thống kê collection.

    Args:
        client: QdrantClient
        collection_name: Tên collection

    Returns:
        Dict với stats: {vectors_count, points_count, ...}
    """
    try:
        info = client.get_collection(collection_name)
        return {
            "vectors_count": info.points_count,
            "indexed": info.indexed_vectors_count if hasattr(info, "indexed_vectors_count") else "N/A",
        }
    except Exception as e:
        logger.error(f"[INDEXER] Failed to get collection stats: {e}")
        return {}


if __name__ == "__main__":
    # Test
    client = get_qdrant_client()
    config = IndexConfig(recreate=True)
    create_collection(client, config)

    # Sample nhat test
    from src.data_pipeline.chunker import LegalChunk

    sample_chunks = [
        LegalChunk(
            content="Điều 1. Phạm vi điều chỉnh",
            so_hieu_van_ban="45/2019/QH14",
            ten_van_ban="Bộ luật Lao động 2019",
            loai_van_ban="Luật",
            chuong="Chương I",
            dieu="Điều 1",
            khoang="",
            nam_ban_hanh=2019,
            trang_thai="Hiện hành",
            co_quan_ban_hanh="Quốc hội",
            ngay_hieu_luc="2021-01-01",
            chunk_index=0,
        ).to_dict()
    ]

    upserted = upsert_chunks(client, sample_chunks, config)
    logger.info(f"Upserted {upserted} chunks")

    stats = get_collection_stats(client)
    logger.info(f"Collection stats: {stats}")
