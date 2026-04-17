"""
src/utils/embedding.py
Singleton wrapper cho SentenceTransformer embedding model.

Public API:
    embed_texts(texts)  → List[List[float]]   – batch encode documents
    embed_query(query)  → List[float]          – encode một câu hỏi

Prefix convention (intfloat/multilingual-e5-large-instruct):
    Document → "passage: <text>"
    Query    → "query: <text>"

Nếu dùng BAAI/bge-m3, prefix không bắt buộc nhưng không ảnh hưởng chất lượng.
"""
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_DEVICE, VECTOR_SIZE
from src.utils.logger import logger

# Models cần prefix theo chuẩn e5-instruct
_E5_MODELS = {
    "intfloat/multilingual-e5-large-instruct",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "intfloat/e5-large-v2",
    "intfloat/e5-base-v2",
}


def _needs_prefix(model_name: str) -> bool:
    """Kiểm tra xem model có cần prefix 'query:'/'passage:' không."""
    return model_name.lower() in {m.lower() for m in _E5_MODELS}


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Load SentenceTransformer model một lần duy nhất (singleton via lru_cache).
    """
    import os
    from src.config import HF_TOKEN
    
    # Set HF_TOKEN environment variable để huggingface_hub có thể dùng
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
        logger.debug("[EMBEDDING] Đã set HF_TOKEN từ config")
    
    # Kiểm tra nếu người dùng muốn chạy Offline hoàn toàn
    is_offline = os.environ.get("HF_HUB_OFFLINE") == "1"

    logger.info(
        f"[EMBEDDING] Đang khởi tạo model '{EMBEDDING_MODEL}' trên device='{EMBEDDING_DEVICE}'..."
    )
    
    if is_offline:
        logger.info("[EMBEDDING] Chế độ OFFLINE đang bật. Hệ thống sẽ chỉ sử dụng model đã tải sẵn.")
    else:
        logger.info(
            "[EMBEDDING] LƯU Ý: Nếu chạy lần đầu, quá trình này có thể mất vài phút để tải model. "
            "Các lần sau sẽ tự động dùng bản cache trên máy."
        )
    
    try:
        # Load model. 
        # Token sẽ được tự động nhận từ os.environ["HF_TOKEN"] nếu có.
        model = SentenceTransformer(
            EMBEDDING_MODEL, 
            device=EMBEDDING_DEVICE,
            trust_remote_code=True
        )
        
        # Log dimension để debug mismatch với VECTOR_SIZE trong config
        dim = model.get_embedding_dimension()
        logger.info(f"[EMBEDDING] Model đã tải xong. Embedding dimension: {dim}")
        
        if dim != VECTOR_SIZE:
            logger.warning(f"[EMBEDDING] Mismatch! Model dim {dim} != VECTOR_SIZE {VECTOR_SIZE} in config")
            
        return model
    except Exception as e:
        logger.error(f"[EMBEDDING] LỖI NGHIÊM TRỌNG: Không thể load model '{EMBEDDING_MODEL}': {e}")
        
        # Gợi ý cách fix nếu là lỗi mạng/auth
        if "unauthorized" in str(e).lower() or "not found" in str(e).lower():
            logger.error("[EMBEDDING] Kiểm tra lại HF_TOKEN hoặc kết nối mạng đến Hugging Face.")
            
        # Re-raise để caller (agent) biết và handle
        raise RuntimeError(f"Could not initialize embedding model: {e}")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Encode một batch văn bản thành dense vectors.

    Args:
        texts: Danh sách chuỗi văn bản (document chunks)

    Returns:
        List các vector float, shape [len(texts), embedding_dim]

    Example::

        vectors = embed_texts(["Điều 105. Thời giờ làm việc..."])
        # → [[0.023, -0.147, ...]]  (1024 chiều)
    """
    if not texts:
        return []

    model = _get_model()

    if _needs_prefix(EMBEDDING_MODEL):
        # E5 instruct models: prefix "passage: " cho documents
        inputs = [f"passage: {t}" for t in texts]
    else:
        inputs = texts

    embeddings = model.encode(
        inputs,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=8,            # Giảm từ 32 → 8 để tiết kiệm RAM (fix 'paging file too small')
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Encode một câu hỏi thành dense vector.

    Args:
        query: Câu hỏi người dùng

    Returns:
        Vector float 1D (độ dài = embedding_dim)

    Example::

        vector = embed_query("Thời gian làm việc tối đa là bao nhiêu?")
        # → [0.012, 0.089, ...]  (1024 chiều)
    """
    model = _get_model()

    if _needs_prefix(EMBEDDING_MODEL):
        # E5 instruct models: prefix "query: " cho query
        text = f"query: {query}"
    else:
        text = query

    embedding = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding.tolist()


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenize text để chuẩn bị cho BM25 sparse indexing.
    Đơn giản: chuyển lowercase, bỏ special chars, split theo whitespace.
    Lưu ý: Qdrant sẽ tự handle BM25 tokenization khi tạo sparse vectors.
    Hàm này dùng cho reference/testing.

    Args:
        text: String cần tokenize

    Returns:
        List[str]: Tokens (lowercase, cleaned)
    """
    import re

    if not text:
        return []

    # Chuyển lowercase
    text = text.lower()

    # Bỏ special characters, giữ lại chữ, số, dấu space
    text = re.sub(r"[^\w\s]", " ", text)

    # Split theo whitespace
    tokens = text.split()

    # Bỏ stop words ngắn (optional, improve search)
    stop_words = {"và", "hoặc", "là", "được", "có", "a", "an", "the", "in", "on", "at"}
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    return tokens


def generate_sparse_vector(text: str) -> dict:
    """
    Tạo sparse vector (keyword-based) từ văn bản.
    Sử dụng hashing để chuyển token thành index và tính toán tần suất (term frequency).

    Args:
        text: Văn bản cần xử lý

    Returns:
        Dict: {"indices": List[int], "values": List[float]} chuẩn Qdrant
    """
    import zlib
    from collections import Counter

    tokens = tokenize_for_bm25(text)
    if not tokens:
        return {"indices": [], "values": []}

    # Đếm tần suất
    counts = Counter(tokens)
    total = sum(counts.values())

    indices = []
    values = []

    for token, count in counts.items():
        # Hash token thành 32-bit integer (unsigned)
        # Qdrant dùng uint32 cho sparse indices
        idx = zlib.adler32(token.encode("utf-8")) & 0xFFFFFFFF
        indices.append(idx)
        # Term Frequency đơn giản (count / total)
        values.append(float(count / total))

    # Qdrant yêu cầu indices phải được sắp xếp tăng dần trong 1 số phiên bản
    combined = sorted(zip(indices, values))
    indices = [c[0] for c in combined]
    values = [c[1] for c in combined]

    return {"indices": indices, "values": values}


def get_vector_size() -> int:
    """
    Trả về kích thước vector của embedding model từ config.
    """
    return VECTOR_SIZE
