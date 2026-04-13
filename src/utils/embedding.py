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

from src.config import EMBEDDING_MODEL, EMBEDDING_DEVICE
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
    Thread-safe vì Python GIL.
    """
    logger.info(
        f"[EMBEDDING] Loading model '{EMBEDDING_MODEL}' on device='{EMBEDDING_DEVICE}'"
    )
    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    # Log dimension để debug mismatch với VECTOR_SIZE trong config
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"[EMBEDDING] Model loaded. Embedding dimension: {dim}")
    return model


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
        batch_size=32,           # Tối ưu RAM khi encode batch lớn
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


def get_vector_size() -> int:
    """
    Trả về kích thước vector của embedding model.
    intfloat/multilingual-e5-large-instruct: 1024
    """
    return 1024
