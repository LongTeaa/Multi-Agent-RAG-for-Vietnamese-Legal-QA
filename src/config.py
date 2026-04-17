"""
src/config.py
Đọc toàn bộ biến môi trường từ .env và export typed constants cho cả project.

Mọi module khác chỉ cần:
    from src.config import LLM_MODEL, TOP_K_DOCUMENTS, ...
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Tìm file .env ở root project (2 cấp trên src/)
_ROOT_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _ROOT_DIR / ".env"

load_dotenv(dotenv_path=_ENV_FILE, override=False)

# ============================
# LLM APIs
# ============================
def _get_gemini_keys() -> list[str]:
    """Lấy danh sách tất cả Gemini API Keys từ .env."""
    keys = []
    # Check default key
    if os.getenv("GEMINI_API_KEY"):
        keys.append(os.getenv("GEMINI_API_KEY", ""))
    # Check numbered keys: GEMINI_API_KEY_1 to GEMINI_API_KEY_10
    for i in range(1, 11):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    # Loại bỏ trùng lặp và giữ nguyên thứ tự
    seen = set()
    return [x for x in keys if not (x in seen or seen.add(x))]

GEMINI_API_KEYS: list[str] = _get_gemini_keys()
GEMINI_API_KEY: str = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""

# Model cấu hình chung - người dùng chỉ cần sửa biến này trong .env
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Các agent có thể override model riêng nếu muốn, nếu không sẽ mặc định dùng LLM_MODEL
ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", LLM_MODEL)
GRADER_MODEL: str = os.getenv("GRADER_MODEL", LLM_MODEL)
GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", LLM_MODEL)
# Model dự phòng cuối cùng (nếu muốn override)
FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", LLM_MODEL)

# LLM Stability
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_REQUEST_TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))

# ============================
# Embedding Model
# ============================
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# Dense vector size tương ứng với embedding model sử dụng (phải khớp với VECTOR_SIZE trong qdrant config)
VECTOR_SIZE: int = 384  # paraphrase-multilingual-MiniLM-L12-v2 có embedding dimension là 384

# ============================
# Qdrant Vector DB
# ============================
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY: str | None = (os.getenv("QDRANT_API_KEY") or "").strip() or None
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "vietnamese_legal_chunks")

# ============================
# Web Search
# ============================
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ============================
# RAG Config
# ============================
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "5"))

# ============================
# App Config
# ============================
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
IS_PRODUCTION: bool = ENVIRONMENT == "production"

# ============================
# Path helpers
# ============================
ROOT_DIR: Path = _ROOT_DIR
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"


def validate_config() -> list[str]:
    """
    Kiểm tra các config bắt buộc đã được set chưa.
    Trả về list các warning (không raise exception để không block startup).
    """
    warnings: list[str] = []

    if not GEMINI_API_KEYS:
        warnings.append("Chưa cấu hình bất kỳ GEMINI_API_KEY nào – LLM sẽ không hoạt động")

    if not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY chưa được set – Web Search sẽ không hoạt động")

    if not HF_TOKEN:
        warnings.append("HF_TOKEN chưa được set – Việc tải model từ HF có thể bị giới hạn tốc độ")

    return warnings
