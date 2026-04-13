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
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

# ============================
# Embedding Model
# ============================
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"
)
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

# Dense vector size tương ứng với embedding model
# intfloat/multilingual-e5-large-instruct → 1024
# BAAI/bge-m3                             → 1024
VECTOR_SIZE: int = 1024

# ============================
# Qdrant Vector DB
# ============================
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
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

    model = LLM_MODEL.lower()
    if model.startswith("gemini") and not GEMINI_API_KEY:
        warnings.append("GEMINI_API_KEY chưa được set – LLM sẽ không hoạt động")
    elif model.startswith("claude") and not ANTHROPIC_API_KEY:
        warnings.append("ANTHROPIC_API_KEY chưa được set – LLM sẽ không hoạt động")
    elif model.startswith("gpt") and not OPENAI_API_KEY:
        warnings.append("OPENAI_API_KEY chưa được set – LLM sẽ không hoạt động")

    if not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY chưa được set – Web Search sẽ không hoạt động")

    return warnings
