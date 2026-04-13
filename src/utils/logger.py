"""
src/utils/logger.py
Cấu hình Loguru logger dùng chung cho toàn project.

Cách dùng trong bất kỳ module nào:
    from src.utils.logger import logger
    logger.info("[MODULE] message")
"""
import sys
import io
from pathlib import Path
from loguru import logger

# Import lazy để tránh circular import (config cũng dùng logger đôi khi)
try:
    from src.config import LOG_LEVEL, IS_PRODUCTION, ROOT_DIR
except ImportError:  # fallback khi chạy độc lập
    LOG_LEVEL = "INFO"
    IS_PRODUCTION = False
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _get_utf8_stdout():
    """Trả về stdout wrapper với encoding UTF-8 để tránh UnicodeEncodeError trên Windows."""
    if hasattr(sys.stdout, "buffer"):
        return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    return sys.stdout


def setup_logger() -> None:
    """Khởi tạo Loguru với format và sink phù hợp với môi trường."""
    logger.remove()  # Xóa stderr handler mặc định

    # Format log dễ đọc, có màu trong development
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Luôn log ra stdout với UTF-8 để hỗ trợ tiếng Việt trên Windows
    logger.add(
        _get_utf8_stdout(),
        format=log_format,
        level=LOG_LEVEL,
        colorize=not IS_PRODUCTION,  # Tắt màu ANSI trong production / file
        enqueue=False,               # Synchronous trong dev – dễ debug
    )

    # Production: thêm file log với rotation & compression
    if IS_PRODUCTION:
        logs_dir = ROOT_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(logs_dir / "app_{time:YYYY-MM-DD}.log"),
            format=log_format,
            level="INFO",
            rotation="00:00",       # Rotate mỗi ngày lúc nửa đêm
            retention="7 days",     # Giữ log 7 ngày
            compression="zip",
            encoding="utf-8",
            colorize=False,
        )


# Tự setup khi import module này (idempotent nếu gọi lại)
setup_logger()

__all__ = ["logger", "setup_logger"]
