"""
src/data_pipeline/extractor.py
Extract text từ PDF và HTML sang plain text sạch.
"""
import os
from pathlib import Path
from typing import Optional

from src.utils.logger import logger


def extract_pdf(file_path: str) -> Optional[str]:
    """
    Extract text từ file PDF.

    Args:
        file_path: Đường dẫn đến file PDF

    Returns:
        Text string hoặc None nếu lỗi
    """
    try:
        import PyPDF2  # noqa

        text_parts = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    logger.debug(f"[EXTRACTOR] PDF page {page_num + 1}: {len(page_text)} chars")

        full_text = "\n".join(text_parts)
        logger.info(f"[EXTRACTOR] PDF extracted: {file_path} → {len(full_text)} chars")
        return full_text

    except Exception as e:
        logger.error(f"[EXTRACTOR] Lỗi extract PDF {file_path}: {e}")
        return None


def extract_html(file_path: str) -> Optional[str]:
    """
    Extract text từ file HTML.

    Args:
        file_path: Đường dẫn đến file HTML

    Returns:
        Text string hoặc None nếu lỗi
    """
    try:
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "lxml")

        # Xóa bỏ script, style, navigation
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        # Làm sạch dòng trống thừa
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)

        logger.info(f"[EXTRACTOR] HTML extracted: {file_path} → {len(text)} chars")
        return text

    except Exception as e:
        logger.error(f"[EXTRACTOR] Lỗi extract HTML {file_path}: {e}")
        return None


def extract_txt(file_path: str) -> Optional[str]:
    """
    Extract text từ file TXT.

    Args:
        file_path: Đường dẫn đến file TXT

    Returns:
        Text string hoặc None nếu lỗi
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Làm sạch: join dòng quá ngắn lại
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)
        logger.info(f"[EXTRACTOR] TXT extracted: {file_path} → {len(text)} chars")
        return text

    except Exception as e:
        logger.error(f"[EXTRACTOR] Lỗi extract TXT {file_path}: {e}")
        return None


def extract_file(file_path: str) -> Optional[str]:
    """
    Auto-detect format và extract text.

    Args:
        file_path: Đường dẫn file (TXT, PDF hoặc HTML)

    Returns:
        Text string hoặc None
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        return extract_txt(file_path)
    elif ext == ".pdf":
        return extract_pdf(file_path)
    elif ext in (".html", ".htm"):
        return extract_html(file_path)
    else:
        logger.warning(f"[EXTRACTOR] Định dạng không hỗ trợ: {ext}")
        return None


def extract_directory(input_dir: str) -> dict[str, str]:
    """
    Extract tất cả TXT/PDF/HTML trong một thư mục.

    Args:
        input_dir: Đường dẫn thư mục chứa file raw

    Returns:
        Dict {file_path: text_content}
    """
    results = {}
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.warning(f"[EXTRACTOR] Thư mục không tồn tại: {input_dir}")
        return results

    # Tìm tất cả file có thể xử lý
    supported_exts = {".txt", ".pdf", ".html", ".htm"}
    files = [f for f in input_path.rglob("*") if f.suffix.lower() in supported_exts]

    logger.info(f"[EXTRACTOR] Tìm thấy {len(files)} files trong {input_dir}")

    for file_path in files:
        text = extract_file(str(file_path))
        if text:
            results[str(file_path)] = text
            logger.debug(f"[EXTRACTOR] ✓ {file_path.name} ({len(text)} chars)")
        else:
            logger.warning(f"[EXTRACTOR] ✗ {file_path.name} (failed)")

    logger.info(f"[EXTRACTOR] Extract thành công {len(results)}/{len(files)} files")
    return results
