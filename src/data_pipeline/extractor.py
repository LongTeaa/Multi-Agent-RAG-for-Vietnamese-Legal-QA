"""
Extract legal source files into normalized text with page-level metadata.

Phase 1 ingestion uses PyMuPDF for PDF text, pdfplumber for tables, and
Tesseract OCR only when a page has too little extractable text.
"""
from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from src.config import (
    OCR_DPI,
    OCR_ENABLED,
    OCR_ENGINE,
    OCR_MIN_TEXT_CHARS,
    PROCESSED_DATA_DIR,
    TESSERACT_CMD,
    TESSERACT_LANG,
)
from src.utils.logger import logger


SUPPORTED_EXTS = {".txt", ".pdf", ".html", ".htm"}


@dataclass
class ExtractedBlock:
    block_id: str
    text: str
    bbox: list[float] = field(default_factory=list)


@dataclass
class ExtractedTable:
    table_id: str
    page_number: int
    rows: list[list[str]]
    table_markdown: str


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    blocks: list[ExtractedBlock] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    extraction_method: str = "text"
    ocr_lang: str = ""
    ocr_dpi: int = 0
    quality_score: float = 1.0


@dataclass
class ExtractedDocument:
    doc_id: str
    source_path: str
    source_url: str = ""
    official_source: bool = True
    file_hash: str = ""
    mime_type: str = ""
    pages: list[ExtractedPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving legal list markers."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "document"


def _infer_doc_id(file_path: Path) -> str:
    return _slugify(file_path.stem)


def _read_registry_seed(input_dir: Path) -> dict[str, dict[str, Any]]:
    """Load optional user-maintained registry metadata from raw directory."""
    candidates = [
        input_dir / "document_registry.jsonl",
        input_dir / "documents.jsonl",
        input_dir / "document_registry.csv",
        input_dir / "documents.csv",
    ]
    registry: dict[str, dict[str, Any]] = {}

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        key = item.get("source_path") or item.get("file_name") or item.get("doc_id")
                        if key:
                            registry[str(key)] = item
            elif path.suffix.lower() == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as f:
                    for item in csv.DictReader(f):
                        key = item.get("source_path") or item.get("file_name") or item.get("doc_id")
                        if key:
                            registry[str(key)] = item
            logger.info(f"[EXTRACTOR] Loaded registry seed: {path}")
        except Exception as exc:
            logger.warning(f"[EXTRACTOR] Could not read registry seed {path}: {exc}")

    return registry


def _lookup_seed(seed: dict[str, dict[str, Any]], file_path: Path) -> dict[str, Any]:
    keys = {
        str(file_path),
        file_path.as_posix(),
        file_path.name,
        file_path.stem,
        _infer_doc_id(file_path),
    }
    for key in keys:
        if key in seed:
            return dict(seed[key])
    return {}


def build_registry_record(file_path: Path, seed: dict[str, Any] | None = None) -> dict[str, Any]:
    seed = seed or {}
    source_path = str(file_path.as_posix())
    file_hash = _file_sha256(file_path)
    doc_id = seed.get("doc_id") or _infer_doc_id(file_path)
    return {
        "doc_id": doc_id,
        "source_path": seed.get("source_path") or source_path,
        "source_url": seed.get("source_url", ""),
        "official_source": str(seed.get("official_source", "true")).lower() != "false",
        "downloaded_at": seed.get("downloaded_at") or date.today().isoformat(),
        "file_hash": file_hash,
        "mime_type": seed.get("mime_type") or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream",
        "ingest_status": "raw",
    }


def write_document_registry(records: list[dict[str, Any]], output_dir: str | Path = PROCESSED_DATA_DIR) -> Path:
    output_path = Path(output_dir) / "document_registry.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"[EXTRACTOR] Wrote document registry: {output_path}")
    return output_path


def _page_quality(text: str) -> float:
    if not text:
        return 0.0
    replacement_ratio = text.count("\ufffd") / max(len(text), 1)
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(len(text), 1)
    length_score = min(len(text) / max(OCR_MIN_TEXT_CHARS, 1), 1.0)
    return max(0.0, min(1.0, (length_score * 0.6) + (alpha_ratio * 0.4) - replacement_ratio))


def _needs_ocr(text: str, quality_score: float) -> bool:
    return OCR_ENABLED and OCR_ENGINE.lower() == "tesseract" and (
        len(text.strip()) < OCR_MIN_TEXT_CHARS or quality_score < 0.25
    )


_TESSERACT_READY: Optional[bool] = None


def _tesseract_available() -> bool:
    global _TESSERACT_READY
    if _TESSERACT_READY is not None:
        return _TESSERACT_READY
    if TESSERACT_CMD:
        _TESSERACT_READY = Path(TESSERACT_CMD).exists()
    else:
        _TESSERACT_READY = shutil.which("tesseract") is not None
    if not _TESSERACT_READY:
        logger.warning("[EXTRACTOR] Tesseract is not available; OCR fallback will be skipped")
    return _TESSERACT_READY


def _ocr_page(page: Any) -> str:
    try:
        import fitz
        import io

        import pytesseract
        from PIL import Image

        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        pix = page.get_pixmap(matrix=fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72), alpha=False)
        image = pix.pil_tobytes(format="PNG")
        return normalize_text(pytesseract.image_to_string(Image.open(io.BytesIO(image)), lang=TESSERACT_LANG))
    except Exception as exc:
        logger.warning(f"[EXTRACTOR] OCR failed on page {page.number + 1}: {exc}")
        return ""


def _table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [[normalize_text(cell or "") for cell in row] + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    sep = ["---"] * width
    body = normalized[1:]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"

    return "\n".join([fmt(header), fmt(sep), *[fmt(row) for row in body]])


def _extract_pdf_tables(file_path: Path) -> dict[int, list[ExtractedTable]]:
    tables_by_page: dict[int, list[ExtractedTable]] = {}
    try:
        import pdfplumber
    except Exception as exc:
        logger.warning(f"[EXTRACTOR] pdfplumber unavailable, skipping tables: {exc}")
        return tables_by_page

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                for table_index, rows in enumerate(page.extract_tables() or [], start=1):
                    clean_rows = [[normalize_text(cell or "") for cell in row] for row in rows if row]
                    if not clean_rows:
                        continue
                    table = ExtractedTable(
                        table_id=f"{_infer_doc_id(file_path)}_p{page_index}_t{table_index}",
                        page_number=page_index,
                        rows=clean_rows,
                        table_markdown=_table_to_markdown(clean_rows),
                    )
                    tables_by_page.setdefault(page_index, []).append(table)
    except Exception as exc:
        logger.warning(f"[EXTRACTOR] Table extraction failed for {file_path}: {exc}")
    return tables_by_page


def extract_pdf_document(file_path: str, registry_record: dict[str, Any] | None = None) -> Optional[ExtractedDocument]:
    path = Path(file_path)
    record = registry_record or build_registry_record(path)
    try:
        import fitz
    except Exception as exc:
        logger.error(f"[EXTRACTOR] PyMuPDF is required for PDF extraction: {exc}")
        return None

    try:
        tables_by_page = _extract_pdf_tables(path)
        pages: list[ExtractedPage] = []
        with fitz.open(str(path)) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                blocks: list[ExtractedBlock] = []
                block_texts: list[str] = []
                for block_index, block in enumerate(page.get_text("blocks") or []):
                    if len(block) < 5:
                        continue
                    text = normalize_text(str(block[4]))
                    if not text:
                        continue
                    bbox = [float(block[0]), float(block[1]), float(block[2]), float(block[3])]
                    blocks.append(ExtractedBlock(f"p{page_index}_b{block_index}", text, bbox))
                    block_texts.append(text)

                page_text = normalize_text("\n".join(block_texts))
                quality = _page_quality(page_text)
                method = "pymupdf"
                ocr_text = ""
                if _needs_ocr(page_text, quality) and _tesseract_available():
                    ocr_text = _ocr_page(page)
                    if len(ocr_text) > len(page_text):
                        page_text = ocr_text
                        method = "ocr_tesseract"
                        quality = _page_quality(page_text)

                tables = tables_by_page.get(page_index, [])
                if tables:
                    table_text = "\n\n".join(table.table_markdown for table in tables if table.table_markdown)
                    page_text = normalize_text(f"{page_text}\n\n{table_text}")

                pages.append(
                    ExtractedPage(
                        page_number=page_index,
                        text=page_text,
                        blocks=blocks,
                        tables=tables,
                        extraction_method=method,
                        ocr_lang=TESSERACT_LANG if method == "ocr_tesseract" else "",
                        ocr_dpi=OCR_DPI if method == "ocr_tesseract" else 0,
                        quality_score=quality,
                    )
                )

        doc = ExtractedDocument(
            doc_id=record["doc_id"],
            source_path=record["source_path"],
            source_url=record.get("source_url", ""),
            official_source=bool(record.get("official_source", True)),
            file_hash=record.get("file_hash", ""),
            mime_type=record.get("mime_type", "application/pdf"),
            pages=pages,
            metadata=record,
        )
        logger.info(f"[EXTRACTOR] PDF extracted: {file_path} -> {len(doc.text)} chars, {len(pages)} pages")
        return doc
    except Exception as exc:
        logger.error(f"[EXTRACTOR] PDF extraction failed for {file_path}: {exc}")
        return None


def extract_txt_document(file_path: str, registry_record: dict[str, Any] | None = None) -> Optional[ExtractedDocument]:
    path = Path(file_path)
    record = registry_record or build_registry_record(path)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(text)
        page = ExtractedPage(page_number=1, text=text, extraction_method="txt", quality_score=_page_quality(text))
        return ExtractedDocument(
            doc_id=record["doc_id"],
            source_path=record["source_path"],
            source_url=record.get("source_url", ""),
            official_source=bool(record.get("official_source", True)),
            file_hash=record.get("file_hash", ""),
            mime_type=record.get("mime_type", "text/plain"),
            pages=[page],
            metadata=record,
        )
    except Exception as exc:
        logger.error(f"[EXTRACTOR] TXT extraction failed for {file_path}: {exc}")
        return None


def extract_html_document(file_path: str, registry_record: dict[str, Any] | None = None) -> Optional[ExtractedDocument]:
    path = Path(file_path)
    record = registry_record or build_registry_record(path)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = normalize_text(soup.get_text(separator="\n"))
        page = ExtractedPage(page_number=1, text=text, extraction_method="html", quality_score=_page_quality(text))
        return ExtractedDocument(
            doc_id=record["doc_id"],
            source_path=record["source_path"],
            source_url=record.get("source_url", ""),
            official_source=bool(record.get("official_source", True)),
            file_hash=record.get("file_hash", ""),
            mime_type=record.get("mime_type", "text/html"),
            pages=[page],
            metadata=record,
        )
    except Exception as exc:
        logger.error(f"[EXTRACTOR] HTML extraction failed for {file_path}: {exc}")
        return None


def extract_document(file_path: str, registry_record: dict[str, Any] | None = None) -> Optional[ExtractedDocument]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf_document(file_path, registry_record)
    if ext == ".txt":
        return extract_txt_document(file_path, registry_record)
    if ext in {".html", ".htm"}:
        return extract_html_document(file_path, registry_record)
    logger.warning(f"[EXTRACTOR] Unsupported format: {ext}")
    return None


def extract_documents(input_dir: str, registry_output_dir: str | Path = PROCESSED_DATA_DIR) -> list[ExtractedDocument]:
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.warning(f"[EXTRACTOR] Input directory does not exist: {input_dir}")
        return []

    seed = _read_registry_seed(input_path)
    files = sorted(f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS)
    logger.info(f"[EXTRACTOR] Found {len(files)} source files in {input_dir}")

    documents: list[ExtractedDocument] = []
    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for file_path in files:
        record = build_registry_record(file_path, _lookup_seed(seed, file_path))
        if record["file_hash"] in seen_hashes:
            logger.warning(f"[EXTRACTOR] Skipping duplicate file by hash: {file_path}")
            continue
        seen_hashes.add(record["file_hash"])
        records.append(record)

        document = extract_document(str(file_path), record)
        if document and document.text:
            documents.append(document)
            logger.debug(f"[EXTRACTOR] OK {file_path.name} ({len(document.text)} chars)")
        else:
            logger.warning(f"[EXTRACTOR] Failed or empty: {file_path.name}")

    write_document_registry(records, registry_output_dir)
    logger.info(f"[EXTRACTOR] Extracted {len(documents)}/{len(files)} documents")
    return documents


def extract_file(file_path: str) -> Optional[str]:
    """Backward-compatible helper returning only extracted text."""
    document = extract_document(file_path)
    return document.text if document else None


def extract_pdf(file_path: str) -> Optional[str]:
    document = extract_pdf_document(file_path)
    return document.text if document else None


def extract_html(file_path: str) -> Optional[str]:
    document = extract_html_document(file_path)
    return document.text if document else None


def extract_txt(file_path: str) -> Optional[str]:
    document = extract_txt_document(file_path)
    return document.text if document else None


def extract_directory(input_dir: str) -> dict[str, str]:
    """Backward-compatible helper returning {source_path: text}."""
    return {doc.source_path: doc.text for doc in extract_documents(input_dir)}
