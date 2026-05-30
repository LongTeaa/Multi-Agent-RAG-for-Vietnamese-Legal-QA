"""
Hierarchical chunking for Vietnamese legal documents.

Phase 2 adds:
- legal structure metadata: Phan/Chuong/Muc/Dieu/Khoan/Diem
- parent-child fields for article -> clause -> point retrieval
- standalone table chunks from pdfplumber output
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from src.data_pipeline.extractor import ExtractedDocument, ExtractedPage, ExtractedTable, extract_document, normalize_text
from src.utils.logger import logger


ARTICLE_PATTERN = re.compile(r"(?m)^Điều\s+(\d+[a-zA-Z]?)\s*[\.:]\s*(.*)$")
CLAUSE_PATTERN = re.compile(r"(?m)^\s*(\d+)\.\s+")
POINT_PATTERN = re.compile(r"(?m)^\s*([a-zđ])\)\s+", re.IGNORECASE)
PART_PATTERN = re.compile(r"(?m)^Phần\s+([A-ZĐIVXLCDM\d]+)\s*[\.:]?\s*(.*)$", re.IGNORECASE)
CHAPTER_PATTERN = re.compile(r"(?m)^Chương\s+([IVXLCDM\d]+)\s*[\.:]?\s*(.*)$", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"(?m)^Mục\s+(\d+[a-zA-Z]?)\s*[\.:]?\s*(.*)$", re.IGNORECASE)
SUBSECTION_PATTERN = re.compile(r"(?m)^Tiểu mục\s+(\d+[a-zA-Z]?)\s*[\.:]?\s*(.*)$", re.IGNORECASE)

MAX_CHUNK_CHARS = 2000
MAX_TABLE_CHARS = 1800
MIN_CHUNK_CHARS = 30


@dataclass
class LegalChunk:
    content: str
    so_hieu_van_ban: str
    ten_van_ban: str
    loai_van_ban: str
    chuong: str
    dieu: str
    khoang: str
    nam_ban_hanh: int
    trang_thai: str
    co_quan_ban_hanh: str
    ngay_hieu_luc: str
    chunk_index: int
    doc_id: str = ""
    chunk_id: str = ""
    source_path: str = ""
    source_url: str = ""
    official_source: bool = True
    file_hash: str = ""
    page_start: int = 0
    page_end: int = 0
    extraction_method: str = ""
    level: str = "article"
    parent_id: str = ""
    parent_article_id: str = ""
    prev_chunk_id: str = ""
    next_chunk_id: str = ""
    part: str = ""
    section: str = ""
    subsection: str = ""
    article_number: int = 0
    clause_number: int = 0
    point_label: str = ""
    table_id: str = ""
    table_markdown: str = ""
    table_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _infer_year(value: str) -> int:
    match = re.search(r"(19|20)\d{2}", value or "")
    return int(match.group(0)) if match else 0


def _infer_law_number(value: str) -> str:
    match = re.search(r"(\d{1,4})[_\-/ ]((?:19|20)\d{2})[_\-/ ]([A-ZĐ0-9]+)", value or "", re.IGNORECASE)
    if not match:
        return ""
    return f"{match.group(1)}/{match.group(2)}/{match.group(3).upper()}"


def _default_metadata(document: ExtractedDocument | None = None) -> dict[str, Any]:
    record = document.metadata if document else {}
    source_path = document.source_path if document else ""
    stem = Path(source_path).stem if source_path else ""
    metadata = {
        "so_hieu_van_ban": _infer_law_number(stem),
        "ten_van_ban": stem.replace("_", " ").strip(),
        "nam_ban_hanh": _infer_year(stem),
        "loai_van_ban": "Luật",
        "trang_thai": "Hiện hành",
        "co_quan_ban_hanh": "",
        "ngay_hieu_luc": "",
        "doc_id": document.doc_id if document else "",
        "source_path": source_path,
        "source_url": document.source_url if document else "",
        "official_source": document.official_source if document else True,
        "file_hash": document.file_hash if document else "",
    }
    for key in (
        "so_hieu_van_ban",
        "ten_van_ban",
        "nam_ban_hanh",
        "loai_van_ban",
        "trang_thai",
        "co_quan_ban_hanh",
        "ngay_hieu_luc",
        "source_url",
        "official_source",
    ):
        if record.get(key) not in (None, ""):
            metadata[key] = record[key]
    return metadata


def _extract_metadata_from_header(text: str) -> tuple[dict[str, Any], int]:
    metadata: dict[str, Any] = {
        "so_hieu_van_ban": "",
        "ten_van_ban": "",
        "nam_ban_hanh": 0,
        "loai_van_ban": "Luật",
        "trang_thai": "Hiện hành",
        "co_quan_ban_hanh": "",
        "ngay_hieu_luc": "",
    }
    mapping = {
        "SO_HIEU_VAN_BAN": "so_hieu_van_ban",
        "TEN_VAN_BAN": "ten_van_ban",
        "NAM_BAN_HANH": "nam_ban_hanh",
        "LOAI_VAN_BAN": "loai_van_ban",
        "TRANG_THAI": "trang_thai",
        "CO_QUAN_BAN_HANH": "co_quan_ban_hanh",
        "BAN_HANH": "co_quan_ban_hanh",
        "NGAY_HIEU_LUC": "ngay_hieu_luc",
    }
    body_start = 0
    for i, line in enumerate(text.splitlines()):
        if ":" not in line:
            body_start = i
            break
        key, value = line.split(":", 1)
        key = key.strip().upper()
        if key not in mapping:
            body_start = i
            break
        field = mapping[key]
        value = value.strip()
        if field == "nam_ban_hanh":
            try:
                metadata[field] = int(value)
            except ValueError:
                pass
        else:
            metadata[field] = value
    return metadata, body_start


def _page_lookup(document: ExtractedDocument) -> list[tuple[int, int, int, str]]:
    ranges: list[tuple[int, int, int, str]] = []
    cursor = 0
    for page in document.pages:
        text = page.text or ""
        start = cursor
        end = cursor + len(text)
        ranges.append((start, end, page.page_number, page.extraction_method))
        cursor = end + 2
    return ranges


def _page_range_for_span(ranges: list[tuple[int, int, int, str]], start: int, end: int) -> tuple[int, int, str]:
    pages = [page for range_start, range_end, page, _ in ranges if range_start <= end and range_end >= start]
    methods = [method for range_start, range_end, _, method in ranges if range_start <= end and range_end >= start and method]
    if not pages:
        return 0, 0, ""
    return min(pages), max(pages), ",".join(sorted(set(methods)))


def _label(match: re.Match[str], prefix: str) -> str:
    value = f"{prefix} {match.group(1)}"
    suffix = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
    if suffix:
        value = f"{value}. {suffix}"
    return value[:180]


def _last_label(pattern: re.Pattern[str], text_before: str, prefix: str) -> str:
    matches = list(pattern.finditer(text_before))
    return _label(matches[-1], prefix) if matches else ""


def _context_at(text_before: str) -> dict[str, str]:
    return {
        "part": _last_label(PART_PATTERN, text_before, "Phần"),
        "chuong": _last_label(CHAPTER_PATTERN, text_before, "Chương"),
        "section": _last_label(SECTION_PATTERN, text_before, "Mục"),
        "subsection": _last_label(SUBSECTION_PATTERN, text_before, "Tiểu mục"),
    }


def _article_spans(text: str) -> list[tuple[int, int, str, int, str]]:
    matches = list(ARTICLE_PATTERN.finditer(text))
    if not matches:
        return [(0, len(text), "", 0, "document")]

    spans: list[tuple[int, int, str, int, str]] = []
    if matches[0].start() > 0:
        spans.append((0, matches[0].start(), "", 0, "preamble"))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_label = f"Điều {match.group(1)}"
        if match.group(2).strip():
            article_label = f"{article_label}. {match.group(2).strip()}"
        article_number_match = re.search(r"\d+", match.group(1))
        article_number = int(article_number_match.group(0)) if article_number_match else 0
        spans.append((match.start(), end, article_label[:180], article_number, "article"))
    return spans


def _subspans(text: str, pattern: re.Pattern[str]) -> list[tuple[int, int, re.Match[str]]]:
    matches = list(pattern.finditer(text))
    spans: list[tuple[int, int, re.Match[str]]] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((match.start(), end, match))
    return spans


def _split_paragraphs(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[int, int, str]]:
    if len(text.strip()) <= max_chars:
        clean = text.strip()
        return [(0, len(text), clean)] if clean else []

    spans: list[tuple[int, int, str]] = []
    current_start: Optional[int] = None
    current_end = 0
    current_parts: list[str] = []
    for match in re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, re.DOTALL):
        paragraph = match.group(0).strip()
        if not paragraph:
            continue
        if current_parts and sum(len(part) + 2 for part in current_parts) + len(paragraph) > max_chars:
            spans.append((current_start or 0, current_end, "\n\n".join(current_parts).strip()))
            current_start = None
            current_parts = []
        if len(paragraph) > max_chars:
            for offset in range(0, len(paragraph), max_chars):
                piece = paragraph[offset:offset + max_chars].strip()
                if piece:
                    spans.append((match.start() + offset, min(match.start() + offset + max_chars, match.end()), piece))
            current_start = None
            current_parts = []
            current_end = match.end()
        else:
            if current_start is None:
                current_start = match.start()
            current_parts.append(paragraph)
            current_end = match.end()
    if current_parts:
        spans.append((current_start or 0, current_end, "\n\n".join(current_parts).strip()))
    return spans


def _compose_content(
    metadata: dict[str, Any],
    ctx: dict[str, str],
    article_label: str,
    clause_label: str,
    point_label: str,
    body: str,
    level: str,
) -> str:
    header = [
        f"Văn bản: {metadata.get('ten_van_ban') or ''}".strip(),
        f"Số hiệu: {metadata.get('so_hieu_van_ban') or ''}".strip(),
    ]
    for key in ("part", "chuong", "section", "subsection"):
        if ctx.get(key):
            header.append(ctx[key])
    if article_label:
        header.append(article_label)
    if clause_label:
        header.append(clause_label)
    if point_label:
        header.append(f"Điểm {point_label})")
    header.append(f"Cấp chunk: {level}")
    return normalize_text("\n".join(header) + "\n\n" + body.strip())


def _make_chunk(
    *,
    document: ExtractedDocument,
    metadata: dict[str, Any],
    chunk_index: int,
    content: str,
    level: str,
    ctx: dict[str, str],
    article_label: str,
    article_number: int,
    clause_number: int,
    point_label: str,
    parent_id: str,
    parent_article_id: str,
    page_start: int,
    page_end: int,
    extraction_method: str,
    table_id: str = "",
    table_markdown: str = "",
    table_json: str = "",
) -> LegalChunk:
    chunk_id = f"{document.doc_id}_c{chunk_index:05d}"
    clause_label = f"Khoản {clause_number}" if clause_number else ""
    return LegalChunk(
        content=content,
        so_hieu_van_ban=str(metadata["so_hieu_van_ban"]),
        ten_van_ban=str(metadata["ten_van_ban"]),
        loai_van_ban=str(metadata["loai_van_ban"]),
        chuong=ctx.get("chuong", ""),
        dieu=article_label,
        khoang=clause_label,
        nam_ban_hanh=int(metadata["nam_ban_hanh"] or 0),
        trang_thai=str(metadata["trang_thai"]),
        co_quan_ban_hanh=str(metadata["co_quan_ban_hanh"]),
        ngay_hieu_luc=str(metadata["ngay_hieu_luc"]),
        chunk_index=chunk_index,
        doc_id=document.doc_id,
        chunk_id=chunk_id,
        source_path=document.source_path,
        source_url=str(metadata.get("source_url") or document.source_url or ""),
        official_source=bool(metadata.get("official_source", document.official_source)),
        file_hash=document.file_hash,
        page_start=page_start,
        page_end=page_end,
        extraction_method=extraction_method,
        level=level,
        parent_id=parent_id,
        parent_article_id=parent_article_id,
        part=ctx.get("part", ""),
        section=ctx.get("section", ""),
        subsection=ctx.get("subsection", ""),
        article_number=article_number,
        clause_number=clause_number,
        point_label=point_label,
        table_id=table_id,
        table_markdown=table_markdown,
        table_json=table_json,
    )


def _link_neighbors(chunks: list[LegalChunk]) -> None:
    by_parent: dict[str, list[LegalChunk]] = {}
    for chunk in chunks:
        key = chunk.parent_id or chunk.parent_article_id or chunk.chunk_id
        by_parent.setdefault(key, []).append(chunk)
    for siblings in by_parent.values():
        siblings.sort(key=lambda item: item.chunk_index)
        for i, chunk in enumerate(siblings):
            if i > 0:
                chunk.prev_chunk_id = siblings[i - 1].chunk_id
            if i + 1 < len(siblings):
                chunk.next_chunk_id = siblings[i + 1].chunk_id


def _table_chunks(document: ExtractedDocument, metadata: dict[str, Any], start_index: int) -> list[LegalChunk]:
    chunks: list[LegalChunk] = []
    chunk_index = start_index
    for page in document.pages:
        for table in page.tables:
            table_rows = table.rows or []
            if not table_rows and not table.table_markdown:
                continue
            row_groups = _split_table_rows(table, max_chars=MAX_TABLE_CHARS)
            for group_index, markdown in enumerate(row_groups, start=1):
                table_id = table.table_id if len(row_groups) == 1 else f"{table.table_id}_part_{group_index}"
                body = normalize_text(markdown)
                if not body:
                    continue
                content = normalize_text(
                    "\n".join([
                        f"Văn bản: {metadata.get('ten_van_ban') or ''}",
                        f"Số hiệu: {metadata.get('so_hieu_van_ban') or ''}",
                        f"Bảng: {table_id}",
                        f"Trang: {table.page_number}",
                        "Cấp chunk: table",
                        "",
                        body,
                    ])
                )
                table_json = json.dumps(
                    {
                        "table_id": table_id,
                        "source_table_id": table.table_id,
                        "page": table.page_number,
                        "rows": table_rows,
                        "table_markdown": markdown,
                    },
                    ensure_ascii=False,
                )
                chunks.append(
                    _make_chunk(
                        document=document,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        content=content,
                        level="table",
                        ctx={},
                        article_label="",
                        article_number=0,
                        clause_number=0,
                        point_label="",
                        parent_id=document.doc_id,
                        parent_article_id="",
                        page_start=table.page_number,
                        page_end=table.page_number,
                        extraction_method=page.extraction_method,
                        table_id=table_id,
                        table_markdown=markdown,
                        table_json=table_json,
                    )
                )
                chunk_index += 1
    return chunks


def _split_table_rows(table: ExtractedTable, max_chars: int = MAX_TABLE_CHARS) -> list[str]:
    rows = table.rows or []
    if not rows:
        return [table.table_markdown] if table.table_markdown else []
    header = rows[0]
    body_rows = rows[1:] or rows
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_len = len(_rows_to_markdown([header]))
    for row in body_rows:
        row_len = len(" | ".join(row))
        if current and current_len + row_len > max_chars:
            groups.append(current)
            current = []
            current_len = len(_rows_to_markdown([header]))
        current.append(row)
        current_len += row_len
    if current:
        groups.append(current)
    if not groups:
        return [table.table_markdown]
    return [_rows_to_markdown([header] + group) for group in groups]


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [[normalize_text(cell or "") for cell in row] + [""] * (width - len(row)) for row in rows]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"

    return "\n".join([fmt(normalized[0]), fmt(["---"] * width), *[fmt(row) for row in normalized[1:]]])


def _build_article_chunks(
    document: ExtractedDocument,
    metadata: dict[str, Any],
    body_text: str,
    body_offset: int,
    page_ranges: list[tuple[int, int, int, str]],
    start_index: int,
) -> list[LegalChunk]:
    chunks: list[LegalChunk] = []
    chunk_index = start_index

    for article_start, article_end, article_label, article_number, article_level in _article_spans(body_text):
        raw_article = body_text[article_start:article_end].strip()
        if len(raw_article) < MIN_CHUNK_CHARS:
            continue
        ctx = _context_at(body_text[:article_start])
        parent_article_id = f"{document.doc_id}_article_{article_number}" if article_number else f"{document.doc_id}_article_{chunk_index}"
        absolute_article_start = body_offset + article_start
        absolute_article_end = body_offset + article_end
        page_start, page_end, extraction_method = _page_range_for_span(page_ranges, absolute_article_start, absolute_article_end)

        if article_level != "article":
            for local_start, local_end, part in _split_paragraphs(raw_article):
                absolute_start = absolute_article_start + local_start
                absolute_end = absolute_article_start + local_end
                ps, pe, method = _page_range_for_span(page_ranges, absolute_start, absolute_end)
                level = article_level if len(part) <= MAX_CHUNK_CHARS else f"{article_level}_part"
                chunks.append(
                    _make_chunk(
                        document=document,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        content=part,
                        level=level,
                        ctx=ctx,
                        article_label="",
                        article_number=0,
                        clause_number=0,
                        point_label="",
                        parent_id=document.doc_id,
                        parent_article_id="",
                        page_start=ps or page_start,
                        page_end=pe or page_end,
                        extraction_method=method or extraction_method,
                    )
                )
                chunk_index += 1
            continue

        if len(raw_article) <= MAX_CHUNK_CHARS:
            content = _compose_content(metadata, ctx, article_label, "", "", raw_article, "article")
            chunks.append(
                _make_chunk(
                    document=document,
                    metadata=metadata,
                    chunk_index=chunk_index,
                    content=content,
                    level="article",
                    ctx=ctx,
                    article_label=article_label,
                    article_number=article_number,
                    clause_number=0,
                    point_label="",
                    parent_id=document.doc_id,
                    parent_article_id=parent_article_id,
                    page_start=page_start,
                    page_end=page_end,
                    extraction_method=extraction_method,
                )
            )
            chunk_index += 1
            continue

        clause_spans = _subspans(raw_article, CLAUSE_PATTERN)
        if not clause_spans:
            for part_number, (local_start, local_end, part) in enumerate(_split_paragraphs(raw_article), start=1):
                absolute_start = absolute_article_start + local_start
                absolute_end = absolute_article_start + local_end
                ps, pe, method = _page_range_for_span(page_ranges, absolute_start, absolute_end)
                content = _compose_content(metadata, ctx, article_label, f"Phần {part_number}", "", part, "article_part")
                chunks.append(
                    _make_chunk(
                        document=document,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        content=content,
                        level="article_part",
                        ctx=ctx,
                        article_label=article_label,
                        article_number=article_number,
                        clause_number=0,
                        point_label="",
                        parent_id=parent_article_id,
                        parent_article_id=parent_article_id,
                        page_start=ps,
                        page_end=pe,
                        extraction_method=method,
                    )
                )
                chunk_index += 1
            continue

        prefix = raw_article[:clause_spans[0][0]].strip()
        for clause_start, clause_end, clause_match in clause_spans:
            clause_number = int(clause_match.group(1))
            clause_text = raw_article[clause_start:clause_end].strip()
            if prefix and not clause_text.startswith(article_label):
                clause_text_for_embedding = f"{prefix}\n{clause_text}"
            else:
                clause_text_for_embedding = clause_text
            absolute_clause_start = absolute_article_start + clause_start
            absolute_clause_end = absolute_article_start + clause_end
            ps, pe, method = _page_range_for_span(page_ranges, absolute_clause_start, absolute_clause_end)

            if len(clause_text_for_embedding) <= MAX_CHUNK_CHARS:
                content = _compose_content(
                    metadata,
                    ctx,
                    article_label,
                    f"Khoản {clause_number}",
                    "",
                    clause_text_for_embedding,
                    "clause",
                )
                chunks.append(
                    _make_chunk(
                        document=document,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        content=content,
                        level="clause",
                        ctx=ctx,
                        article_label=article_label,
                        article_number=article_number,
                        clause_number=clause_number,
                        point_label="",
                        parent_id=parent_article_id,
                        parent_article_id=parent_article_id,
                        page_start=ps,
                        page_end=pe,
                        extraction_method=method,
                    )
                )
                chunk_index += 1
                continue

            point_spans = _subspans(clause_text, POINT_PATTERN)
            if point_spans:
                clause_prefix = clause_text[:point_spans[0][0]].strip()
                for point_start, point_end, point_match in point_spans:
                    point_label = point_match.group(1).lower()
                    point_text = clause_text[point_start:point_end].strip()
                    body = f"{clause_prefix}\n{point_text}" if clause_prefix else point_text
                    absolute_point_start = absolute_clause_start + point_start
                    absolute_point_end = absolute_clause_start + point_end
                    ps, pe, method = _page_range_for_span(page_ranges, absolute_point_start, absolute_point_end)
                    content = _compose_content(
                        metadata,
                        ctx,
                        article_label,
                        f"Khoản {clause_number}",
                        point_label,
                        body,
                        "point",
                    )
                    chunks.append(
                        _make_chunk(
                            document=document,
                            metadata=metadata,
                            chunk_index=chunk_index,
                            content=content,
                            level="point",
                            ctx=ctx,
                            article_label=article_label,
                            article_number=article_number,
                            clause_number=clause_number,
                            point_label=point_label,
                            parent_id=f"{parent_article_id}_clause_{clause_number}",
                            parent_article_id=parent_article_id,
                            page_start=ps,
                            page_end=pe,
                            extraction_method=method,
                        )
                    )
                    chunk_index += 1
            else:
                for part_number, (local_start, local_end, part) in enumerate(_split_paragraphs(clause_text_for_embedding), start=1):
                    absolute_start = absolute_clause_start + local_start
                    absolute_end = absolute_clause_start + local_end
                    ps, pe, method = _page_range_for_span(page_ranges, absolute_start, absolute_end)
                    content = _compose_content(
                        metadata,
                        ctx,
                        article_label,
                        f"Khoản {clause_number}, phần {part_number}",
                        "",
                        part,
                        "clause_part",
                    )
                    chunks.append(
                        _make_chunk(
                            document=document,
                            metadata=metadata,
                            chunk_index=chunk_index,
                            content=content,
                            level="clause_part",
                            ctx=ctx,
                            article_label=article_label,
                            article_number=article_number,
                            clause_number=clause_number,
                            point_label="",
                            parent_id=parent_article_id,
                            parent_article_id=parent_article_id,
                            page_start=ps,
                            page_end=pe,
                            extraction_method=method,
                        )
                    )
                    chunk_index += 1

    return chunks


def _metadata_for_document(document: ExtractedDocument, text: str) -> tuple[dict[str, Any], str, int]:
    header_metadata, body_start = _extract_metadata_from_header(text)
    lines = text.splitlines()
    body_offset = sum(len(line) + 1 for line in lines[:body_start])
    body_text = normalize_text("\n".join(lines[body_start:]))

    metadata = _default_metadata(document)
    for key, value in header_metadata.items():
        if value:
            metadata[key] = value
    if not metadata["ten_van_ban"]:
        metadata["ten_van_ban"] = Path(document.source_path).stem.replace("_", " ")
    if not metadata["nam_ban_hanh"]:
        metadata["nam_ban_hanh"] = _infer_year(document.source_path)
    if not metadata["so_hieu_van_ban"]:
        metadata["so_hieu_van_ban"] = _infer_law_number(document.source_path)
    return metadata, body_text, body_offset


def chunk_extracted_document(document: ExtractedDocument) -> list[LegalChunk]:
    text = normalize_text(document.text)
    if not text:
        return []

    metadata, body_text, body_offset = _metadata_for_document(document, text)
    page_ranges = _page_lookup(document)

    article_chunks = _build_article_chunks(
        document=document,
        metadata=metadata,
        body_text=body_text,
        body_offset=body_offset,
        page_ranges=page_ranges,
        start_index=0,
    )
    table_chunks = _table_chunks(document, metadata, start_index=len(article_chunks))
    chunks = article_chunks + table_chunks
    _link_neighbors(chunks)

    logger.info(
        f"[CHUNKER] {metadata['ten_van_ban']}: created {len(chunks)} chunks "
        f"({len(article_chunks)} legal, {len(table_chunks)} table)"
    )
    return chunks


def chunk_legal_text(
    text: str,
    so_hieu_van_ban: str,
    ten_van_ban: str,
    loai_van_ban: str = "Luật",
    nam_ban_hanh: int = 0,
    trang_thai: str = "Hiện hành",
    co_quan_ban_hanh: str = "",
    ngay_hieu_luc: str = "",
) -> list[LegalChunk]:
    document = ExtractedDocument(
        doc_id="inline_document",
        source_path="",
        pages=[ExtractedPage(page_number=1, text=normalize_text(text), extraction_method="text")],
        metadata={
            "so_hieu_van_ban": so_hieu_van_ban,
            "ten_van_ban": ten_van_ban,
            "loai_van_ban": loai_van_ban,
            "nam_ban_hanh": nam_ban_hanh,
            "trang_thai": trang_thai,
            "co_quan_ban_hanh": co_quan_ban_hanh,
            "ngay_hieu_luc": ngay_hieu_luc,
        },
    )
    return chunk_extracted_document(document)


def chunk_text_file(file_path: str, output_jsonl: Optional[str] = None) -> list[LegalChunk]:
    document = extract_document(file_path)
    if not document:
        logger.error(f"[CHUNKER] Could not extract text from {file_path}")
        return []
    chunks = chunk_extracted_document(document)
    if output_jsonl:
        save_chunks_to_jsonl(chunks, output_jsonl)
    return chunks


def save_chunks_to_jsonl(chunks: list[LegalChunk], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.to_jsonl_line() + "\n")
    logger.info(f"[CHUNKER] Saved {len(chunks)} chunks -> {output_path}")


def load_chunks_from_jsonl(jsonl_path: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    logger.info(f"[CHUNKER] Loaded {len(chunks)} chunks from {jsonl_path}")
    return chunks
