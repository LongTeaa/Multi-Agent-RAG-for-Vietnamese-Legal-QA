"""
src/data_pipeline/chunker.py
Chunking văn bản pháp lý theo cấu trúc Điều/Khoản.
Ưu tiên giữ nguyên đơn vị logic thay vì cắt theo số ký tự.
"""
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from src.utils.logger import logger


# ─────────────────────────────────────────────
# Patterns nhận diện cấu trúc văn bản pháp luật
# ─────────────────────────────────────────────
DIEU_PATTERN = re.compile(
    r"(?m)^(Điều\s+\d+[\.\:].*?)(?=\nĐiều\s+\d+|\Z)",
    re.DOTALL,
)
KHOAN_PATTERN = re.compile(
    r"(\d+\.\s.+?)(?=\n\d+\.|\Z)",
    re.DOTALL,
)
CHUONG_PATTERN = re.compile(
    r"(Chương\s+[IVXLCDM\d]+[\.\:]?\s*.+?)(?=\nChương|\Z)",
    re.DOTALL,
)

MAX_CHUNK_CHARS = 2000   # ~500–800 tokens, an toàn cho context window


@dataclass
class LegalChunk:
    """Một chunk văn bản pháp lý đầy đủ metadata."""
    content: str
    so_hieu_van_ban: str        # VD: "45/2019/QH14"
    ten_van_ban: str            # VD: "Bộ luật Lao động 2019"
    loai_van_ban: str           # "Luật" | "Nghị định" | "Thông tư" | ...
    chuong: str                 # VD: "Chương VII: Thời giờ làm việc, thời giờ nghỉ ngơi"
    dieu: str                   # VD: "Điều 105: Thời giờ làm việc bình thường"
    khoang: str                 # VD: "Khoản 1" (rỗng nếu cắt tại Điều)
    nam_ban_hanh: int           # VD: 2019
    trang_thai: str             # "Hiện hành" | "Hết hiệu lực" | "Sửa đổi"
    co_quan_ban_hanh: str       # VD: "Quốc hội"
    ngay_hieu_luc: str          # VD: "2021-01-01"
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def chunk_legal_text(
    text: str,
    so_hieu_van_ban: str,
    ten_van_ban: str,
    loai_van_ban: str = "Luật",
    nam_ban_hanh: int = 0,
    trang_thai: str = "Hiện hành",
    co_quan_ban_hanh: str = "Quốc hội",
    ngay_hieu_luc: str = "",
) -> List[LegalChunk]:
    """
    Chunking thông minh văn bản pháp lý.

    Thuật toán:
    1. Tách theo ranh giới Chương để lấy context chuong.
    2. Trong mỗi Chương, tách theo ranh giới Điều.
    3. Nếu Điều > MAX_CHUNK_CHARS → cắt tiếp theo Khoản.
    4. Gắn metadata đầy đủ cho mỗi chunk.

    Args:
        text: Toàn bộ text văn bản pháp luật
        so_hieu_van_ban: Số hiệu văn bản (VD: "45/2019/QH14")
        ten_van_ban: Tên đầy đủ của văn bản
        loai_van_ban: Loại văn bản
        nam_ban_hanh: Năm ban hành
        trang_thai: Trạng thái hiệu lực
        co_quan_ban_hanh: Cơ quan ban hành
        ngay_hieu_luc: Ngày có hiệu lực (YYYY-MM-DD)

    Returns:
        List[LegalChunk]
    """
    chunks: List[LegalChunk] = []
    chunk_index = 0
    current_chuong = ""

    # Tách text theo Chương, lấy context Chương
    sections = _split_by_chuong(text)

    for chuong_label, section_text in sections:
        if chuong_label:
            current_chuong = chuong_label

        # Tách từng Điều trong Chương
        dieu_matches = DIEU_PATTERN.findall(section_text)

        if not dieu_matches:
            # Không có Điều → chunk nguyên section nếu không rỗng
            content = section_text.strip()
            if content and len(content) > 50:
                chunks.append(LegalChunk(
                    content=content[:MAX_CHUNK_CHARS],
                    so_hieu_van_ban=so_hieu_van_ban,
                    ten_van_ban=ten_van_ban,
                    loai_van_ban=loai_van_ban,
                    chuong=current_chuong,
                    dieu="",
                    khoang="",
                    nam_ban_hanh=nam_ban_hanh,
                    trang_thai=trang_thai,
                    co_quan_ban_hanh=co_quan_ban_hanh,
                    ngay_hieu_luc=ngay_hieu_luc,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
            continue

        for dieu_text in dieu_matches:
            dieu_text = dieu_text.strip()
            if not dieu_text:
                continue

            # Lấy nhãn Điều từ dòng đầu
            first_line = dieu_text.split("\n")[0].strip()
            dieu_label = first_line[:100]

            if len(dieu_text) <= MAX_CHUNK_CHARS:
                # Điều đủ nhỏ → 1 chunk
                chunks.append(LegalChunk(
                    content=dieu_text,
                    so_hieu_van_ban=so_hieu_van_ban,
                    ten_van_ban=ten_van_ban,
                    loai_van_ban=loai_van_ban,
                    chuong=current_chuong,
                    dieu=dieu_label,
                    khoang="",
                    nam_ban_hanh=nam_ban_hanh,
                    trang_thai=trang_thai,
                    co_quan_ban_hanh=co_quan_ban_hanh,
                    ngay_hieu_luc=ngay_hieu_luc,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
            else:
                # Điều quá dài → cắt theo Khoản
                khoan_matches = KHOAN_PATTERN.findall(dieu_text)
                if khoan_matches:
                    for i, khoan_text in enumerate(khoan_matches, 1):
                        khoan_text = khoan_text.strip()
                        if not khoan_text or len(khoan_text) < 30:
                            continue
                        chunks.append(LegalChunk(
                            content=f"{dieu_label}\n{khoan_text}",
                            so_hieu_van_ban=so_hieu_van_ban,
                            ten_van_ban=ten_van_ban,
                            loai_van_ban=loai_van_ban,
                            chuong=current_chuong,
                            dieu=dieu_label,
                            khoang=f"Khoản {i}",
                            nam_ban_hanh=nam_ban_hanh,
                            trang_thai=trang_thai,
                            co_quan_ban_hanh=co_quan_ban_hanh,
                            ngay_hieu_luc=ngay_hieu_luc,
                            chunk_index=chunk_index,
                        ))
                        chunk_index += 1
                else:
                    # Không có Khoản → cắt hard theo MAX_CHUNK_CHARS
                    for start in range(0, len(dieu_text), MAX_CHUNK_CHARS):
                        part = dieu_text[start:start + MAX_CHUNK_CHARS]
                        chunks.append(LegalChunk(
                            content=part,
                            so_hieu_van_ban=so_hieu_van_ban,
                            ten_van_ban=ten_van_ban,
                            loai_van_ban=loai_van_ban,
                            chuong=current_chuong,
                            dieu=dieu_label,
                            khoang="",
                            nam_ban_hanh=nam_ban_hanh,
                            trang_thai=trang_thai,
                            co_quan_ban_hanh=co_quan_ban_hanh,
                            ngay_hieu_luc=ngay_hieu_luc,
                            chunk_index=chunk_index,
                        ))
                        chunk_index += 1

    logger.info(
        f"[CHUNKER] {ten_van_ban}: {len(chunks)} chunks tạo ra"
    )
    return chunks


def _split_by_chuong(text: str) -> List[tuple[str, str]]:
    """
    Tách text thành các section theo Chương.

    Returns:
        List of (chuong_label, section_text)
    """
    result = []
    chuong_positions = [m.start() for m in re.finditer(
        r"(?m)^Chương\s+[IVXLCDM\d]+", text
    )]

    if not chuong_positions:
        return [("", text)]

    # Nội dung trước Chương đầu tiên
    if chuong_positions[0] > 0:
        result.append(("", text[:chuong_positions[0]]))

    for i, pos in enumerate(chuong_positions):
        end = chuong_positions[i + 1] if i + 1 < len(chuong_positions) else len(text)
        section = text[pos:end]
        first_line = section.split("\n")[0].strip()
        result.append((first_line[:120], section))

    return result


def _extract_metadata_from_header(text: str) -> dict:
    """
    Parse metadata từ header của file (các dòng bắt đầu với SO_HIEU_VAN_BAN:, TEN_VAN_BAN:, v.v.).

    Args:
        text: Toàn bộ text file

    Returns:
        Dict metadata: {so_hieu_van_ban, ten_van_ban, nam_ban_hanh, ...}
    """
    metadata = {
        "so_hieu_van_ban": "",
        "ten_van_ban": "",
        "nam_ban_hanh": 0,
        "loai_van_ban": "Luật",
        "trang_thai": "Hiện hành",
        "co_quan_ban_hanh": "Quốc hội",
        "ngay_hieu_luc": "",
    }

    lines = text.split("\n")
    body_start = 0

    for i, line in enumerate(lines):
        line_upper = line.upper()
        if ":" not in line:
            body_start = i
            break

        if line.startswith("SO_HIEU_VAN_BAN:"):
            metadata["so_hieu_van_ban"] = line.replace("SO_HIEU_VAN_BAN:", "").strip()
        elif line.startswith("TEN_VAN_BAN:"):
            metadata["ten_van_ban"] = line.replace("TEN_VAN_BAN:", "").strip()
        elif line.startswith("NAM_BAN_HANH:"):
            try:
                metadata["nam_ban_hanh"] = int(line.replace("NAM_BAN_HANH:", "").strip())
            except ValueError:
                pass
        elif line.startswith("LOAI_VAN_BAN:"):
            metadata["loai_van_ban"] = line.replace("LOAI_VAN_BAN:", "").strip()
        elif line.startswith("TRANG_THAI:"):
            metadata["trang_thai"] = line.replace("TRANG_THAI:", "").strip()
        elif line.startswith("BAN_HANH:") or line.startswith("CO_QUAN_BAN_HANH:"):
            metadata["co_quan_ban_hanh"] = line.split(":", 1)[1].strip()
        elif line.startswith("NGAY_HIEU_LUC:"):
            metadata["ngay_hieu_luc"] = line.replace("NGAY_HIEU_LUC:", "").strip()

    return metadata, body_start


def chunk_text_file(
    file_path: str,
    output_jsonl: Optional[str] = None,
) -> List[LegalChunk]:
    """
    Chunk một file pháp lý (TXT, PDF, v.v.) theo cấu trúc Điều/Khoản.

    Args:
        file_path: Đường dẫn file
        output_jsonl: (Optional) Lưu chunks ra file JSONL

    Returns:
        List[LegalChunk]
    """
    from src.data_pipeline.extractor import extract_file

    text = extract_file(file_path)
    if not text:
        logger.error(f"[CHUNKER] Không thể extract text từ {file_path}")
        return []

    # Parse metadata từ header
    metadata, body_start = _extract_metadata_from_header(text)

    # Lấy body text (bỏ header)
    body_lines = text.split("\n")[body_start:]
    body_text = "\n".join(body_lines).strip()

    # Chunking
    chunks = chunk_legal_text(
        text=body_text,
        so_hieu_van_ban=metadata["so_hieu_van_ban"],
        ten_van_ban=metadata["ten_van_ban"],
        loai_van_ban=metadata["loai_van_ban"],
        nam_ban_hanh=metadata["nam_ban_hanh"],
        trang_thai=metadata["trang_thai"],
        co_quan_ban_hanh=metadata["co_quan_ban_hanh"],
        ngay_hieu_luc=metadata["ngay_hieu_luc"],
    )

    # Lưu nếu có path output
    if output_jsonl:
        save_chunks_to_jsonl(chunks, output_jsonl)

    return chunks


def save_chunks_to_jsonl(chunks: List[LegalChunk], output_path: str) -> None:
    """Lưu danh sách chunks ra file JSONL (mỗi dòng 1 JSON object)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.to_jsonl_line() + "\n")
    logger.info(f"[CHUNKER] Saved {len(chunks)} chunks → {output_path}")


def load_chunks_from_jsonl(jsonl_path: str) -> List[dict]:
    """Load chunks từ file JSONL."""
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info(f"[CHUNKER] Loaded {len(chunks)} chunks from {jsonl_path}")
    return chunks
