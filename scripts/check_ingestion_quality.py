"""
Check OCR/extraction quality before improving Phase 2 chunking.

The report is intentionally heuristic: it flags documents/chunks that deserve
manual review before we build richer legal parsers on top of them.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SUSPICIOUS_PATTERNS = {
    "replacement_char": "\ufffd",
    "mojibake_marker": r"(?:Ã.|Ä.|Æ.|áº|á»|â€)",
    "ocr_digit_in_word": r"[A-Za-zÀ-ỹ]{2,}\d+[A-Za-zÀ-ỹ]*|[A-Za-zÀ-ỹ]*\d+[A-Za-zÀ-ỹ]{2,}",
    "broken_legal_heading": r"(?:Điẻu|Di[eê]u|Khoàn|Chưong|Chuong J|Quc\d|C0NG|H0A)",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def suspicious_counts(text: str) -> dict[str, int]:
    return {
        name: len(re.findall(pattern, text, flags=re.IGNORECASE))
        for name, pattern in SUSPICIOUS_PATTERNS.items()
    }


def safe_ratio(count: int, total: int) -> float:
    return count / max(total, 1)


def summarize(chunks: list[dict[str, Any]], registry: list[dict[str, Any]]) -> dict[str, Any]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.get("doc_id", "")].append(chunk)

    registry_by_doc = {row.get("doc_id", ""): row for row in registry}
    summary: dict[str, Any] = {
        "registry_count": len(registry),
        "chunk_count": len(chunks),
        "documents": {},
        "global": {},
    }

    methods = Counter(chunk.get("extraction_method", "") for chunk in chunks)
    levels = Counter(chunk.get("level", "") for chunk in chunks)
    missing = {
        "missing_doc_id": sum(1 for chunk in chunks if not chunk.get("doc_id")),
        "missing_page": sum(1 for chunk in chunks if not chunk.get("page_start")),
        "missing_source_path": sum(1 for chunk in chunks if not chunk.get("source_path")),
        "missing_source_url": sum(1 for chunk in chunks if not chunk.get("source_url")),
        "missing_article_label": sum(1 for chunk in chunks if chunk.get("level", "").startswith(("article", "clause", "point")) and not chunk.get("dieu")),
        "missing_parent_id": sum(1 for chunk in chunks if chunk.get("level") in {"clause", "clause_part", "point", "table"} and not chunk.get("parent_id")),
        "missing_table_id": sum(1 for chunk in chunks if chunk.get("level") == "table" and not chunk.get("table_id")),
    }

    lengths = [len(chunk.get("content", "")) for chunk in chunks]
    summary["global"] = {
        "methods": dict(methods),
        "levels": dict(levels),
        "missing": missing,
        "min_chunk_chars": min(lengths) if lengths else 0,
        "avg_chunk_chars": round(mean(lengths), 1) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
    }

    for doc_id, doc_chunks in sorted(by_doc.items()):
        text = "\n".join(chunk.get("content", "") for chunk in doc_chunks)
        chars = len(text)
        counts = suspicious_counts(text)
        methods_doc = Counter(chunk.get("extraction_method", "") for chunk in doc_chunks)
        pages = [
            page
            for chunk in doc_chunks
            for page in (chunk.get("page_start"), chunk.get("page_end"))
            if isinstance(page, int) and page > 0
        ]
        short_chunks = [chunk for chunk in doc_chunks if len(chunk.get("content", "")) < 80]
        long_chunks = [chunk for chunk in doc_chunks if len(chunk.get("content", "")) > 2200]
        article_chunks = [chunk for chunk in doc_chunks if chunk.get("level", "").startswith("article")]
        clause_chunks = [chunk for chunk in doc_chunks if chunk.get("level", "").startswith("clause")]
        point_chunks = [chunk for chunk in doc_chunks if chunk.get("level") == "point"]
        table_chunks = [chunk for chunk in doc_chunks if chunk.get("level") == "table"]
        missing_articles = [chunk for chunk in article_chunks if not chunk.get("dieu")]
        missing_parent = [
            chunk for chunk in doc_chunks
            if chunk.get("level") in {"clause", "clause_part", "point", "table"} and not chunk.get("parent_id")
        ]

        summary["documents"][doc_id] = {
            "source_path": registry_by_doc.get(doc_id, {}).get("source_path", doc_chunks[0].get("source_path", "")),
            "source_url": registry_by_doc.get(doc_id, {}).get("source_url", ""),
            "chunks": len(doc_chunks),
            "chars": chars,
            "pages_min": min(pages) if pages else 0,
            "pages_max": max(pages) if pages else 0,
            "methods": dict(methods_doc),
            "suspicious_counts": counts,
            "suspicious_per_10k_chars": {
                key: round(safe_ratio(value, chars) * 10000, 2)
                for key, value in counts.items()
            },
            "short_chunks": len(short_chunks),
            "long_chunks": len(long_chunks),
            "article_chunks": len(article_chunks),
            "clause_chunks": len(clause_chunks),
            "point_chunks": len(point_chunks),
            "table_chunks": len(table_chunks),
            "missing_article_labels": len(missing_articles),
            "missing_parent_ids": len(missing_parent),
            "sample_flagged_chunks": flagged_samples(doc_chunks),
        }

    return summary


def flagged_samples(chunks: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    for chunk in chunks:
        text = chunk.get("content", "")
        counts = suspicious_counts(text)
        score = (
            counts["replacement_char"] * 20
            + counts["mojibake_marker"] * 5
            + counts["broken_legal_heading"] * 8
            + counts["ocr_digit_in_word"] * 3
        )
        if score:
            scored.append((score, chunk))

    samples: list[dict[str, Any]] = []
    for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]:
        text = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
        samples.append(
            {
                "score": score,
                "chunk_id": chunk.get("chunk_id", ""),
                "page_start": chunk.get("page_start", 0),
                "page_end": chunk.get("page_end", 0),
                "extraction_method": chunk.get("extraction_method", ""),
                "preview": text[:280],
            }
        )
    return samples


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Ingestion Quality Report")
    lines.append("")
    lines.append("## Global")
    lines.append(f"- Registry documents: {summary['registry_count']}")
    lines.append(f"- Chunks: {summary['chunk_count']}")
    lines.append(f"- Extraction methods: `{summary['global']['methods']}`")
    lines.append(f"- Levels: `{summary['global']['levels']}`")
    lines.append(f"- Missing metadata: `{summary['global']['missing']}`")
    lines.append(
        "- Chunk chars: "
        f"min={summary['global']['min_chunk_chars']}, "
        f"avg={summary['global']['avg_chunk_chars']}, "
        f"max={summary['global']['max_chunk_chars']}"
    )
    lines.append("")

    lines.append("## By Document")
    for doc_id, doc in summary["documents"].items():
        lines.append(f"### {doc_id}")
        lines.append(f"- Source: `{doc['source_path']}`")
        lines.append(f"- Source URL: `{doc['source_url'] or '(missing)'}`")
        lines.append(f"- Chunks/pages/chars: {doc['chunks']} chunks, pages {doc['pages_min']}-{doc['pages_max']}, {doc['chars']} chars")
        lines.append(f"- Methods: `{doc['methods']}`")
        lines.append(f"- Suspicious per 10k chars: `{doc['suspicious_per_10k_chars']}`")
        lines.append(f"- Short/long chunks: {doc['short_chunks']} short, {doc['long_chunks']} long")
        lines.append(
            "- Structure chunks: "
            f"article={doc['article_chunks']}, clause={doc['clause_chunks']}, "
            f"point={doc['point_chunks']}, table={doc['table_chunks']}"
        )
        lines.append(f"- Missing article labels / parent ids: {doc['missing_article_labels']} / {doc['missing_parent_ids']}")
        if doc["sample_flagged_chunks"]:
            lines.append("- Flagged samples:")
            for sample in doc["sample_flagged_chunks"]:
                lines.append(
                    f"  - `{sample['chunk_id']}` p.{sample['page_start']}-{sample['page_end']} "
                    f"{sample['extraction_method']} score={sample['score']}: {sample['preview']}"
                )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check processed ingestion quality")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--registry", default="data/processed/document_registry.jsonl")
    parser.add_argument("--output", default="data/processed/ingestion_quality_report.md")
    args = parser.parse_args()

    chunks = load_jsonl(Path(args.chunks))
    registry = load_jsonl(Path(args.registry))
    summary = summarize(chunks, registry)
    output_path = Path(args.output)
    write_markdown(summary, output_path)

    print(json.dumps(summary["global"], ensure_ascii=False, indent=2))
    print(f"Report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
