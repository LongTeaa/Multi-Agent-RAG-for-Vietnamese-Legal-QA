"""
scripts/build_index.py
Orchestrator script để chạy toàn bộ data pipeline:
1. Extract text từ data/raw/ 
2. Chunking theo Điều/Khoản cấu trúc
3. Embedding batch
4. Upsert vào Qdrant

Usage:
    # Chạy full pipeline từ raw files
    python scripts/build_index.py --input data/raw/ --full-pipeline

    # Hoặc load từ chunks.jsonl đã tồn tại
    python scripts/build_index.py --chunks data/processed/chunks.jsonl
"""
import sys
import io
from pathlib import Path

# Fix UTF-8 encoding cho Windows PowerShell (hỗ trợ tiếng Việt)
if sys.platform == "win32":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Thêm project root vào sys.path để import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from typing import List

from tqdm import tqdm

from src.utils.logger import logger
from src.config import COLLECTION_NAME
from src.data_pipeline.extractor import extract_directory
from src.data_pipeline.chunker import chunk_text_file, save_chunks_to_jsonl
from src.data_pipeline.indexer import (
    get_qdrant_client,
    create_collection,
    upsert_chunks,
    get_collection_stats,
    IndexConfig,
)


def run_full_pipeline(
    input_dir: str,
    output_dir: str = "data/processed",
    recreate_collection: bool = False,
    use_simple_client: bool = False,
) -> int:
    """
    Chạy toàn bộ pipeline: Extract → Chunk → Embed → Upsert Qdrant.

    Args:
        input_dir: Thư mục chứa PDF/HTML/TXT files
        output_dir: Thư mục lưu chunks.jsonl
        recreate_collection: Xóa collection cũ trước tạo mới
        use_simple_client: Skip Qdrant client, dùng simple client ngay (tránh encoding issue)

    Returns:
        Số chunks được upsert thành công
    """
    logger.info("=" * 80)
    logger.info("STARTING FULL DATA PIPELINE")
    logger.info("=" * 80)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_output = output_path / "chunks.jsonl"

    # ─── STEP 1: Extract files ───
    logger.info("\n[STEP 1] EXTRACTING files from raw directory...")
    extracted_texts = extract_directory(str(input_path))
    if not extracted_texts:
        logger.error("No files extracted. Exiting.")
        return 0

    logger.info(f"✓ Extracted {len(extracted_texts)} files")

    # ─── STEP 2: Chunking ───
    logger.info("\n[STEP 2] CHUNKING documents...")
    all_chunks = []

    for file_path, text in tqdm(extracted_texts.items(), desc="Chunking files"):
        chunks = chunk_text_file(file_path)
        all_chunks.extend(chunks)
        logger.debug(f"  {Path(file_path).name}: {len(chunks)} chunks")

    logger.info(f"✓ Created {len(all_chunks)} chunks total")

    # ─── Save chunks to JSONL ───
    logger.info(f"\n[STEP 2B] Saving chunks to {jsonl_output}...")
    chunks_dicts = [c.to_dict() for c in all_chunks]
    save_chunks_to_jsonl(all_chunks, str(jsonl_output))
    logger.info(f"✓ Saved {len(chunks_dicts)} chunks to JSONL")

    # ─── STEP 3: Index to Qdrant ───
    logger.info("\n[STEP 3] INDEXING to Qdrant...")
    try:
        if use_simple_client:
            # Skip qdrant-client entirely, use simple client to avoid encoding issues
            logger.info("[STEP 3] Using simple client (skip qdrant-client to avoid encoding issues)...")
            from src.data_pipeline.qdrant_simple import upsert_chunks_simple
            from src.utils.embedding import embed_texts
            
            logger.info("[STEP 3] Embedding chunks...")
            texts = [c["content"] for c in chunks_dicts]
            embeddings = embed_texts(texts)

            for chunk, emb in zip(chunks_dicts, embeddings):
                chunk["vector"] = emb

            upserted = upsert_chunks_simple(
                chunks_dicts,
                collection_name=COLLECTION_NAME,
                recreate=recreate_collection,
            )
            logger.info(f"✓ Upserted {upserted} chunks (simple client)")
        else:
            # Try using regular qdrant-client first
            try:
                client = get_qdrant_client()
                config = IndexConfig(recreate=recreate_collection)
                create_collection(client, config)
                upserted = upsert_chunks(client, chunks_dicts, config)
                logger.info(f"✓ Upserted {upserted} chunks (qdrant-client)")
            except Exception as e:
                # Fallback to simple client using requests
                logger.warning(f"[STEP 3] qdrant-client failed, trying simple client: {str(e)[:80]}")
                from src.data_pipeline.qdrant_simple import upsert_chunks_simple
                from src.utils.embedding import embed_texts

                # Add embeddings to chunks first
                logger.info("[STEP 3] Embedding chunks for simple client...")
                texts = [c["content"] for c in chunks_dicts]
                embeddings = embed_texts(texts)

                for chunk, emb in zip(chunks_dicts, embeddings):
                    chunk["vector"] = emb

                upserted = upsert_chunks_simple(
                    chunks_dicts,
                    collection_name=COLLECTION_NAME,
                    recreate=recreate_collection,
                )
                logger.info(f"✓ Upserted {upserted} chunks (simple client)")

        # Print collection stats
        if upserted > 0:
            try:
                stats = get_collection_stats(client)
                logger.info(f"Collection stats: {stats}")
            except:
                logger.info(f"✓ Indexed {upserted} chunks to Qdrant")

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        return upserted

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 0


def load_and_index_chunks(
    jsonl_path: str,
    recreate_collection: bool = False,
) -> int:
    """
    Load chunks từ file JSONL đã tồn tại và upsert vào Qdrant.

    Args:
        jsonl_path: Đường dẫn file chunks.jsonl
        recreate_collection: Xóa collection cũ

    Returns:
        Số chunks được upsert
    """
    logger.info(f"Loading chunks from {jsonl_path}...")

    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info(f"✓ Loaded {len(chunks)} chunks")

    # Upsert to Qdrant
    try:
        client = get_qdrant_client()
        config = IndexConfig(recreate=recreate_collection)
        create_collection(client, config)

        upserted = upsert_chunks(client, chunks, config)
        logger.info(f"✓ Upserted {upserted} chunks to Qdrant")

        stats = get_collection_stats(client)
        logger.info(f"Collection stats: {stats}")

        return upserted

    except Exception as e:
        logger.error(f"Failed to index: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Build data pipeline: Extract → Chunk → Index to Qdrant"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/",
        help="Input directory chứa PDF/HTML/TXT files (for --full-pipeline)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/",
        help="Output directory lưu chunks.jsonl",
    )

    parser.add_argument(
        "--chunks",
        type=str,
        help="Load từ chunks.jsonl và index (bỏ qua Extract/Chunk steps)",
    )

    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Chạy full pipeline từ raw files (Extract → Chunk → Index)",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa collection cũ trước khi tạo mới",
    )

    parser.add_argument(
        "--use-simple-client",
        action="store_true",
        help="Skip Qdrant client, dùng simple client để tránh encoding issues trên Windows",
    )

    args = parser.parse_args()
    
    # Windows + encoding issues → Mặc định sử dụng simple client
    import platform
    if platform.system() == "Windows" and not args.chunks:
        args.use_simple_client = True
        logger.info("[AUTO-CONFIG] Windows detected: using simple client to avoid encoding issues")

    # Validate args
    if args.chunks:
        if not Path(args.chunks).exists():
            logger.error(f"File not found: {args.chunks}")
            return 1
        load_and_index_chunks(args.chunks, recreate_collection=args.recreate)

    elif args.full_pipeline or not args.chunks:
        if not Path(args.input).exists():
            logger.error(f"Input directory not found: {args.input}")
            return 1
        run_full_pipeline(args.input, args.output, recreate_collection=args.recreate, use_simple_client=args.use_simple_client)

    return 0


if __name__ == "__main__":
    exit(main())
