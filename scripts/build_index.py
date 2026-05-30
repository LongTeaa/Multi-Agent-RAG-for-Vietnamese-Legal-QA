"""
Build the RAG index from raw legal documents.

Pipeline:
1. Register and extract files from data/raw
2. Chunk documents with page/source metadata
3. Embed chunks
4. Upsert into Qdrant
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.config import COLLECTION_NAME
from src.data_pipeline.chunker import chunk_extracted_document, save_chunks_to_jsonl
from src.data_pipeline.extractor import extract_documents
from src.data_pipeline.indexer import (
    IndexConfig,
    create_collection,
    get_collection_stats,
    get_qdrant_client,
    upsert_chunks,
)
from src.utils.logger import logger


def run_full_pipeline(
    input_dir: str,
    output_dir: str = "data/processed",
    recreate_collection: bool = False,
    use_simple_client: bool = False,
    skip_index: bool = False,
) -> int:
    """Run Extract -> Chunk -> Embed -> Upsert."""
    logger.info("=" * 80)
    logger.info("STARTING FULL DATA PIPELINE")
    logger.info("=" * 80)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_output = output_path / "chunks.jsonl"

    logger.info("\n[STEP 1] Extracting source files and writing document registry...")
    documents = extract_documents(str(input_path), registry_output_dir=output_path)
    if not documents:
        logger.error("No documents extracted. Exiting.")
        return 0
    logger.info(f"Extracted {len(documents)} documents")

    logger.info("\n[STEP 2] Chunking documents with page/source metadata...")
    all_chunks = []
    for document in tqdm(documents, desc="Chunking documents"):
        chunks = chunk_extracted_document(document)
        all_chunks.extend(chunks)
        logger.debug(f"  {Path(document.source_path).name}: {len(chunks)} chunks")

    if not all_chunks:
        logger.error("No chunks created. Exiting.")
        return 0

    logger.info(f"Created {len(all_chunks)} chunks total")
    logger.info(f"\n[STEP 2B] Saving chunks to {jsonl_output}...")
    chunks_dicts = [chunk.to_dict() for chunk in all_chunks]
    save_chunks_to_jsonl(all_chunks, str(jsonl_output))
    logger.info(f"Saved {len(chunks_dicts)} chunks to JSONL")

    if skip_index:
        logger.info("Skipping Qdrant indexing by request")
        return len(chunks_dicts)

    logger.info("\n[STEP 3] Indexing to Qdrant...")
    client = None
    try:
        if use_simple_client:
            from src.data_pipeline.qdrant_simple import upsert_chunks_simple
            from src.utils.embedding import embed_texts

            logger.info("[STEP 3] Using simple Qdrant client")
            texts = [chunk["content"] for chunk in chunks_dicts]
            embeddings = embed_texts(texts)
            for chunk, embedding in zip(chunks_dicts, embeddings):
                chunk["vector"] = embedding

            upserted = upsert_chunks_simple(
                chunks_dicts,
                collection_name=COLLECTION_NAME,
                recreate=recreate_collection,
            )
        else:
            try:
                client = get_qdrant_client()
                config = IndexConfig(recreate=recreate_collection)
                create_collection(client, config)
                upserted = upsert_chunks(client, chunks_dicts, config)
            except Exception as exc:
                logger.warning(f"[STEP 3] qdrant-client failed, trying simple client: {str(exc)[:120]}")
                from src.data_pipeline.qdrant_simple import upsert_chunks_simple
                from src.utils.embedding import embed_texts

                texts = [chunk["content"] for chunk in chunks_dicts]
                embeddings = embed_texts(texts)
                for chunk, embedding in zip(chunks_dicts, embeddings):
                    chunk["vector"] = embedding

                upserted = upsert_chunks_simple(
                    chunks_dicts,
                    collection_name=COLLECTION_NAME,
                    recreate=recreate_collection,
                )

        if upserted > 0 and client is not None:
            stats = get_collection_stats(client)
            logger.info(f"Collection stats: {stats}")
        elif upserted > 0:
            logger.info(f"Indexed {upserted} chunks to Qdrant")

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        return upserted
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        return 0


def load_and_index_chunks(
    jsonl_path: str,
    recreate_collection: bool = False,
    use_simple_client: bool = False,
) -> int:
    """Load existing chunks.jsonl and upsert it into Qdrant."""
    logger.info(f"Loading chunks from {jsonl_path}...")
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} chunks")

    try:
        if use_simple_client:
            from src.data_pipeline.qdrant_simple import upsert_chunks_simple
            from src.utils.embedding import embed_texts

            texts = [chunk["content"] for chunk in chunks]
            embeddings = embed_texts(texts)
            for chunk, embedding in zip(chunks, embeddings):
                chunk["vector"] = embedding
            upserted = upsert_chunks_simple(
                chunks,
                collection_name=COLLECTION_NAME,
                recreate=recreate_collection,
            )
            logger.info(f"Upserted {upserted} chunks to Qdrant")
            return upserted

        client = get_qdrant_client()
        config = IndexConfig(recreate=recreate_collection)
        create_collection(client, config)
        upserted = upsert_chunks(client, chunks, config)
        logger.info(f"Upserted {upserted} chunks to Qdrant")
        logger.info(f"Collection stats: {get_collection_stats(client)}")
        return upserted
    except Exception as exc:
        logger.error(f"Failed to index: {exc}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data pipeline: Extract -> Chunk -> Index to Qdrant")
    parser.add_argument("--input", type=str, default="data/raw/", help="Input directory for raw legal files")
    parser.add_argument("--output", type=str, default="data/processed/", help="Output directory for processed JSONL")
    parser.add_argument("--chunks", type=str, help="Load existing chunks.jsonl and index it")
    parser.add_argument("--full-pipeline", action="store_true", help="Run full pipeline from raw files")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate target collection")
    parser.add_argument("--use-simple-client", action="store_true", help="Use requests-based Qdrant client")
    parser.add_argument("--skip-index", action="store_true", help="Only extract and chunk; do not upsert to Qdrant")
    args = parser.parse_args()

    import platform

    if platform.system() == "Windows" and not args.chunks:
        args.use_simple_client = True
        logger.info("[AUTO-CONFIG] Windows detected: using simple client")

    if args.chunks:
        if not Path(args.chunks).exists():
            logger.error(f"File not found: {args.chunks}")
            return 1
        load_and_index_chunks(
            args.chunks,
            recreate_collection=args.recreate,
            use_simple_client=args.use_simple_client,
        )
    else:
        if not Path(args.input).exists():
            logger.error(f"Input directory not found: {args.input}")
            return 1
        run_full_pipeline(
            args.input,
            args.output,
            recreate_collection=args.recreate,
            use_simple_client=args.use_simple_client,
            skip_index=args.skip_index,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
