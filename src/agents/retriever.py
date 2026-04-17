"""
Step 8: Retriever Agent - Hybrid Search (BM25 + Vector)

Chức năng: Truy xuất tài liệu pháp lý từ Qdrant sử dụng hybrid search
kết hợp BM25 (keyword-based) + dense vector search (semantic).
"""

from typing import Dict, Any, List
from qdrant_client import models
from src.config import TOP_K_DOCUMENTS
from src.utils.embedding import embed_query, generate_sparse_vector
from src.utils.logger import logger
from src.data_pipeline.indexer import get_qdrant_client
from src.graph.state import GraphState


def retriever_node(state: GraphState) -> Dict[str, Any]:
    """
    Retriever node: Thực hiện hybrid search trên Qdrant collection.
    
    Input từ state:
    - question: str (câu hỏi tiếng Việt)
    - top_k: int (mặc định 5 từ config)
    
    Output cập nhật state:
    - documents: List[Dict] with keys: content, metadata, score
    """
    try:
        question = state.get("question", "")
        top_k = TOP_K_DOCUMENTS
        
        if not question:
            logger.error("No question provided to retriever")
            return {
                "documents": [],
                "error": None
            }
        
        logger.info(f"Retrieving documents for question: {question[:100]}...")
        
        # 1. Embed query với prefix "query: " (e5 model convention)
        query_vector = embed_query(question)
        logger.debug(f"Query vector shape: {len(query_vector)}")
        
        # 2. Kết nối Qdrant client
        client = get_qdrant_client()
        
        # 3. Tạo sparse vector cho câu hỏi
        sparse_vector = generate_sparse_vector(question)

        # 4. Thực hiện hybrid search với RRF Fusion
        # qdrant-client >= 1.10 mang lại universal query API
        response = client.query_points(
            collection_name="vietnamese_legal_chunks",
            prefetch=[
                # Nhánh 1: Dense Search (Ngữ nghĩa)
                models.Prefetch(
                    query=query_vector,
                    using="default",
                    limit=top_k * 2,  # Lấy nhiều hơn để fusion
                ),
                # Nhánh 2: Sparse Search (Từ khóa/BM25)
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"]
                    ),
                    using="bm25",
                    limit=top_k * 2,
                ),
            ],
            # Kết hợp kết quả dùng Reciprocal Rank Fusion
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )
        search_results = response.points  # QueryResponse trả về .points thay vì list trực tiếp
        
        logger.info(f"Retrieved {len(search_results)} documents from Qdrant")
        
        # 4. Format output: [{"content": str, "metadata": dict, "score": float}]
        documents = []
        for result in search_results:
            try:
                doc = {
                    "content": result.payload.get("chunk_text", ""),
                    "metadata": {
                        "so_hieu_van_ban": result.payload.get("so_hieu_van_ban", ""),
                        "ten_van_ban": result.payload.get("ten_van_ban", ""),
                        "loai_van_ban": result.payload.get("loai_van_ban", ""),
                        "chuong": result.payload.get("chuong", ""),
                        "dieu": result.payload.get("dieu", ""),
                        "khoang": result.payload.get("khoang", ""),
                        "nam_ban_hanh": result.payload.get("nam_ban_hanh", ""),
                        "trang_thai": result.payload.get("trang_thai", ""),
                        "co_quan_ban_hanh": result.payload.get("co_quan_ban_hanh", ""),
                        "ngay_hieu_luc": result.payload.get("ngay_hieu_luc", ""),
                    },
                    "score": result.score,
                }
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Error formatting document result: {e}")
                continue
        
        logger.info(f"Successfully formatted {len(documents)} documents")
        return {
            "documents": documents,
            "error": None
        }
        
    except Exception as e:
        logger.error("Error in retriever_node: {}", e, exc_info=True)
        return {
            "documents": [],
            "error": f"Retrieval failed: {str(e)}"
        }
