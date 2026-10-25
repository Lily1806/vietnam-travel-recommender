"""
Retriever - Tìm kiếm tài liệu liên quan từ vector store cho RAG.
"""

from src.retrieval.embedder import TextEmbedder
from src.retrieval.vector_store import FAISSVectorStore, SearchResult
from src.utils.config import RAGConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRetriever:
    """
    Retrieval component của RAG pipeline.

    Nhận câu hỏi → tạo embedding → tìm tài liệu liên quan → trả về context.
    """

    def __init__(self, embedder: TextEmbedder, vector_store: FAISSVectorStore, config: RAGConfig):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = config.top_k_retrieval

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """
        Tìm tài liệu liên quan nhất với câu truy vấn.

        Args:
            query: Câu hỏi hoặc từ khóa tìm kiếm
            top_k: Số lượng tài liệu trả về (mặc định từ config)

        Returns:
            Danh sách SearchResult sắp xếp theo độ liên quan
        """
        k = top_k or self.top_k
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=k)
        logger.debug("Query: '%s' → Tìm được %d tài liệu liên quan", query, len(results))
        return results

    def retrieve_as_context(self, query: str, top_k: int | None = None) -> str:
        """
        Tìm và gộp tài liệu thành một chuỗi context cho LLM.

        Args:
            query: Câu hỏi của người dùng
            top_k: Số lượng tài liệu

        Returns:
            Context string, mỗi tài liệu được phân cách bởi dấu ---
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin liên quan."

        context_parts = []
        for i, result in enumerate(results, start=1):
            context_parts.append(
                f"[Nguồn {i}: {result.source}]\n{result.content}"
            )
        return "\n\n---\n\n".join(context_parts)
