"""
Embedder - Chuyển đổi văn bản thành vector embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import RAGConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextEmbedder:
    """
    Tạo embeddings từ văn bản sử dụng Sentence-Transformers.

    Mô hình mặc định: all-MiniLM-L6-v2
        - Kích thước vector: 384 chiều
        - Tốc độ: Nhanh, phù hợp demo
        - Chất lượng: Tốt cho tiếng Anh và đa ngôn ngữ
    """

    def __init__(self, config: RAGConfig):
        logger.info("Đang tải embedding model: %s", config.embedding_model)
        self.model = SentenceTransformer(config.embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding model sẵn sàng. Dimension: %d", self.embedding_dim)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Tạo embeddings cho danh sách văn bản.

        Args:
            texts: Danh sách chuỗi văn bản
            batch_size: Số lượng xử lý một lúc

        Returns:
            numpy array shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # L2 normalize -> cosine similarity = dot product
        )
        logger.info("Đã tạo %d embeddings, shape: %s", len(texts), embeddings.shape)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Tạo embedding cho một câu truy vấn đơn.

        Args:
            query: Câu hỏi hoặc từ khóa tìm kiếm

        Returns:
            numpy array shape (embedding_dim,)
        """
        return self.model.encode(query, normalize_embeddings=True)
