"""
Vector Store - Quản lý lưu trữ và tìm kiếm vector embeddings bằng FAISS.
"""

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Kết quả tìm kiếm từ vector store."""
    chunk_id: int
    source: str
    content: str
    score: float


class FAISSVectorStore:
    """
    Vector store dùng FAISS để tìm kiếm tương đồng nhanh.

    Sử dụng IndexFlatIP (Inner Product) với vector đã L2-normalized
    tương đương với cosine similarity.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self._documents: list[dict] = []  # {"source": str, "content": str}

    def add_documents(self, documents: list[dict], embeddings: np.ndarray) -> None:
        """
        Thêm tài liệu và embeddings vào store.

        Args:
            documents: Danh sách dict {"source": ..., "content": ...}
            embeddings: Array embeddings tương ứng, shape (n, dim)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Số lượng documents và embeddings phải bằng nhau")

        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self._documents.extend(documents)
        logger.info("Đã thêm %d documents vào vector store. Tổng: %d",
                    len(documents), len(self._documents))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """
        Tìm kiếm top_k tài liệu tương đồng nhất.

        Args:
            query_embedding: Embedding của câu truy vấn, shape (dim,)
            top_k: Số lượng kết quả trả về

        Returns:
            Danh sách SearchResult, sắp xếp theo score giảm dần
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store đang trống!")
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self._documents[idx]
            results.append(SearchResult(
                chunk_id=idx,
                source=doc["source"],
                content=doc["content"],
                score=float(score),
            ))
        return results

    def save(self, save_dir: Path) -> None:
        """Lưu index FAISS và documents ra disk."""
        save_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_dir / "faiss.index"))

        import json
        with open(save_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False, indent=2)

        logger.info("Đã lưu vector store vào: %s", save_dir)

    @classmethod
    def load(cls, save_dir: Path) -> "FAISSVectorStore":
        """Load vector store đã lưu từ disk."""
        import json

        index = faiss.read_index(str(save_dir / "faiss.index"))
        embedding_dim = index.d

        store = cls(embedding_dim=embedding_dim)
        store.index = index

        with open(save_dir / "documents.json", "r", encoding="utf-8") as f:
            store._documents = json.load(f)

        logger.info("Đã load vector store: %d documents, dim=%d",
                    len(store._documents), embedding_dim)
        return store
