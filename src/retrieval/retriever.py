"""
Retriever - Tìm kiếm tài liệu liên quan từ vector store cho RAG với Hybrid Search.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.retrieval.embedder import TextEmbedder
from src.retrieval.vector_store import FAISSVectorStore, SearchResult
from src.utils.config import RAGConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Danh sách địa danh và thực thể phổ biến trong du lịch Việt Nam
DESTINATION_ENTITIES = [
    ("sa pa", ["sa pa", "sapa"]),
    ("đà lạt", ["đà lạt", "da lat", "a lat"]),
    ("hà nội", ["hà nội", "ha noi"]),
    ("hội an", ["hội an", "hoi an"]),
    ("phú quốc", ["phú quốc", "phu quoc"]),
    ("đà nẵng", ["đà nẵng", "da nang", "a nang"]),
    ("huế", ["huế", "hue"]),
    ("hạ long", ["hạ long", "ha long"]),
    ("nha trang", ["nha trang"]),
    ("ninh bình", ["ninh bình", "ninh binh"]),
    ("cát bà", ["cát bà", "cat ba", "ao cat ba"]),
    ("mỹ sơn", ["mỹ sơn", "my son"]),
    ("mũi né", ["mũi né", "mui ne"]),
    ("fansipan", ["fansipan", "phan xi pang"]),
    ("vé máy bay", ["vé máy bay", "ve may bay", "chuyến bay", "đặt vé"]),
]


def detect_entity(query: str):
    """Phát hiện địa danh hoặc chủ đề chính trong câu hỏi."""
    q = query.lower()
    for name, aliases in DESTINATION_ENTITIES:
        for alias in aliases:
            if alias in q:
                return name, aliases
    return None, []


def _normalize_text(text: str) -> str:
    """Chuẩn hóa tiếng Việt để matching từ khóa tốt hơn."""
    text = text.lower()
    text = text.replace("đ", "d")
    return text


class DocumentRetriever:
    """
    Retrieval component của RAG pipeline với Hybrid Search + Entity Recognition.
    """

    def __init__(self, embedder: TextEmbedder, vector_store: FAISSVectorStore, config: RAGConfig):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = config.top_k_retrieval

        self._documents = getattr(vector_store, "_documents", [])
        if self._documents:
            contents = [d["content"] for d in self._documents]
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
            self.tfidf_matrix = self.tfidf.fit_transform(contents)
            logger.info("✅ Hybrid Search sẵn sàng: TF-IDF index cho %d chunks.", len(contents))
        else:
            self.tfidf = None
            self.tfidf_matrix = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """
        Tìm tài liệu liên quan nhất kết hợp Entity Recognition, Dense FAISS và Lexical TF-IDF.
        """
        k = top_k or self.top_k
        n_docs = len(self._documents)
        if n_docs == 0:
            return []

        dest_name, aliases = detect_entity(query)

        # 1. Dense Search qua FAISS
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_embedding, top_k=min(k * 4, n_docs))
        dense_scores = np.zeros(n_docs, dtype=np.float32)
        for r in dense_results:
            dense_scores[r.chunk_id] = max(0.0, float(r.score))

        if dense_scores.max() > dense_scores.min():
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)

        # 2. Lexical Search qua TF-IDF
        lexical_scores = np.zeros(n_docs, dtype=np.float32)
        if self.tfidf is not None:
            q_vec = self.tfidf.transform([query])
            tfidf_sim = (self.tfidf_matrix * q_vec.T).toarray().ravel()
            if tfidf_sim.max() > 0:
                lexical_scores = tfidf_sim / (tfidf_sim.max() + 1e-8)

        # 3. Entity & Keyword matching
        entity_scores = np.zeros(n_docs, dtype=np.float32)
        for i, doc in enumerate(self._documents):
            src_norm = doc["source"].lower().replace("_", " ")
            content_lower = doc["content"].lower()

            if dest_name:
                is_match = any(a in src_norm or a in content_lower for a in aliases)
                if is_match:
                    entity_scores[i] = 0.5
                    if any(a in src_norm for a in aliases):
                        entity_scores[i] += 0.3  # Filename matches entity
                else:
                    entity_scores[i] = -0.4  # Penalize documents of unrelated destinations

        # 4. Tổng hợp điểm số
        if dest_name:
            final_scores = 0.20 * dense_scores + 0.40 * lexical_scores + entity_scores
        else:
            final_scores = 0.40 * dense_scores + 0.60 * lexical_scores

        top_indices = np.argsort(final_scores)[::-1][:k]

        results = []
        for idx in top_indices:
            doc = self._documents[idx]
            results.append(SearchResult(
                chunk_id=idx,
                source=doc["source"],
                content=doc["content"],
                score=float(final_scores[idx]),
            ))

        logger.debug("Entity '%s' | Query: '%s' → Top 1: %s (score %.4f)",
                     dest_name or "None", query,
                     results[0].source if results else "None",
                     results[0].score if results else 0)
        return results


    def retrieve_as_context(self, query: str, top_k: int | None = None) -> str:
        """
        Tìm và gộp tài liệu thành chuỗi context cho LLM.

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

