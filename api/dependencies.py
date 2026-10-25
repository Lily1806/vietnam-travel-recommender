"""
Dependencies - Quản lý việc khởi tạo và cung cấp models cho API.

Sử dụng FastAPI dependency injection để inject KNNRecommender và TravelAdvisor
vào các endpoint. Models được nạp một lần khi server start (lifespan).
"""

from pathlib import Path

from fastapi import HTTPException, Request

from src.data.loader import DataLoader
from src.data.preprocessor import DestinationPreprocessor
from src.models.knn_recommender import KNNRecommender
from src.recommendation.advisor import TravelAdvisor
from src.retrieval.embedder import TextEmbedder
from src.retrieval.retriever import DocumentRetriever
from src.retrieval.vector_store import FAISSVectorStore
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_knn_recommender(config: AppConfig) -> KNNRecommender:
    """Nạp dữ liệu, huấn luyện và trả về KNNRecommender đã fit."""
    loader = DataLoader(config.data.raw_dir)
    df = loader.load_destinations()

    preprocessor = DestinationPreprocessor()
    feature_matrix = preprocessor.fit_transform(df)

    recommender = KNNRecommender(config.knn)
    recommender.fit(feature_matrix, df)

    logger.info("✅ KNNRecommender sẵn sàng. %d địa điểm.", len(df))
    return recommender


def build_rag_components(config: AppConfig):
    """
    Nạp FAISS index, khởi tạo embedder và retriever.

    Returns:
        (TextEmbedder, FAISSVectorStore, DocumentRetriever)
    """
    embedder = TextEmbedder(config.rag)
    store = FAISSVectorStore.load(config.data.embeddings_dir)
    retriever = DocumentRetriever(embedder, store, config.rag)
    logger.info("✅ RAG retriever sẵn sàng. %d chunks.", len(store._documents))
    return embedder, store, retriever


def build_travel_advisor(retriever: DocumentRetriever, config: AppConfig) -> TravelAdvisor:
    """
    Khởi tạo TravelAdvisor.
    Nếu có LLM API key, hệ thống sẽ gọi LLM.
    Nếu chưa có API key, hệ thống tự động sử dụng thuật toán Smart Extractive Synthesis.
    """
    has_openai = bool(config.rag.openai_api_key)
    has_gemini = bool(config.rag.gemini_api_key)

    if not has_openai and not has_gemini:
        logger.info(
            "ℹ️  Chưa cấu hình LLM API key. "
            "RAG advisor hoạt động ở chế độ Trích lọc thông minh (Smart Extractive)."
        )

    try:
        advisor = TravelAdvisor(retriever, config.rag)
        logger.info("✅ TravelAdvisor sẵn sàng.")
        return advisor
    except Exception as e:
        logger.error("❌ Lỗi khởi tạo TravelAdvisor: %s", e)
        return TravelAdvisor(retriever, config.rag)



# ──────────────── FastAPI Dependency Getters ────────────────────────────────

def get_recommender(request: Request) -> KNNRecommender:
    """Dependency: Lấy KNNRecommender từ app.state."""
    recommender: KNNRecommender | None = getattr(request.app.state, "recommender", None)
    if recommender is None:
        raise HTTPException(status_code=503, detail="KNN Recommender chưa được khởi tạo.")
    return recommender


def get_retriever(request: Request) -> DocumentRetriever:
    """Dependency: Lấy DocumentRetriever từ app.state."""
    retriever: DocumentRetriever | None = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="RAG Retriever chưa được khởi tạo.")
    return retriever


def get_advisor(request: Request) -> TravelAdvisor | None:
    """Dependency: Lấy TravelAdvisor từ app.state (có thể None nếu không có API key)."""
    return getattr(request.app.state, "advisor", None)
