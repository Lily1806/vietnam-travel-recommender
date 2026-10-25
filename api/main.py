"""
FastAPI Application - REST API cho hệ thống gợi ý du lịch.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import build_knn_recommender, build_rag_components, build_travel_advisor
from api.routers import advise, recommend
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo tất cả models khi server start, dọn dẹp khi shutdown."""
    config = load_config()
    logger.info("🚀 Khởi động Travel Recommendation API...")

    # 1. KNN Recommender
    try:
        app.state.recommender = build_knn_recommender(config)
    except Exception as e:
        logger.error("❌ Không thể khởi tạo KNN Recommender: %s", e)
        app.state.recommender = None

    # 2. RAG Components (Embedder + FAISS + Retriever)
    try:
        _embedder, _store, retriever = build_rag_components(config)
        app.state.retriever = retriever
    except Exception as e:
        logger.error("❌ Không thể khởi tạo RAG components: %s", e)
        app.state.retriever = None
        retriever = None

    # 3. Travel Advisor (LLM) - chỉ khởi tạo nếu retriever OK
    if retriever is not None:
        app.state.advisor = build_travel_advisor(retriever, config)
    else:
        app.state.advisor = None

    logger.info("✅ Server sẵn sàng nhận request.")
    yield
    logger.info("🛑 Server đang tắt...")


app = FastAPI(
    title="🗺️ Travel Recommendation API",
    description="""
API hệ thống gợi ý và tư vấn du lịch thông minh cho Việt Nam.

## Chức năng
- **KNN Recommendation**: Gợi ý địa điểm tương tự (Item-based) hoặc dựa trên sở thích người dùng (User-based)
- **RAG Advisor**: Tư vấn du lịch qua ngôn ngữ tự nhiên

## Chế độ hoạt động
- Nếu có **OPENAI_API_KEY** hoặc **GEMINI_API_KEY** trong `.env` → RAG trả lời đầy đủ bằng LLM
- Nếu không có API key → RAG trả về đoạn trích dẫn từ knowledge base (fallback mode)
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, prefix="/api/v1", tags=["Recommendation"])
app.include_router(advise.router, prefix="/api/v1", tags=["Advisor"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Kiểm tra trạng thái server và các components."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "components": {
            "knn_recommender": "ready" if app.state.recommender is not None else "unavailable",
            "rag_retriever": "ready" if app.state.retriever is not None else "unavailable",
            "llm_advisor": "ready" if app.state.advisor is not None else "fallback (no API key)",
        },
    }
