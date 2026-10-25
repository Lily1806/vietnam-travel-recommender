"""
FastAPI Application - REST API cho hệ thống gợi ý du lịch.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import advise, recommend
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo models khi server start, dọn dẹp khi shutdown."""
    logger.info("🚀 Khởi động Travel Recommendation API...")
    # TODO: Load models vào app.state ở đây
    yield
    logger.info("🛑 Server đang tắt...")


app = FastAPI(
    title="🗺️ Travel Recommendation API",
    description="""
    API hệ thống gợi ý và tư vấn du lịch thông minh.
    
    ## Chức năng
    - **KNN Recommendation**: Gợi ý địa điểm tương tự
    - **RAG Advisor**: Tư vấn du lịch qua ngôn ngữ tự nhiên
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
    return {"status": "ok", "version": "1.0.0"}
