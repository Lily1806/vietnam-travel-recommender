"""
Pydantic schemas - Định nghĩa request/response models cho API.
"""

from pydantic import BaseModel, Field


# ──────────────────────────── KNN Recommendation ────────────────────────────

class SimilarRequest(BaseModel):
    """Request gợi ý địa điểm tương tự."""
    destination_id: str = Field(..., example="HAN_001", description="ID địa điểm tham chiếu")
    k: int = Field(default=5, ge=1, le=50, description="Số lượng gợi ý")


class UserBasedRequest(BaseModel):
    """Request gợi ý dựa trên lịch sử người dùng."""
    user_ratings: dict[str, float] = Field(
        ...,
        example={"HAN_001": 4.5, "HCM_002": 3.0},
        description="Dict {destination_id: rating}"
    )
    k: int = Field(default=10, ge=1, le=50, description="Số lượng gợi ý")


class RecommendationItem(BaseModel):
    destination_id: str
    name: str
    similarity_score: float
    rank: int


class RecommendResponse(BaseModel):
    query: str
    results: list[RecommendationItem]
    total: int


# ──────────────────────────── RAG Advisor ────────────────────────────────────

class AdviseRequest(BaseModel):
    """Request tư vấn du lịch."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        example="Hà Nội có những địa điểm nào đẹp để chụp ảnh?",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Số tài liệu retrieve")


class AdviseResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    retrieved_docs: list[dict]
