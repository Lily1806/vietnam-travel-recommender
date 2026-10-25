"""
Recommendation Router - Endpoints cho KNN-based recommendation.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import AdviseRequest, AdviseResponse, RecommendResponse, SimilarRequest, UserBasedRequest

router = APIRouter()


@router.post("/recommend/similar", response_model=RecommendResponse)
async def recommend_similar(request: SimilarRequest):
    """
    Gợi ý địa điểm tương tự với một địa điểm cho trước (Item-based KNN).
    """
    # TODO: Inject recommender từ app.state
    raise HTTPException(status_code=501, detail="Chưa tích hợp KNN model. Xem hướng dẫn trong docs/architecture.md")


@router.post("/recommend/for-user", response_model=RecommendResponse)
async def recommend_for_user(request: UserBasedRequest):
    """
    Gợi ý địa điểm dựa trên lịch sử đánh giá của người dùng (User-based KNN).
    """
    raise HTTPException(status_code=501, detail="Chưa tích hợp KNN model.")
