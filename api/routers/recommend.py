"""
Recommendation Router - Endpoints cho KNN-based recommendation.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_recommender
from api.schemas import RecommendationItem, RecommendResponse, SimilarRequest, UserBasedRequest
from src.models.knn_recommender import KNNRecommender

router = APIRouter()


def _to_response(query: str, recs) -> RecommendResponse:
    """Chuyển đổi danh sách Recommendation thành RecommendResponse."""
    return RecommendResponse(
        query=query,
        results=[
            RecommendationItem(
                destination_id=r.destination_id,
                name=r.name,
                similarity_score=r.similarity_score,
                rank=r.rank,
            )
            for r in recs
        ],
        total=len(recs),
    )


@router.post(
    "/recommend/similar",
    response_model=RecommendResponse,
    summary="Gợi ý địa điểm tương tự (Item-based KNN)",
    response_description="Danh sách địa điểm tương tự, sắp xếp theo độ tương đồng giảm dần",
)
async def recommend_similar(
    request: SimilarRequest,
    recommender: KNNRecommender = Depends(get_recommender),
):
    """
    Gợi ý địa điểm du lịch tương tự với một địa điểm được chỉ định.

    - **destination_id**: ID của địa điểm tham chiếu (ví dụ: `VN001`)
    - **k**: Số lượng địa điểm gợi ý (mặc định 5, tối đa 50)

    **Ví dụ**: Nhập `VN001` (Hà Nội) → nhận gợi ý các thành phố/di tích tương tự.
    """
    try:
        results = recommender.recommend_similar(
            destination_id=request.destination_id,
            k=request.k,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _to_response(
        query=f"Địa điểm tương tự với {request.destination_id}",
        recs=results,
    )


@router.post(
    "/recommend/for-user",
    response_model=RecommendResponse,
    summary="Gợi ý dựa trên sở thích người dùng (User-based KNN)",
    response_description="Danh sách địa điểm phù hợp với profile người dùng",
)
async def recommend_for_user(
    request: UserBasedRequest,
    recommender: KNNRecommender = Depends(get_recommender),
):
    """
    Gợi ý địa điểm dựa trên lịch sử đánh giá của người dùng.

    Thuật toán tạo **user profile vector** = trung bình có trọng số của các địa điểm đã đánh giá,
    sau đó tìm K địa điểm gần nhất chưa từng ghé thăm.

    - **user_ratings**: Dict `{destination_id: rating}`, rating từ 1.0–5.0
    - **k**: Số lượng gợi ý (mặc định 10)

    **Ví dụ**: `{"VN001": 5.0, "VN010": 4.0}` → gợi ý dựa trên Hà Nội + Hội An.
    """
    try:
        results = recommender.recommend_for_user(
            user_ratings=request.user_ratings,
            k=request.k,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    visited = ", ".join(request.user_ratings.keys())
    return _to_response(
        query=f"Gợi ý cho người dùng đã thích: {visited}",
        recs=results,
    )


@router.get(
    "/recommend/destinations",
    summary="Lấy danh sách tất cả địa điểm hợp lệ",
    response_description="Danh sách destination_id và tên địa điểm",
)
async def list_destinations(recommender: KNNRecommender = Depends(get_recommender)):
    """
    Lấy danh sách tất cả địa điểm du lịch trong hệ thống.
    Dùng để tra cứu **destination_id** hợp lệ trước khi gọi các endpoint recommend.
    """
    df = recommender._destinations_df
    return {
        "total": len(df),
        "destinations": [
            {
                "destination_id": str(row["destination_id"]),
                "name": str(row["name"]),
                "location": str(row.get("location", "")),
                "category": str(row.get("category", "")),
                "rating": float(row.get("rating", 0)),
            }
            for _, row in df.iterrows()
        ],
    }
