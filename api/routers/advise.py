"""
Advise Router - Endpoints cho RAG-based travel advisor.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import AdviseRequest, AdviseResponse

router = APIRouter()


@router.post("/advise", response_model=AdviseResponse)
async def advise(request: AdviseRequest):
    """
    Tư vấn du lịch thông minh qua ngôn ngữ tự nhiên (RAG Pipeline).
    """
    # TODO: Inject advisor từ app.state
    raise HTTPException(status_code=501, detail="Chưa tích hợp RAG advisor. Xem hướng dẫn trong docs/architecture.md")
