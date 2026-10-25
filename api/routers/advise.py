"""
Advise Router - Endpoints cho RAG-based travel advisor.
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_advisor, get_retriever
from api.schemas import AdviseRequest, AdviseResponse
from src.recommendation.advisor import TravelAdvisor
from src.retrieval.retriever import DocumentRetriever

router = APIRouter()


def _fallback_answer(question: str, retriever: DocumentRetriever, top_k: int) -> AdviseResponse:
    """
    Fallback khi không có LLM: trả về trực tiếp các đoạn tài liệu liên quan
    thay vì câu trả lời được tổng hợp.
    """
    results = retriever.retrieve(question, top_k=top_k)
    if not results:
        answer = (
            "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn "
            "trong cơ sở dữ liệu hiện tại."
        )
    else:
        # Ghép các đoạn tài liệu liên quan nhất thành câu trả lời
        parts = [
            f"**[Nguồn: {r.source}]**\n{r.content[:600]}"
            for r in results[:3]
        ]
        answer = (
            "📚 *Chế độ fallback (không có LLM API key) - Trích dẫn từ knowledge base:*\n\n"
            + "\n\n---\n\n".join(parts)
        )

    return AdviseResponse(
        question=question,
        answer=answer,
        sources=list({r.source for r in results}),
        retrieved_docs=[{"source": r.source, "score": round(r.score, 4)} for r in results],
    )


@router.post(
    "/advise",
    response_model=AdviseResponse,
    summary="Tư vấn du lịch thông minh (RAG Pipeline)",
    response_description="Câu trả lời tổng hợp từ LLM dựa trên knowledge base",
)
async def advise(
    request: AdviseRequest,
    retriever: DocumentRetriever = Depends(get_retriever),
    advisor: TravelAdvisor | None = Depends(get_advisor),
):
    """
    Tư vấn du lịch Việt Nam bằng ngôn ngữ tự nhiên sử dụng RAG Pipeline.

    **Luồng xử lý:**
    1. **Retrieve**: Tìm tài liệu liên quan nhất trong FAISS vector store
    2. **Augment**: Tạo prompt kết hợp context tài liệu + câu hỏi
    3. **Generate**: Gọi LLM (OpenAI / Gemini) để sinh câu trả lời

    **Nếu chưa cấu hình API key**: Trả về trích đoạn từ knowledge base (fallback mode).

    - **question**: Câu hỏi du lịch (tiếng Việt hoặc tiếng Anh)
    - **top_k**: Số tài liệu dùng làm context (mặc định 5)
    """
    if advisor is None:
        # Fallback mode: không có LLM
        return _fallback_answer(request.question, retriever, request.top_k)

    result = advisor.answer(request.question)
    return AdviseResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        retrieved_docs=result["retrieved_docs"],
    )


@router.get(
    "/advise/health",
    summary="Kiểm tra trạng thái RAG Advisor",
    tags=["Health"],
)
async def advise_health(
    retriever: DocumentRetriever = Depends(get_retriever),
    advisor: TravelAdvisor | None = Depends(get_advisor),
):
    """Kiểm tra số lượng tài liệu trong knowledge base và trạng thái LLM."""
    return {
        "knowledge_base_chunks": retriever.vector_store.index.ntotal,
        "llm_mode": (
            f"{advisor.config.llm_provider}/{advisor.config.llm_model}"
            if advisor is not None
            else "fallback (no API key)"
        ),
    }
