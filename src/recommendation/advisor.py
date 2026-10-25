"""
Travel Advisor - RAG-based chatbot tư vấn du lịch.
"""

from src.retrieval.retriever import DocumentRetriever
from src.utils.config import RAGConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý tư vấn du lịch thông minh cho Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp.

Nguyên tắc:
- Chỉ trả lời dựa trên CONTEXT được cung cấp
- Nếu không có đủ thông tin, hãy nói rõ và gợi ý người dùng hỏi câu khác
- Trả lời bằng tiếng Việt, thân thiện và hữu ích
- Đề xuất thêm địa điểm liên quan nếu phù hợp
"""


class TravelAdvisor:
    """
    RAG-based Travel Advisor: Retrieve → Augment → Generate

    Flow:
        1. Retrieve: Tìm tài liệu liên quan bằng DocumentRetriever
        2. Augment: Tạo prompt kết hợp context + câu hỏi
        3. Generate: Gọi LLM để tạo câu trả lời
    """

    def __init__(self, retriever: DocumentRetriever, config: RAGConfig):
        self.retriever = retriever
        self.config = config
        self._llm = self._init_llm()

    def _init_llm(self):
        """Khởi tạo LLM client dựa trên config provider."""
        if self.config.llm_provider == "openai":
            from openai import OpenAI
            logger.info("Sử dụng OpenAI: %s", self.config.llm_model)
            return OpenAI(api_key=self.config.openai_api_key)

        elif self.config.llm_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.config.gemini_api_key)
            logger.info("Sử dụng Gemini: %s", self.config.llm_model)
            return genai.GenerativeModel(self.config.llm_model)

        else:
            raise ValueError(f"LLM provider không hỗ trợ: {self.config.llm_provider}")

    def answer(self, question: str) -> dict:
        """
        Trả lời câu hỏi du lịch bằng RAG pipeline.

        Args:
            question: Câu hỏi của người dùng (tiếng Việt hoặc tiếng Anh)

        Returns:
            dict với các key:
                - answer (str): Câu trả lời từ LLM
                - sources (list[str]): Danh sách nguồn tài liệu đã dùng
                - retrieved_docs (list): Tài liệu đã retrieve
        """
        # Step 1: Retrieve
        results = self.retriever.retrieve(question)
        context = self.retriever.retrieve_as_context(question)

        # Step 2: Augment - tạo prompt
        prompt = self._build_prompt(question, context)

        # Step 3: Generate
        answer_text = self._generate(prompt)

        return {
            "answer": answer_text,
            "sources": list({r.source for r in results}),
            "retrieved_docs": [{"source": r.source, "score": r.score} for r in results],
        }

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""{SYSTEM_PROMPT}

CONTEXT (Thông tin tham khảo):
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""

    def _generate(self, prompt: str) -> str:
        """Gọi LLM để sinh câu trả lời."""
        if self.config.llm_provider == "openai":
            response = self._llm.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()

        elif self.config.llm_provider == "gemini":
            response = self._llm.generate_content(prompt)
            return response.text.strip()

        return "Lỗi: LLM provider không được hỗ trợ."
