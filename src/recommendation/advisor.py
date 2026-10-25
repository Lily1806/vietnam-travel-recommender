"""
Travel Advisor - RAG-based chatbot tư vấn du lịch Việt Nam.
"""

import re
from src.retrieval.retriever import DocumentRetriever
from src.utils.config import RAGConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý tư vấn du lịch thông minh và chu đáo cho Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi của du khách dựa trên thông tin CONTEXT được cung cấp.

Nguyên tắc trả lời:
- Trả lời NGẮN GỌN, SÚC TÍCH, ĐI THẲNG VÀO TRỌNG TÂM (tối đa 150-200 từ).
- Sử dụng các gạch đầu dòng (•) rõ ràng cho các mốc thời gian, món ăn, hoặc điểm tham quan.
- Thêm 1 lời khuyên thực tế (thời tiết, chuẩn bị đồ, hoặc chi phí) nếu có.
- Tuyệt đối không sao chép nguyên văn cả đoạn văn bản dài; tổng hợp tự nhiên, lịch sự bằng tiếng Việt.
- Nếu không có thông tin trong CONTEXT, hãy thông báo lịch sự và gợi ý câu hỏi liên quan.
"""

SOURCE_NAME_MAP = {
    "a_lat_thanh_pho_ngan_hoa.txt": "Đà Lạt: Thành phố ngàn hoa",
    "a_nang_thanh_pho_nang_ong_ben_bien.txt": "Đà Nẵng: Thành phố năng động bên biển",
    "ai_noi_hue_kinh_thanh_trieu_nguyen.txt": "Đại Nội Huế: Kinh thành triều Nguyễn",
    "at_ve_may_bay_en_viet_nam.txt": "Hướng dẫn đặt vé máy bay đến Việt Nam",
    "ao_cat_ba_va_vuon_quoc_gia.txt": "Đảo Cát Bà & Vườn quốc gia",
    "ao_cu_lao_cham_lan_ngam_san_ho.txt": "Đảo Cù Lao Chàm: Lặn ngắm san hô",
    "ban_ao_son_tra_nui_khi.txt": "Bán đảo Sơn Trà: Núi khỉ",
    "dinh_oc_lap.txt": "Dinh Độc Lập - TP. Hồ Chí Minh",
    "ha_noi_thu_o_ngan_nam_van_hien.txt": "Hà Nội: Thủ đô ngàn năm văn hiến",
    "cha_ca_la_vong.txt": "Ẩm thực Hà Nội: Chả cá Lã Vọng",
    "bun_cha.txt": "Ẩm thực Hà Nội: Bún chả",
    "bun_bo_hue.txt": "Ẩm thực miền Trung: Bún bò Huế",
    "pho_viet_nam.txt": "Ẩm thực: Phở Việt Nam",
    "banh_mi.txt": "Ẩm thực đường phố: Bánh mì",
    "banh_xeo.txt": "Ẩm thực: Bánh xèo",
    "ga_xe_lua_co_a_lat.txt": "Ga xe lửa cổ Đà Lạt",
    "cac_thac_nuoc_o_a_lat.txt": "Các thác nước đẹp tại Đà Lạt",
    "suoi_tien_mui_ne.txt": "Suối Tiên Mũi Né - Phan Thiết",
    "thanh_dia_my_son.txt": "Thánh địa Mỹ Sơn - Quảng Nam",
    "pho_co_hoi_an.txt": "Phố cổ Hội An",
    "vinh_ha_long_di_san_the_gioi.txt": "Vịnh Hạ Long: Di sản thế giới",
}


def clean_source_name(src: str) -> str:
    """Chuẩn hóa tên nguồn hiển thị đẹp mắt."""
    if src in SOURCE_NAME_MAP:
        return SOURCE_NAME_MAP[src]
    name = src.replace(".txt", "").replace("_", " ")
    if name.startswith("a "):
        name = "Đà " + name[2:]
    return name.title()


def generate_smart_answer(question: str, results: list) -> str:
    """
    Sinh câu trả lời thông minh dựa trên trích xuất câu và tổng hợp trọng tâm
    (Áp dụng khi chưa cấu hình LLM API Key hoặc LLM gặp lỗi).
    """
    if not results:
        return "Xin lỗi, hiện tại tôi chưa tìm thấy thông tin phù hợp với câu hỏi của bạn trong cơ sở dữ liệu."

    from src.retrieval.retriever import detect_entity
    dest_name, aliases = detect_entity(question)

    # Nếu câu hỏi nhắc đến địa danh cụ thể, lọc chỉ lấy các tài liệu liên quan đến địa danh đó
    if dest_name and aliases:
        filtered = [
            r for r in results
            if any(a in r.source.lower().replace("_", " ") or a in r.content.lower() for a in aliases)
        ]
        target_results = filtered if filtered else [results[0]]
    else:
        target_results = results

    q_norm = question.lower()
    q_words = set(re.findall(r"\w+", q_norm))

    candidates = []
    tips = []

    for r in target_results:
        content = r.content
        content = re.sub(r"#+\s*Đoạn\s*\d+", "", content)
        content = re.sub(r"#+\s*[^\n]+", "", content)

        sents = re.split(r"[\n\.]+", content)
        for s in sents:
            s = s.strip()
            if not s or len(s) < 25 or len(s) > 280:
                continue
            if s.startswith("**") and s.endswith("**"):
                continue

            s_norm = s.lower()
            s_words = set(re.findall(r"\w+", s_norm))
            overlap = len(q_words.intersection(s_words))

            # Điểm ưu tiên ngữ nghĩa
            bonus = 0
            if any(k in q_norm for k in ["chi phí", "giá", "ngân sách", "bao nhiêu"]):
                if any(k in s_norm for k in ["chi phí", "dao động", "triệu", "đồng/ngày", "ngân sách", "giá cả", "tiết kiệm", "vnđ"]):
                    bonus += 4
            if any(k in q_norm for k in ["thời điểm", "khi nào", "tháng", "mùa", "thời tiết"]):
                if any(k in s_norm for k in ["tháng", "mùa", "thời điểm", "thời tiết", "lý tưởng", "đẹp nhất", "thời gian"]):
                    bonus += 4
            if any(k in q_norm for k in ["ăn", "món", "ẩm thực", "ngon", "đặc sản"]):
                if any(k in s_norm for k in ["món", "ẩm thực", "thưởng thức", "đặc sản", "nước dùng", "hương vị", "quán"]):
                    bonus += 4
            if any(k in q_norm for k in ["vé", "bay", "đặt", "giá vé", "chuyến bay"]):
                if any(k in s_norm for k in ["vé", "đặt vé", "giá vé", "hãng hàng không", "sớm", "tiết kiệm"]):
                    bonus += 4
            if any(k in s_norm for k in ["lời khuyên", "lưu ý", "chuẩn bị", "đừng quên", "kinh nghiệm"]):
                tips.append((overlap + bonus, s))

            score = overlap + bonus
            candidates.append((score, s, r.source))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Chọn 2-3 câu không trùng lặp ngữ nghĩa
    selected = []
    seen = []
    for score, sent, src in candidates:
        words = set(re.findall(r"\w+", sent.lower()))
        if any(len(words.intersection(p)) / max(len(words), 1) > 0.45 for p in seen):
            continue
        selected.append((sent, src))
        seen.append(words)
        if len(selected) >= 3:
            break

    if not selected:
        return f"💡 **Thông tin gợi ý**: {target_results[0].content[:250]}..."

    bullets = "\n".join([f"• {s}." for s, _ in selected])
    unique_sources = list(dict.fromkeys([clean_source_name(src) for _, src in selected]))

    tip_text = ""
    if tips:
        tips.sort(key=lambda x: x[0], reverse=True)
        tip_sent = tips[0][1]
        tip_text = f"\n\n💡 **Lời khuyên**: {tip_sent}."

    answer = (
        f"🧭 **Thông tin tư vấn cho chuyến đi:**\n\n"
        f"{bullets}"
        f"{tip_text}\n\n"
        f"📌 *Cẩm nang tham khảo: {', '.join(unique_sources)}*"
    )
    return answer



class TravelAdvisor:
    """
    RAG-based Travel Advisor: Retrieve → Augment → Generate.
    """

    def __init__(self, retriever: DocumentRetriever, config: RAGConfig):
        self.retriever = retriever
        self.config = config
        self._llm = self._init_llm()

    def _init_llm(self):
        """Khởi tạo LLM client dựa trên config provider."""
        if self.config.llm_provider == "openai" and self.config.openai_api_key:
            from openai import OpenAI
            logger.info("Sử dụng OpenAI: %s", self.config.llm_model)
            return OpenAI(api_key=self.config.openai_api_key)

        elif self.config.llm_provider == "gemini" and self.config.gemini_api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.config.gemini_api_key)
            logger.info("Sử dụng Gemini: %s", self.config.llm_model)
            return genai.GenerativeModel(self.config.llm_model)

        return None

    def answer(self, question: str) -> dict:
        """
        Trả lời câu hỏi du lịch bằng RAG pipeline.
        Nếu chưa có LLM, sử dụng thuật toán tổng hợp thông minh.
        """
        results = self.retriever.retrieve(question)

        if self._llm is None:
            answer_text = generate_smart_answer(question, results)
        else:
            context = self.retriever.retrieve_as_context(question)
            prompt = self._build_prompt(question, context)
            try:
                answer_text = self._generate(prompt)
            except Exception as e:
                logger.error("Lỗi gọi LLM: %s. Chuyển sang tổng hợp trích xuất.", e)
                answer_text = generate_smart_answer(question, results)

        return {
            "answer": answer_text,
            "sources": list({r.source for r in results}),
            "retrieved_docs": [{"source": r.source, "score": round(r.score, 4)} for r in results],
        }

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""{SYSTEM_PROMPT}

CONTEXT (Thông tin tham khảo từ cơ sở dữ liệu du lịch Việt Nam):
{context}

CÂU HỎI CỦA DU KHÁCH: {question}

CÂU TRẢ LỜI NGẮN GỌN, RÕ Ý:"""

    def _generate(self, prompt: str) -> str:
        """Gọi LLM để sinh câu trả lời."""
        if self.config.llm_provider == "openai":
            response = self._llm.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=400,
            )
            return response.choices[0].message.content.strip()

        elif self.config.llm_provider == "gemini":
            response = self._llm.generate_content(prompt)
            return response.text.strip()

        return "Lỗi: LLM provider không được hỗ trợ."

