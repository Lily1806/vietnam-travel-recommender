"""
Data Loader - Load và validate dữ liệu từ các nguồn khác nhau.
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load dữ liệu thô từ thư mục data/raw/."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def load_destinations(self, filename: str = "destinations.csv") -> pd.DataFrame:
        """
        Load danh sách địa điểm du lịch.

        Returns:
            DataFrame với các cột:
                - destination_id (str): ID duy nhất
                - name (str): Tên địa điểm
                - location (str): Tỉnh/thành phố
                - category (str): Loại hình (biển, núi, thành phố, ...)
                - rating (float): Đánh giá trung bình (0-5)
                - price_level (int): Mức giá (1=rẻ, 2=TB, 3=cao)
                - tags (str): Nhãn phân loại (comma-separated)
        """
        path = self.raw_dir / filename
        self._validate_file(path)
        df = pd.read_csv(path, encoding="utf-8")
        logger.info("Loaded %d destinations from %s", len(df), path)
        return df

    def load_reviews(self, filename: str = "reviews.csv") -> pd.DataFrame:
        """
        Load dữ liệu đánh giá của người dùng.

        Returns:
            DataFrame với các cột:
                - user_id (str): ID người dùng
                - destination_id (str): ID địa điểm
                - rating (float): Điểm đánh giá (1-5)
                - review_text (str): Nội dung nhận xét
        """
        path = self.raw_dir / filename
        self._validate_file(path)
        df = pd.read_csv(path, encoding="utf-8")
        logger.info("Loaded %d reviews from %s", len(df), path)
        return df

    def load_knowledge_base_docs(self) -> list[dict]:
        """
        Load tài liệu từ thư mục knowledge_base/ cho RAG.

        Returns:
            Danh sách dict: {"source": filename, "content": text}
        """
        kb_dir = self.raw_dir / "knowledge_base"
        docs = []

        for file_path in kb_dir.glob("*.txt"):
            content = file_path.read_text(encoding="utf-8")
            docs.append({"source": file_path.name, "content": content})

        logger.info("Loaded %d knowledge base documents", len(docs))
        return docs

    @staticmethod
    def _validate_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
