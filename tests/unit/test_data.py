"""
Unit tests cho Data Loader và Preprocessor.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.preprocessor import DestinationPreprocessor


# ─── DataLoader tests ─────────────────────────────────────────────────────────

class TestDataLoader:
    @pytest.fixture
    def loader(self):
        """Dùng thư mục data/raw thực tế."""
        return DataLoader(Path("data/raw"))

    def test_load_destinations_returns_dataframe(self, loader):
        df = loader.load_destinations()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_destinations_has_required_columns(self, loader):
        df = loader.load_destinations()
        required = {"destination_id", "name", "location", "category", "rating", "price_level"}
        assert required.issubset(set(df.columns))

    def test_load_reviews_returns_dataframe(self, loader):
        df = loader.load_reviews()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_reviews_has_required_columns(self, loader):
        df = loader.load_reviews()
        required = {"user_id", "destination_id", "rating"}
        assert required.issubset(set(df.columns))

    def test_missing_file_raises(self, tmp_path):
        loader = DataLoader(tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load_destinations("nonexistent.csv")

    def test_load_knowledge_base_returns_list(self, loader):
        docs = loader.load_knowledge_base_docs()
        assert isinstance(docs, list)
        assert all("source" in d and "content" in d for d in docs)


# ─── DestinationPreprocessor tests ───────────────────────────────────────────

@pytest.fixture
def sample_destinations_df():
    """DataFrame địa điểm mẫu đủ đa dạng."""
    return pd.DataFrame({
        "destination_id": ["D001", "D002", "D003", "D004"],
        "name": ["Hà Nội", "Đà Nẵng", "Hội An", "Phú Quốc"],
        "location": ["Hà Nội", "Đà Nẵng", "Quảng Nam", "Kiên Giang"],
        "category": ["thành phố", "biển", "phố cổ", "biển"],
        "rating": [4.5, 4.2, 4.8, 4.6],
        "price_level": [2, 2, 1, 3],
        "tags": ["lịch sử,văn hóa", "biển,ẩm thực", "phố cổ,đèn lồng", "nghỉ dưỡng,biển"],
    })


class TestDestinationPreprocessor:
    def test_fit_transform_returns_ndarray(self, sample_destinations_df):
        preprocessor = DestinationPreprocessor()
        X = preprocessor.fit_transform(sample_destinations_df)
        assert isinstance(X, np.ndarray)

    def test_output_shape_correct(self, sample_destinations_df):
        preprocessor = DestinationPreprocessor()
        X = preprocessor.fit_transform(sample_destinations_df)
        assert X.shape[0] == len(sample_destinations_df)
        assert X.shape[1] > 2  # Nhiều hơn chỉ 2 numerical features

    def test_fit_transform_sets_fitted(self, sample_destinations_df):
        preprocessor = DestinationPreprocessor()
        assert not preprocessor._is_fitted
        preprocessor.fit_transform(sample_destinations_df)
        assert preprocessor._is_fitted

    def test_transform_without_fit_raises(self, sample_destinations_df):
        preprocessor = DestinationPreprocessor()
        with pytest.raises(RuntimeError):
            preprocessor.transform(sample_destinations_df)

    def test_handles_missing_values(self):
        """Preprocessor phải xử lý được NaN mà không raise."""
        df = pd.DataFrame({
            "destination_id": ["D001", "D002"],
            "name": ["A", "B"],
            "location": ["HN", None],
            "category": [None, "biển"],
            "rating": [4.0, None],
            "price_level": [None, 2],
            "tags": ["lịch sử", None],
        })
        preprocessor = DestinationPreprocessor()
        X = preprocessor.fit_transform(df)
        assert X.shape[0] == 2
        assert not np.any(np.isnan(X))

    def test_values_scaled_between_0_and_1(self, sample_destinations_df):
        """Numerical features (rating, price_level) phải được scale về [0,1]."""
        preprocessor = DestinationPreprocessor()
        X = preprocessor.fit_transform(sample_destinations_df)
        # 2 cột đầu là numerical (sau scaling)
        numerical_part = X[:, :2]
        assert numerical_part.min() >= -1e-6
        assert numerical_part.max() <= 1.0 + 1e-6
