"""
Preprocessor - Làm sạch và biến đổi dữ liệu thô thành feature vectors.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DestinationPreprocessor:
    """
    Tiền xử lý dữ liệu địa điểm để phục vụ KNN.

    Pipeline:
        1. Xử lý missing values
        2. Encode biến categorical (location, category)
        3. Binarize tags (multi-label)
        4. Normalize numerical features (rating, price_level)
        5. Tạo feature matrix cuối
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.mlb = MultiLabelBinarizer()
        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor và transform dữ liệu.

        Args:
            df: DataFrame destinations thô

        Returns:
            Feature matrix shape (n_destinations, n_features)
        """
        df = self._clean(df)
        features = self._engineer_features(df, fit=True)
        self._is_fitted = True
        logger.info("Feature matrix shape: %s", features.shape)
        return features

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform dữ liệu mới (sau khi đã fit).

        Args:
            df: DataFrame destinations mới

        Returns:
            Feature matrix
        """
        if not self._is_fitted:
            raise RuntimeError("Gọi fit_transform() trước khi dùng transform()")
        df = self._clean(df)
        return self._engineer_features(df, fit=False)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý missing values và dữ liệu bất thường."""
        df = df.copy()
        df["rating"] = df["rating"].fillna(df["rating"].median())
        df["price_level"] = df["price_level"].fillna(2)
        df["tags"] = df["tags"].fillna("").apply(
            lambda x: [t.strip() for t in x.split(",") if t.strip()]
        )
        df["category"] = df["category"].fillna("unknown")
        df["location"] = df["location"].fillna("unknown")
        return df

    def _engineer_features(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        """Tạo feature vector từ DataFrame đã làm sạch."""
        # 1. Numerical features
        numerical = df[["rating", "price_level"]].values
        if fit:
            numerical_scaled = self.scaler.fit_transform(numerical)
        else:
            numerical_scaled = self.scaler.transform(numerical)

        # 2. Categorical features (One-Hot)
        category_dummies = pd.get_dummies(df["category"], prefix="cat").values
        location_dummies = pd.get_dummies(df["location"], prefix="loc").values

        # 3. Multi-label tags (Binary)
        if fit:
            tag_features = self.mlb.fit_transform(df["tags"])
        else:
            tag_features = self.mlb.transform(df["tags"])

        # 4. Concatenate all features
        return np.hstack([numerical_scaled, category_dummies, location_dummies, tag_features])
