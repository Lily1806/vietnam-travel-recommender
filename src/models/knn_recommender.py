"""
KNN Recommender - Gợi ý địa điểm du lịch dựa trên K-Nearest Neighbors.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.utils.config import KNNConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Recommendation:
    """Kết quả gợi ý một địa điểm."""
    destination_id: str
    name: str
    similarity_score: float
    rank: int


class KNNRecommender:
    """
    Hệ thống gợi ý dựa trên KNN.

    Hỗ trợ hai chế độ:
        - Item-based: Tìm địa điểm tương tự với một địa điểm cho trước
        - User-based: Gợi ý dựa trên lịch sử của người dùng
    """

    def __init__(self, config: KNNConfig):
        self.config = config
        self.model = NearestNeighbors(
            n_neighbors=config.k + 1,    # +1 vì bản thân cũng được tính
            metric=config.metric,
            algorithm=config.algorithm,
            n_jobs=config.n_jobs,
        )
        self._feature_matrix: np.ndarray | None = None
        self._destinations_df: pd.DataFrame | None = None
        self._is_fitted = False

    def fit(self, feature_matrix: np.ndarray, destinations_df: pd.DataFrame) -> "KNNRecommender":
        """
        Huấn luyện KNN trên feature matrix.

        Args:
            feature_matrix: Ma trận đặc trưng shape (n_items, n_features)
            destinations_df: DataFrame chứa thông tin địa điểm (với cột destination_id, name)

        Returns:
            self (để chain method)
        """
        self._feature_matrix = feature_matrix
        self._destinations_df = destinations_df.reset_index(drop=True)
        self.model.fit(feature_matrix)
        self._is_fitted = True
        logger.info("KNN fitted với %d địa điểm, metric=%s, k=%d",
                    len(destinations_df), self.config.metric, self.config.k)
        return self

    def recommend_similar(
        self,
        destination_id: str,
        k: int | None = None,
    ) -> list[Recommendation]:
        """
        Gợi ý địa điểm tương tự với một địa điểm đã chọn.

        Args:
            destination_id: ID của địa điểm tham chiếu
            k: Số lượng gợi ý (mặc định dùng config.k)

        Returns:
            Danh sách Recommendation, sắp xếp theo độ tương tự giảm dần
        """
        self._check_fitted()
        k = k or self.config.k

        idx = self._get_index(destination_id)
        query_vector = self._feature_matrix[idx].reshape(1, -1)
        distances, indices = self.model.kneighbors(query_vector, n_neighbors=k + 1)

        results = []
        for rank, (dist, neighbor_idx) in enumerate(
            zip(distances[0][1:], indices[0][1:]), start=1  # Bỏ qua chính nó
        ):
            row = self._destinations_df.iloc[neighbor_idx]
            similarity = 1.0 - dist if self.config.metric == "cosine" else 1.0 / (1.0 + dist)
            results.append(Recommendation(
                destination_id=str(row["destination_id"]),
                name=str(row["name"]),
                similarity_score=round(similarity, 4),
                rank=rank,
            ))

        return results

    def recommend_for_user(
        self,
        user_ratings: dict[str, float],
        k: int | None = None,
    ) -> list[Recommendation]:
        """
        Gợi ý địa điểm dựa trên lịch sử đánh giá của người dùng.

        Tạo user profile = weighted average của các địa điểm đã thích.

        Args:
            user_ratings: {destination_id: rating} - các địa điểm user đã đánh giá
            k: Số lượng gợi ý

        Returns:
            Danh sách Recommendation
        """
        self._check_fitted()
        k = k or self.config.k

        # Tạo user profile vector từ weighted average
        visited_ids = set(user_ratings.keys())
        user_vector = self._build_user_profile(user_ratings)

        distances, indices = self.model.kneighbors(user_vector.reshape(1, -1), n_neighbors=k + len(visited_ids))

        results = []
        rank = 1
        for dist, idx in zip(distances[0], indices[0]):
            row = self._destinations_df.iloc[idx]
            dest_id = str(row["destination_id"])

            if dest_id in visited_ids:
                continue  # Bỏ qua nơi đã đến

            similarity = 1.0 - dist if self.config.metric == "cosine" else 1.0 / (1.0 + dist)
            results.append(Recommendation(
                destination_id=dest_id,
                name=str(row["name"]),
                similarity_score=round(similarity, 4),
                rank=rank,
            ))
            rank += 1
            if rank > k:
                break

        return results

    def _build_user_profile(self, user_ratings: dict[str, float]) -> np.ndarray:
        """Tính user profile vector = trung bình có trọng số."""
        vectors, weights = [], []
        for dest_id, rating in user_ratings.items():
            try:
                idx = self._get_index(dest_id)
                vectors.append(self._feature_matrix[idx])
                weights.append(rating)
            except ValueError:
                logger.warning("Không tìm thấy destination_id: %s, bỏ qua.", dest_id)
        
        if not vectors:
            raise ValueError("Không có địa điểm hợp lệ trong user_ratings")

        weights_arr = np.array(weights) / sum(weights)
        return np.average(np.vstack(vectors), axis=0, weights=weights_arr)

    def _get_index(self, destination_id: str) -> int:
        """Lấy chỉ số hàng trong DataFrame theo destination_id."""
        matches = self._destinations_df[
            self._destinations_df["destination_id"] == destination_id
        ].index.tolist()
        if not matches:
            raise ValueError(f"Không tìm thấy destination_id: {destination_id}")
        return matches[0]

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Gọi fit() trước khi sử dụng recommender")
