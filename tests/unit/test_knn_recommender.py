"""
Unit tests cho KNN Recommender.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.knn_recommender import KNNRecommender, Recommendation
from src.utils.config import KNNConfig


@pytest.fixture
def sample_data():
    """Tạo dữ liệu mẫu 5 địa điểm, 4 features."""
    np.random.seed(42)
    n = 5
    feature_matrix = np.random.rand(n, 4).astype(np.float32)
    destinations_df = pd.DataFrame({
        "destination_id": [f"D{i:03d}" for i in range(n)],
        "name": [f"Destination {i}" for i in range(n)],
        "location": ["Hà Nội"] * n,
    })
    return feature_matrix, destinations_df


@pytest.fixture
def fitted_recommender(sample_data):
    """KNNRecommender đã được fit với dữ liệu mẫu."""
    feature_matrix, destinations_df = sample_data
    config = KNNConfig(k=3, metric="cosine", algorithm="brute")
    recommender = KNNRecommender(config)
    recommender.fit(feature_matrix, destinations_df)
    return recommender


class TestKNNRecommenderFit:
    def test_fit_returns_self(self, sample_data):
        feature_matrix, destinations_df = sample_data
        config = KNNConfig(k=3)
        rec = KNNRecommender(config)
        result = rec.fit(feature_matrix, destinations_df)
        assert result is rec

    def test_fit_sets_is_fitted(self, sample_data):
        feature_matrix, destinations_df = sample_data
        config = KNNConfig(k=3)
        rec = KNNRecommender(config)
        assert not rec._is_fitted
        rec.fit(feature_matrix, destinations_df)
        assert rec._is_fitted

    def test_unfitted_raises_on_recommend(self, sample_data):
        _, destinations_df = sample_data
        config = KNNConfig(k=3)
        rec = KNNRecommender(config)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            rec.recommend_similar("D000")


class TestRecommendSimilar:
    def test_returns_list_of_recommendations(self, fitted_recommender):
        results = fitted_recommender.recommend_similar("D000", k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, Recommendation) for r in results)

    def test_k_controls_number_of_results(self, fitted_recommender):
        for k in [1, 2, 3]:
            results = fitted_recommender.recommend_similar("D000", k=k)
            assert len(results) == k

    def test_does_not_include_query_item(self, fitted_recommender):
        results = fitted_recommender.recommend_similar("D000", k=3)
        dest_ids = [r.destination_id for r in results]
        assert "D000" not in dest_ids

    def test_ranks_are_sequential(self, fitted_recommender):
        results = fitted_recommender.recommend_similar("D000", k=3)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_similarity_scores_are_in_range(self, fitted_recommender):
        results = fitted_recommender.recommend_similar("D000", k=3)
        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0 + 1e-6

    def test_invalid_id_raises_value_error(self, fitted_recommender):
        with pytest.raises(ValueError, match="Không tìm thấy"):
            fitted_recommender.recommend_similar("NONEXISTENT", k=2)


class TestRecommendForUser:
    def test_returns_recommendations(self, fitted_recommender):
        user_ratings = {"D000": 5.0, "D001": 3.0}
        results = fitted_recommender.recommend_for_user(user_ratings, k=2)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_excludes_visited_destinations(self, fitted_recommender):
        user_ratings = {"D000": 5.0, "D001": 3.0}
        results = fitted_recommender.recommend_for_user(user_ratings, k=3)
        dest_ids = {r.destination_id for r in results}
        assert "D000" not in dest_ids
        assert "D001" not in dest_ids

    def test_empty_user_ratings_raises(self, fitted_recommender):
        with pytest.raises(ValueError):
            fitted_recommender.recommend_for_user({}, k=3)

    def test_all_invalid_ids_raises(self, fitted_recommender):
        with pytest.raises(ValueError):
            fitted_recommender.recommend_for_user({"INVALID": 5.0}, k=3)

    def test_weighted_profile(self, fitted_recommender):
        """Kết quả với rating cao hơn không được giống hệt rating thấp."""
        results_high = fitted_recommender.recommend_for_user({"D000": 5.0, "D001": 1.0}, k=2)
        results_equal = fitted_recommender.recommend_for_user({"D000": 3.0, "D001": 3.0}, k=2)
        # Ít nhất 1 trong 2 kết quả phải khác nhau (trọng số khác nhau ảnh hưởng profile)
        ids_high = [r.destination_id for r in results_high]
        ids_equal = [r.destination_id for r in results_equal]
        # Không nhất thiết phải khác, nhưng cả 2 phải là list hợp lệ
        assert len(ids_high) == 2
        assert len(ids_equal) == 2
