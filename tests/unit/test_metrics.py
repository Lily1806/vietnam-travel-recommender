"""
Unit tests cho Evaluation Metrics.
"""

import pytest

from src.evaluation.metrics import (
    evaluate_recommender,
    f1_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["A", "B", "C"], {"A", "B", "C"}, k=3) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["X", "Y", "Z"], {"A", "B"}, k=3) == 0.0

    def test_partial_precision(self):
        # 1 trong 2 đúng → 0.5
        assert precision_at_k(["A", "X"], {"A", "B"}, k=2) == 0.5

    def test_k_greater_than_list(self):
        # Chỉ có 1 item nhưng k=5 → 1/5 = 0.2
        result = precision_at_k(["A"], {"A"}, k=5)
        assert result == pytest.approx(0.2)

    def test_zero_k_returns_zero(self):
        assert precision_at_k(["A"], {"A"}, k=0) == 0.0

    def test_only_top_k_considered(self):
        # Item thứ 4 đúng nhưng k=3 → không tính
        assert precision_at_k(["X", "Y", "Z", "A"], {"A"}, k=3) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["A", "B"], {"A", "B"}, k=2) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["X", "Y"], {"A", "B"}, k=2) == 0.0

    def test_partial_recall(self):
        # Tìm được 1/2 relevant → 0.5
        assert recall_at_k(["A", "X"], {"A", "B"}, k=2) == 0.5

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["A"], set(), k=3) == 0.0


class TestNDCGAtK:
    def test_perfect_ndcg(self):
        # Relevant items xếp đầu → NDCG = 1.0
        assert ndcg_at_k(["A", "B", "C"], {"A", "B", "C"}, k=3) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert ndcg_at_k(["X", "Y"], {"A", "B"}, k=2) == 0.0

    def test_ndcg_penalizes_lower_rank(self):
        # A ở vị trí 1 vs A ở vị trí 3 → score cao hơn
        score_rank1 = ndcg_at_k(["A", "X", "Y"], {"A"}, k=3)
        score_rank3 = ndcg_at_k(["X", "Y", "A"], {"A"}, k=3)
        assert score_rank1 > score_rank3

    def test_empty_relevant_returns_zero(self):
        assert ndcg_at_k(["A"], set(), k=3) == 0.0


class TestMRR:
    def test_perfect_mrr(self):
        # Cả 2 users đều có item đúng ở rank 1
        recs = [["A", "B"], ["C", "D"]]
        rels = [{"A"}, {"C"}]
        assert mrr(recs, rels) == pytest.approx(1.0)

    def test_mrr_with_mixed_ranks(self):
        # User 1: rank 1 (RR=1.0), User 2: rank 2 (RR=0.5) → MRR = 0.75
        recs = [["A", "B"], ["X", "C"]]
        rels = [{"A"}, {"C"}]
        assert mrr(recs, rels) == pytest.approx(0.75)

    def test_zero_mrr(self):
        recs = [["X", "Y"], ["P", "Q"]]
        rels = [{"A"}, {"B"}]
        assert mrr(recs, rels) == 0.0

    def test_empty_returns_zero(self):
        assert mrr([], []) == 0.0


class TestEvaluateRecommender:
    def test_returns_all_k_keys(self):
        recs = {"u1": ["A", "B", "C"]}
        gt = {"u1": {"A"}}
        results = evaluate_recommender(recs, gt, k_values=[3, 5])
        assert "@3" in results
        assert "@5" in results
        assert "MRR" in results

    def test_perfect_evaluation(self):
        recs = {"u1": ["A", "B"], "u2": ["C", "D"]}
        gt = {"u1": {"A"}, "u2": {"C"}}
        results = evaluate_recommender(recs, gt, k_values=[1])
        assert results["@1"]["Precision"] == pytest.approx(1.0)
        assert results["MRR"] == pytest.approx(1.0)

    def test_ignores_users_not_in_ground_truth(self):
        recs = {"u1": ["A"], "u2": ["B"]}
        gt = {"u1": {"A"}}  # u2 không có ground truth
        results = evaluate_recommender(recs, gt, k_values=[1])
        # Chỉ đánh giá u1 → Precision = 1.0
        assert results["@1"]["Precision"] == pytest.approx(1.0)

    def test_empty_recommendations_returns_empty(self):
        results = evaluate_recommender({}, {}, k_values=[5])
        assert results == {}
