"""
Evaluation Metrics - Đánh giá hiệu quả hệ thống gợi ý.

Các chỉ số được implement:
    - Precision@K
    - Recall@K
    - F1@K
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - MRR (Mean Reciprocal Rank)
    - Coverage (% catalog được gợi ý)
"""

import math
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Precision@K = |recommended[:K] ∩ relevant| / K

    Args:
        recommended: Danh sách ID được gợi ý (theo thứ tự)
        relevant: Tập hợp ID thực sự relevant
        k: Ngưỡng đánh giá

    Returns:
        Giá trị từ 0.0 đến 1.0
    """
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Recall@K = |recommended[:K] ∩ relevant| / |relevant|

    Args:
        recommended: Danh sách ID được gợi ý
        relevant: Tập hợp ID thực sự relevant
        k: Ngưỡng đánh giá

    Returns:
        Giá trị từ 0.0 đến 1.0
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list, relevant: set, k: int) -> float:
    """F1@K = harmonic mean của Precision@K và Recall@K."""
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K

    DCG@K = Σ rel_i / log2(i+1) với i từ 1 đến K
    IDCG = DCG của thứ tự gợi ý lý tưởng (relevant items đầu tiên)

    Args:
        recommended: Danh sách ID được gợi ý
        relevant: Tập hợp ID thực sự relevant
        k: Ngưỡng đánh giá

    Returns:
        Giá trị từ 0.0 đến 1.0
    """
    if not relevant or k <= 0:
        return 0.0

    # DCG thực tế
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # IDCG: lý tưởng - relevant items xếp đầu
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant_in_k + 1))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommendations: list[list], relevant_sets: list[set]) -> float:
    """
    Mean Reciprocal Rank = (1/|Q|) Σ 1/rank_first_relevant

    Args:
        recommendations: Danh sách các list gợi ý (mỗi user)
        relevant_sets: Danh sách các tập relevant tương ứng

    Returns:
        Giá trị MRR
    """
    rr_sum = 0.0
    for rec_list, relevant in zip(recommendations, relevant_sets):
        for rank, item in enumerate(rec_list, start=1):
            if item in relevant:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(recommendations) if recommendations else 0.0


def evaluate_recommender(
    recommendations: dict[str, list],  # {user_id: [dest_id, ...]}
    ground_truth: dict[str, set],       # {user_id: {dest_id, ...}}
    k_values: list[int] = (5, 10, 20),
) -> dict[str, Any]:
    """
    Đánh giá toàn bộ hệ thống gợi ý trên tập test.

    Args:
        recommendations: Dict user_id → danh sách gợi ý
        ground_truth: Dict user_id → tập địa điểm relevant
        k_values: Danh sách các giá trị K cần đánh giá

    Returns:
        Dict kết quả với Precision, Recall, F1, NDCG tại từng K + MRR
    """
    user_ids = [uid for uid in recommendations if uid in ground_truth]
    if not user_ids:
        logger.warning("Không có user nào để đánh giá!")
        return {}

    results: dict = {f"@{k}": {} for k in k_values}

    for k in k_values:
        precisions, recalls, f1s, ndcgs = [], [], [], []
        for uid in user_ids:
            rec = recommendations[uid]
            rel = ground_truth[uid]
            precisions.append(precision_at_k(rec, rel, k))
            recalls.append(recall_at_k(rec, rel, k))
            f1s.append(f1_at_k(rec, rel, k))
            ndcgs.append(ndcg_at_k(rec, rel, k))

        results[f"@{k}"] = {
            "Precision": round(np.mean(precisions), 4),
            "Recall": round(np.mean(recalls), 4),
            "F1": round(np.mean(f1s), 4),
            "NDCG": round(np.mean(ndcgs), 4),
        }

    # MRR
    rec_lists = [recommendations[uid] for uid in user_ids]
    rel_sets = [ground_truth[uid] for uid in user_ids]
    results["MRR"] = round(mrr(rec_lists, rel_sets), 4)

    logger.info("Đánh giá hoàn tất trên %d users", len(user_ids))
    return results
