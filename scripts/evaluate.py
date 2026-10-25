"""
Script đánh giá KNN Recommender trên tập test thực tế từ reviews.csv.

Chiến lược đánh giá:
    - Chia user ratings theo tỉ lệ 80/20 (train/test)
    - Train: Dùng làm lịch sử người dùng (input cho KNN)
    - Test : Dùng làm ground truth (địa điểm cần được gợi ý đúng)
    - Chỉ đánh giá user có >= 3 ratings để đảm bảo đủ data train/test

Metrics:
    - Precision@K : Tỷ lệ địa điểm đúng trong top-K gợi ý
    - Recall@K    : Tỷ lệ địa điểm đúng được tìm thấy trong top-K
    - NDCG@K      : Normalized Discounted Cumulative Gain (xét thứ hạng)
    - MRR         : Mean Reciprocal Rank (vị trí đúng đầu tiên)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --k 5 10 20 --min-ratings 3 --test-ratio 0.2
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.data.loader import DataLoader
from src.data.preprocessor import DestinationPreprocessor
from src.evaluation.metrics import evaluate_recommender
from src.models.knn_recommender import KNNRecommender
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_user_ratings(
    user_ratings: dict[str, float],
    test_ratio: float = 0.2,
    min_test: int = 1,
) -> tuple[dict[str, float], set[str]]:
    """
    Chia ratings của 1 user thành train/test.

    Args:
        user_ratings: {destination_id: rating}
        test_ratio: Tỉ lệ dành cho tập test
        min_test: Số lượng test item tối thiểu

    Returns:
        (train_ratings, test_ids)
    """
    items = list(user_ratings.items())
    random.shuffle(items)
    n_test = max(min_test, int(len(items) * test_ratio))
    n_test = min(n_test, len(items) - 1)  # Đảm bảo train không rỗng

    test_items = items[:n_test]
    train_items = items[n_test:]

    train_ratings = dict(train_items)
    test_ids = {dest_id for dest_id, _ in test_items}
    return train_ratings, test_ids


def load_user_ratings(reviews_df: pd.DataFrame, min_ratings: int = 3) -> dict[str, dict[str, float]]:
    """
    Tổng hợp ratings từ reviews.csv theo user.

    Args:
        reviews_df: DataFrame reviews.csv
        min_ratings: Loại bỏ user có ít hơn min_ratings

    Returns:
        {user_id: {destination_id: rating}}
    """
    user_ratings: dict[str, dict[str, float]] = {}
    for _, row in reviews_df.iterrows():
        uid = str(row["user_id"])
        did = str(row["destination_id"])
        rating = float(row["rating"])
        user_ratings.setdefault(uid, {})[did] = rating

    # Lọc user có đủ ratings
    filtered = {uid: r for uid, r in user_ratings.items() if len(r) >= min_ratings}
    logger.info(
        "Tổng %d users, sau khi lọc (>= %d ratings): %d users",
        len(user_ratings), min_ratings, len(filtered),
    )
    return filtered


def main(k_values: list[int], min_ratings: int, test_ratio: float, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    config = load_config()
    loader = DataLoader(config.data.raw_dir)

    # ── Load dữ liệu ─────────────────────────────────────────────────────────
    destinations_df = loader.load_destinations()
    reviews_df = loader.load_reviews()

    valid_dest_ids = set(destinations_df["destination_id"].astype(str))
    logger.info("Tổng %d địa điểm, %d reviews", len(destinations_df), len(reviews_df))

    # ── Huấn luyện KNN trên toàn bộ destinations ─────────────────────────────
    preprocessor = DestinationPreprocessor()
    feature_matrix = preprocessor.fit_transform(destinations_df)
    recommender = KNNRecommender(config.knn)
    recommender.fit(feature_matrix, destinations_df)

    # ── Chia train/test theo user ─────────────────────────────────────────────
    all_user_ratings = load_user_ratings(reviews_df, min_ratings=min_ratings)

    if not all_user_ratings:
        logger.error("Không có đủ dữ liệu user để đánh giá (yêu cầu >= %d ratings/user).", min_ratings)
        return

    recommendations: dict[str, list[str]] = {}
    ground_truth: dict[str, set[str]] = {}

    skipped = 0
    for user_id, ratings in all_user_ratings.items():
        # Chỉ giữ ratings có destination_id hợp lệ
        valid_ratings = {d: r for d, r in ratings.items() if d in valid_dest_ids}
        if len(valid_ratings) < min_ratings:
            skipped += 1
            continue

        train_ratings, test_ids = split_user_ratings(valid_ratings, test_ratio=test_ratio)
        if not train_ratings or not test_ids:
            skipped += 1
            continue

        try:
            max_k = max(k_values)
            recs = recommender.recommend_for_user(train_ratings, k=max_k)
            recommendations[user_id] = [r.destination_id for r in recs]
            ground_truth[user_id] = test_ids
        except Exception as e:
            logger.debug("Bỏ qua user %s: %s", user_id, e)
            skipped += 1

    logger.info(
        "Đánh giá %d users (bỏ qua %d users không hợp lệ)",
        len(recommendations), skipped,
    )

    if not recommendations:
        logger.error("Không có user nào để đánh giá!")
        return

    # ── Tính metrics ─────────────────────────────────────────────────────────
    results = evaluate_recommender(recommendations, ground_truth, k_values=k_values)

    # ── In kết quả ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊  KẾT QUẢ ĐÁNH GIÁ HỆ THỐNG GỢI Ý DU LỊCH (KNN)")
    print("=" * 65)
    print(f"  Số users đánh giá : {len(recommendations)}")
    print(f"  Min ratings/user  : {min_ratings}")
    print(f"  Test ratio        : {test_ratio}")
    print(f"  Seed              : {seed}")
    print("-" * 65)

    for k in k_values:
        key = f"@{k}"
        if key in results:
            print(f"\n  ── Top-{k} ──")
            for metric, value in results[key].items():
                bar = "█" * int(value * 20)
                print(f"    {metric:<15s}: {value:.4f}  {bar}")

    print(f"\n  {'MRR':<15s}: {results.get('MRR', 0):.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KNN Recommender")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20],
                        help="Danh sách giá trị K cần đánh giá")
    parser.add_argument("--min-ratings", type=int, default=3,
                        help="Số lượng ratings tối thiểu mỗi user (default: 3)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Tỉ lệ tập test (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    main(
        k_values=args.k,
        min_ratings=args.min_ratings,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
