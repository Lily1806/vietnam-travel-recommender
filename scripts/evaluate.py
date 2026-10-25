"""
Script đánh giá KNN Recommender trên tập test.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --k 5 10 20
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate_recommender
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main(k_values: list[int]) -> None:
    logger.info("Bắt đầu đánh giá với K = %s", k_values)

    # TODO: Load model và dữ liệu test thực tế
    # Ví dụ giả lập:
    recommendations = {
        "user_001": ["dest_A", "dest_B", "dest_C", "dest_D", "dest_E"],
        "user_002": ["dest_F", "dest_A", "dest_G", "dest_H", "dest_I"],
    }
    ground_truth = {
        "user_001": {"dest_A", "dest_C"},
        "user_002": {"dest_F", "dest_I"},
    }

    results = evaluate_recommender(recommendations, ground_truth, k_values=k_values)

    # In kết quả dạng bảng
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ HỆ THỐNG GỢI Ý")
    print("=" * 60)

    for k in k_values:
        key = f"@{k}"
        if key in results:
            print(f"\n  Top-{k}:")
            for metric, value in results[key].items():
                print(f"    {metric:12s}: {value:.4f}")

    print(f"\n  MRR: {results.get('MRR', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KNN Recommender")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="K values")
    args = parser.parse_args()
    main(args.k)
