"""
Logging utility - chuẩn hóa logging toàn dự án.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Tạo và trả về một logger với format chuẩn.

    Args:
        name: Tên module (thường dùng __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Logger instance đã được cấu hình
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Tránh thêm handler trùng lặp

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
