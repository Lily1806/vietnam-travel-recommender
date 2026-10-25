"""
Config loader - đọc cấu hình từ config.yaml và biến môi trường.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    raw_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    embeddings_dir: Path = BASE_DIR / "data" / "embeddings"
    knowledge_base_dir: Path = BASE_DIR / "data" / "raw" / "knowledge_base"


@dataclass
class KNNConfig:
    k: int = 10
    metric: str = "cosine"          # cosine | euclidean | manhattan
    algorithm: str = "brute"        # brute | ball_tree | kd_tree | auto
    n_jobs: int = -1


@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "openai"    # openai | gemini | ollama
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5
    vector_db: str = "faiss"        # faiss | chroma
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    log_level: str = "INFO"


def load_config(config_path: Path | None = None) -> AppConfig:
    """
    Load cấu hình từ file YAML, override bằng biến môi trường.

    Args:
        config_path: Đường dẫn tới file config.yaml (tùy chọn)

    Returns:
        AppConfig instance
    """
    config = AppConfig()

    if config_path is None:
        config_path = BASE_DIR / "config" / "config.yaml"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "knn" in raw:
            for k, v in raw["knn"].items():
                if hasattr(config.knn, k):
                    setattr(config.knn, k, v)

        if "rag" in raw:
            for k, v in raw["rag"].items():
                if hasattr(config.rag, k):
                    setattr(config.rag, k, v)

        config.log_level = raw.get("log_level", config.log_level)

    return config


# Singleton config instance
config = load_config()
