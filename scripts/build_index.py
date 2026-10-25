"""
Script xây dựng FAISS vector index từ knowledge base documents.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --kb-dir data/raw/knowledge_base --output data/embeddings
"""

import argparse
from pathlib import Path

# Thêm project root vào Python path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.embedder import TextEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Chia văn bản thành các đoạn nhỏ có overlap."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def main(kb_dir: Path, output_dir: Path) -> None:
    config = load_config()
    embedder = TextEmbedder(config.rag)

    # Load tài liệu
    docs = []
    for txt_file in kb_dir.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(text, config.rag.chunk_size, config.rag.chunk_overlap)
        for chunk in chunks:
            docs.append({"source": txt_file.name, "content": chunk})
        logger.info("Đã chunk %s: %d chunks", txt_file.name, len(chunks))

    if not docs:
        logger.warning("Không tìm thấy tài liệu nào trong %s", kb_dir)
        return

    # Tạo embeddings
    texts = [d["content"] for d in docs]
    embeddings = embedder.embed_texts(texts)

    # Build và lưu index
    store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    store.add_documents(docs, embeddings)
    store.save(output_dir)

    logger.info("✅ Build index thành công! %d chunks → %s", len(docs), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vector index")
    parser.add_argument("--kb-dir", type=Path, default=Path("data/raw/knowledge_base"))
    parser.add_argument("--output", type=Path, default=Path("data/embeddings"))
    args = parser.parse_args()

    main(args.kb_dir, args.output)
