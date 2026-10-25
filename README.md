# 🗺️ Ứng dụng KNN và RAG trong Hệ thống Gợi ý và Tư vấn Du lịch Thông minh

> **Môn học:** Khai phá Dữ liệu | **Nhóm:** [Tên nhóm] | **GVHD:** [Tên giảng viên]

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

---

## 📌 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Kiến thức nền tảng cần nắm](#-kiến-thức-nền-tảng-cần-nắm)
- [Chuẩn bị môi trường](#-chuẩn-bị-môi-trường)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Hướng dẫn chạy](#-hướng-dẫn-chạy)
- [Kết quả & Đánh giá](#-kết-quả--đánh-giá)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Giới thiệu

### Bài toán
Hệ thống gợi ý du lịch thông minh kết hợp hai kỹ thuật mạnh mẽ:

| Thành phần | Kỹ thuật | Vai trò |
|-----------|----------|---------|
| **Gợi ý địa điểm** | KNN (K-Nearest Neighbors) | Tìm địa điểm tương đồng dựa trên đặc điểm và sở thích người dùng |
| **Tư vấn thông minh** | RAG (Retrieval-Augmented Generation) | Trả lời câu hỏi tự nhiên về địa điểm, thời tiết, văn hóa |

### Mục tiêu
- ✅ Xây dựng pipeline gợi ý địa điểm du lịch cá nhân hóa bằng KNN
- ✅ Tích hợp RAG để tư vấn thông tin du lịch qua ngôn ngữ tự nhiên
- ✅ Đánh giá hiệu quả mô hình bằng các chỉ số chuẩn (Precision@K, NDCG, MRR)
- ✅ Xây dựng API phục vụ demo thực tế

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│              (CLI Demo / FastAPI / Streamlit)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
┌─────────────────┐                ┌──────────────────┐
│  KNN MODULE     │                │   RAG MODULE     │
│                 │                │                  │
│ • Feature Eng   │                │ • Embedding      │
│ • Similarity    │                │ • Vector Store   │
│ • Top-K Recom   │                │ • LLM Generation │
└────────┬────────┘                └────────┬─────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│    Raw Data → Preprocessing → Feature Store / Vector DB      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Kiến thức nền tảng cần nắm

### 1. Machine Learning cơ bản
- **KNN Algorithm**: Nguyên lý hoạt động, các độ đo khoảng cách (Euclidean, Cosine, Manhattan)
- **Feature Engineering**: Chuẩn hóa dữ liệu (Min-Max, Z-score), xử lý biến categorical (One-Hot, Label Encoding)
- **Curse of Dimensionality**: Tại sao cần giảm chiều (PCA, t-SNE) khi dùng KNN

### 2. NLP & Embeddings
- **Word Embeddings**: Word2Vec, GloVe, FastText
- **Sentence Embeddings**: Sentence-BERT, all-MiniLM
- **Vector Similarity**: Cosine similarity, Dot product

### 3. RAG (Retrieval-Augmented Generation)
- **Vector Database**: FAISS, ChromaDB, Pinecone
- **Chunking Strategy**: Cách chia nhỏ tài liệu
- **LLM Integration**: OpenAI API / Gemini API / Ollama (local)
- **Prompt Engineering**: Cách viết prompt hiệu quả

### 4. Đánh giá hệ thống gợi ý
- **Precision@K**: Tỷ lệ gợi ý đúng trong top K
- **Recall@K**: Tỷ lệ bao phủ trong top K
- **NDCG**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

### 5. Kỹ năng lập trình
- Python: NumPy, Pandas, Scikit-learn
- Vector DB: FAISS hoặc ChromaDB
- API: FastAPI
- Notebook: Jupyter

---

## ⚙️ Chuẩn bị môi trường

### Yêu cầu hệ thống
- Python >= 3.10
- RAM >= 8GB (khuyến nghị 16GB)
- Disk >= 5GB (cho embeddings)

### Cài đặt

```bash
# 1. Clone project
git clone <repo-url>
cd KhaiThacDuLieu

# 2. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Tạo file cấu hình
cp config/config.example.yaml config/config.yaml
# Chỉnh sửa API keys trong config.yaml
```

### Biến môi trường
```bash
cp .env.example .env
```
```ini
# .env
OPENAI_API_KEY=sk-...
VECTOR_DB_PATH=./data/embeddings
LOG_LEVEL=INFO
```

---

## 📁 Cấu trúc dự án

```
KhaiThacDuLieu/
│
├── 📂 data/                        # Toàn bộ dữ liệu
│   ├── raw/                        # Dữ liệu thô (chưa xử lý)
│   │   ├── destinations.csv        # Danh sách địa điểm du lịch
│   │   ├── reviews.csv             # Đánh giá của người dùng
│   │   └── knowledge_base/         # Tài liệu cho RAG (txt, pdf)
│   ├── processed/                  # Dữ liệu đã làm sạch
│   └── embeddings/                 # Vector embeddings đã tính
│
├── 📂 src/                         # Source code chính
│   ├── __init__.py
│   ├── 📂 data/                    # Data layer
│   │   ├── __init__.py
│   │   ├── loader.py               # Load & validate dữ liệu
│   │   └── preprocessor.py         # Tiền xử lý dữ liệu
│   │
│   ├── 📂 models/                  # Model layer
│   │   ├── __init__.py
│   │   └── knn_recommender.py      # KNN model
│   │
│   ├── 📂 retrieval/               # RAG retrieval layer
│   │   ├── __init__.py
│   │   ├── embedder.py             # Tạo embeddings
│   │   ├── vector_store.py         # Quản lý vector database
│   │   └── retriever.py            # Tìm kiếm tài liệu liên quan
│   │
│   ├── 📂 recommendation/          # Recommendation layer
│   │   ├── __init__.py
│   │   ├── engine.py               # Pipeline gợi ý chính
│   │   └── advisor.py              # RAG-based travel advisor
│   │
│   ├── 📂 evaluation/              # Đánh giá mô hình
│   │   ├── __init__.py
│   │   └── metrics.py              # Precision@K, NDCG, MRR
│   │
│   └── 📂 utils/                   # Tiện ích dùng chung
│       ├── __init__.py
│       ├── logger.py               # Logging chuẩn hóa
│       └── config.py               # Đọc cấu hình
│
├── 📂 notebooks/                   # Jupyter Notebooks
│   ├── 01_EDA.ipynb                # Phân tích khám phá dữ liệu
│   ├── 02_KNN_Experiments.ipynb    # Thử nghiệm KNN
│   ├── 03_RAG_Pipeline.ipynb       # Xây dựng RAG pipeline
│   └── 04_Evaluation.ipynb         # Đánh giá & so sánh
│
├── 📂 api/                         # REST API (FastAPI)
│   ├── __init__.py
│   ├── main.py                     # Entry point API
│   ├── routers/
│   │   ├── recommend.py            # /recommend endpoint
│   │   └── advise.py               # /advise endpoint
│   └── schemas.py                  # Request/Response models
│
├── 📂 tests/                       # Kiểm thử
│   ├── unit/
│   └── integration/
│
├── 📂 docs/                        # Tài liệu mở rộng
│   ├── architecture.md
│   ├── data_description.md
│   └── experiments.md
│
├── 📂 scripts/                     # Script tiện ích
│   ├── download_data.py
│   ├── build_index.py
│   └── evaluate.py
│
├── 📂 config/
│   ├── config.example.yaml
│   └── config.yaml                 # (gitignore)
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Hướng dẫn chạy

### Bước 1: Chuẩn bị dữ liệu
```bash
python scripts/download_data.py
```

### Bước 2: Xây dựng Vector Index (cho RAG)
```bash
python scripts/build_index.py
```

### Bước 3: Chạy demo qua CLI
```bash
python -m src.recommendation.engine --user-id 123
```

### Bước 4: Chạy API Server
```bash
uvicorn api.main:app --reload --port 8000
# Truy cập docs: http://localhost:8000/docs
```

### Bước 5: Đánh giá mô hình
```bash
python scripts/evaluate.py --model knn --k 5 10 20
```

---

## 📊 Kết quả & Đánh giá

### Chỉ số đánh giá KNN

| K | Precision@K | Recall@K | NDCG@K |
|---|------------|---------|--------|
| 5 | - | - | - |
| 10 | - | - | - |
| 20 | - | - | - |

> *Bảng sẽ được cập nhật sau khi chạy thực nghiệm*

### So sánh Similarity Metrics

| Metric | Precision@5 | Thời gian (ms) |
|--------|------------|---------------|
| Cosine | - | - |
| Euclidean | - | - |
| Manhattan | - | - |

---

## 🗂️ Tập dữ liệu

| Dataset | Mô tả | Nguồn |
|---------|-------|-------|
| Vietnam Travel Destinations | 500+ địa điểm du lịch VN | Tự thu thập / Kaggle |
| User Reviews | Đánh giá & rating | TripAdvisor / Google |
| Knowledge Base | Thông tin mô tả địa điểm | Wikipedia / Blogs |

---

## 📖 Tài liệu tham khảo

### Papers
- Cover, T., & Hart, P. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.

### Thư viện chính
- [scikit-learn](https://scikit-learn.org/) - KNN implementation
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [LangChain](https://langchain.com/) - RAG framework
- [Sentence-Transformers](https://sbert.net/) - Text embeddings
- [FastAPI](https://fastapi.tiangolo.com/) - API framework

### Tài liệu học
- [CS229 Machine Learning - Stanford](https://cs229.stanford.edu/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Recommender Systems Specialization - Coursera](https://www.coursera.org/specializations/recommender-systems)

---

## 👥 Thành viên nhóm

| Họ tên | MSSV | Vai trò |
|--------|------|---------|
| [Tên] | [MSSV] | Data Collection & KNN |
| [Tên] | [MSSV] | RAG Pipeline & LLM |
| [Tên] | [MSSV] | Evaluation & API |

---

<div align="center">
  <i>📌 Dự án thực hiện cho môn Khai phá Dữ liệu - Học kỳ [HK] năm [Năm]</i>
</div>
