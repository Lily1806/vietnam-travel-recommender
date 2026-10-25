"""
Integration tests cho FastAPI endpoints.

Dùng TestClient của FastAPI để kiểm tra toàn bộ request/response flow
mà không cần chạy server thực.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Khởi tạo TestClient một lần cho toàn module (nạp model 1 lần)."""
    with TestClient(app) as c:
        yield c


# ─── Health Check ─────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "components" in data

    def test_health_reports_knn_status(self, client):
        response = client.get("/health")
        components = response.json()["components"]
        assert "knn_recommender" in components

    def test_health_reports_rag_status(self, client):
        response = client.get("/health")
        components = response.json()["components"]
        assert "rag_retriever" in components


# ─── KNN: List Destinations ───────────────────────────────────────────────────

class TestListDestinations:
    def test_returns_destination_list(self, client):
        response = client.get("/api/v1/recommend/destinations")
        assert response.status_code == 200
        data = response.json()
        assert "destinations" in data
        assert "total" in data
        assert data["total"] > 0

    def test_each_destination_has_required_fields(self, client):
        response = client.get("/api/v1/recommend/destinations")
        destinations = response.json()["destinations"]
        for dest in destinations[:5]:  # Kiểm tra 5 đầu tiên
            assert "destination_id" in dest
            assert "name" in dest


# ─── KNN: Recommend Similar ───────────────────────────────────────────────────

class TestRecommendSimilar:
    def test_valid_request_returns_200(self, client):
        response = client.post(
            "/api/v1/recommend/similar",
            json={"destination_id": "VN001", "k": 3},
        )
        assert response.status_code == 200

    def test_response_structure(self, client):
        response = client.post(
            "/api/v1/recommend/similar",
            json={"destination_id": "VN001", "k": 3},
        )
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total" in data

    def test_returns_correct_number_of_results(self, client):
        k = 4
        response = client.post(
            "/api/v1/recommend/similar",
            json={"destination_id": "VN001", "k": k},
        )
        data = response.json()
        assert len(data["results"]) == k
        assert data["total"] == k

    def test_result_items_have_required_fields(self, client):
        response = client.post(
            "/api/v1/recommend/similar",
            json={"destination_id": "VN001", "k": 2},
        )
        for item in response.json()["results"]:
            assert "destination_id" in item
            assert "name" in item
            assert "similarity_score" in item
            assert "rank" in item

    def test_invalid_destination_id_returns_error(self, client):
        response = client.post(
            "/api/v1/recommend/similar",
            json={"destination_id": "INVALID_ID_XYZ", "k": 3},
        )
        assert response.status_code in (404, 422, 500)

    def test_k_validation_too_large(self, client):
        response = client.post(
            "/api/v1/recommend/similar",
            json={"destination_id": "VN001", "k": 9999},
        )
        assert response.status_code == 422  # Pydantic validation


# ─── KNN: Recommend For User ──────────────────────────────────────────────────

class TestRecommendForUser:
    def test_valid_request_returns_200(self, client):
        response = client.post(
            "/api/v1/recommend/for-user",
            json={"user_ratings": {"VN001": 5.0, "VN002": 4.0}, "k": 5},
        )
        assert response.status_code == 200

    def test_excludes_visited_destinations(self, client):
        visited = {"VN001": 5.0, "VN002": 3.0}
        response = client.post(
            "/api/v1/recommend/for-user",
            json={"user_ratings": visited, "k": 5},
        )
        result_ids = {item["destination_id"] for item in response.json()["results"]}
        for vid in visited:
            assert vid not in result_ids

    def test_empty_ratings_returns_error(self, client):
        response = client.post(
            "/api/v1/recommend/for-user",
            json={"user_ratings": {}, "k": 5},
        )
        # Pydantic min_length hoặc logic error
        assert response.status_code in (422, 500)


# ─── RAG: Advise ─────────────────────────────────────────────────────────────

class TestAdvise:
    def test_valid_question_returns_200(self, client):
        response = client.post(
            "/api/v1/advise",
            json={"question": "Hà Nội có gì hay?", "top_k": 3},
        )
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        response = client.post(
            "/api/v1/advise",
            json={"question": "Đặc sản miền Trung là gì?", "top_k": 3},
        )
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "sources" in data
        assert "retrieved_docs" in data

    def test_answer_is_non_empty(self, client):
        response = client.post(
            "/api/v1/advise",
            json={"question": "Du lịch Phú Quốc như thế nào?", "top_k": 2},
        )
        assert len(response.json()["answer"]) > 0

    def test_sources_is_list(self, client):
        response = client.post(
            "/api/v1/advise",
            json={"question": "Hội An nổi tiếng với gì?", "top_k": 2},
        )
        assert isinstance(response.json()["sources"], list)

    def test_short_question_returns_422(self, client):
        response = client.post(
            "/api/v1/advise",
            json={"question": "Hi", "top_k": 3},  # < min_length=3
        )
        assert response.status_code == 422

    def test_advise_health_endpoint(self, client):
        response = client.get("/api/v1/advise/health")
        assert response.status_code == 200
        data = response.json()
        assert "knowledge_base_chunks" in data
        assert "llm_mode" in data
