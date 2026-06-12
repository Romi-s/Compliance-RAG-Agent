from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestIngestEndpoint:
    def test_rejects_oversized_file(self):
        huge = b"x" * (51 * 1024 * 1024)
        response = client.post(
            "/ingest",
            files={"file": ("big.txt", huge, "text/plain")},
        )
        assert response.status_code == 413

    def test_rejects_unsupported_type(self):
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        response = client.post(
            "/ingest",
            files={"file": ("image.png", png_header, "image/png")},
        )
        assert response.status_code == 415

    @patch("app.api.routes.ingest_text")
    @patch("app.api.routes.invalidate_bm25_cache")
    def test_ingests_text_file(self, mock_cache, mock_ingest):
        mock_ingest.return_value = {"filename": "rules.txt", "chunks_added": 5}
        response = client.post(
            "/ingest",
            files={"file": ("rules.txt", b"Some compliance text", "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_added"] == 5
        assert data["filename"] == "rules.txt"
        mock_cache.assert_called_once()


class TestQueryEndpoint:
    @patch("app.api.routes.qa_graph")
    def test_returns_answer_with_citations(self, mock_graph):
        mock_graph.invoke.return_value = {
            "question": "What is GDPR?",
            "answer": "GDPR is a regulation [1].",
            "citations": [
                {"source": "gdpr.pdf", "page": 1, "chunk_index": 0,
                 "text": "The General Data Protection Regulation..."},
            ],
            "retrieved_chunks": [{"text": "chunk", "source": "gdpr.pdf",
                                   "page": 1, "chunk_index": 0, "score": 0.9}],
            "error": None,
        }
        response = client.post("/query", data={"question": "What is GDPR?"})
        assert response.status_code == 200
        data = response.json()
        assert "GDPR" in data["answer"]
        assert len(data["citations"]) == 1
        assert data["chunks_used"] == 1

    @patch("app.api.routes.qa_graph")
    def test_returns_422_on_pipeline_error(self, mock_graph):
        mock_graph.invoke.return_value = {
            "question": "test",
            "answer": "",
            "citations": [],
            "retrieved_chunks": [],
            "error": "No relevant documents found. Please ingest documents first.",
        }
        response = client.post("/query", data={"question": "test"})
        assert response.status_code == 422


class TestCollectionStats:
    @patch("app.api.routes.get_collection")
    def test_empty_collection(self, mock_get):
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_get.return_value = mock_col
        response = client.get("/collection/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_chunks"] == 0
        assert data["sources"] == []


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestApiKeyAuth:
    """The API-key gate protects /ingest (owner-only). /collection/stats is public,
    read-only so the demo UI can display the chunk/document count without a key."""

    @patch("app.api.routes.settings")
    @patch("app.api.routes.get_collection")
    def test_stats_is_public(self, mock_get, mock_settings):
        mock_settings.api_key = "secret-key"
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_get.return_value = mock_col
        response = client.get("/collection/stats")
        assert response.status_code == 200

    @patch("app.api.routes.settings")
    def test_ingest_rejects_missing_key(self, mock_settings):
        mock_settings.api_key = "secret-key"
        mock_settings.max_file_size_mb = 50
        response = client.post(
            "/ingest",
            files={"file": ("rules.txt", b"hi", "text/plain")},
        )
        assert response.status_code == 401

    @patch("app.api.routes.settings")
    def test_ingest_rejects_wrong_key(self, mock_settings):
        mock_settings.api_key = "secret-key"
        mock_settings.max_file_size_mb = 50
        response = client.post(
            "/ingest",
            files={"file": ("rules.txt", b"hi", "text/plain")},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    @patch("app.api.routes.settings")
    @patch("app.api.routes.ingest_text")
    @patch("app.api.routes.invalidate_bm25_cache")
    def test_ingest_accepts_valid_key(self, mock_cache, mock_ingest, mock_settings):
        mock_settings.api_key = "secret-key"
        mock_settings.max_file_size_mb = 50
        mock_ingest.return_value = {"filename": "rules.txt", "chunks_added": 3}
        response = client.post(
            "/ingest",
            files={"file": ("rules.txt", b"hi", "text/plain")},
            headers={"X-API-Key": "secret-key"},
        )
        assert response.status_code == 200
