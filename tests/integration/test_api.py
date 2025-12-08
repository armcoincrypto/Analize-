"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from analize.api.app import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestAPIEndpoints:
    """Tests for API endpoints."""

    def test_root(self, client: TestClient) -> None:
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Analize" in data["message"]

    def test_health(self, client: TestClient) -> None:
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_status(self, client: TestClient) -> None:
        """Test status endpoint."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_signals_requires_dates(self, client: TestClient) -> None:
        """Test signals endpoint requires date parameters."""
        response = client.get("/signals")
        assert response.status_code == 422  # Validation error

    def test_signals_with_dates(self, client: TestClient) -> None:
        """Test signals endpoint with dates."""
        response = client.get("/signals?start_date=2024-01-01&end_date=2024-01-31")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "signals" in data

    def test_jobs_list(self, client: TestClient) -> None:
        """Test jobs list endpoint."""
        response = client.get("/jobs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_suggestions(self, client: TestClient) -> None:
        """Test suggestions endpoint."""
        response = client.get("/suggestions")
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert "generated_at" in data
