from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify GET /api/health returns exact expected payload and HTTP 200."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "MediaTrace API",
        "version": "1.0.0",
    }


def test_api_root():
    """Verify GET /api returns basic API status information."""
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MediaTrace API"
    assert data["version"] == "1.0.0"
    assert "message" in data
