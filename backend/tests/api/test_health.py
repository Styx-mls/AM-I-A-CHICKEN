def test_health_returns_200(client):
    response = client.get("/health")

    assert response.status_code == 200


def test_health_reports_model_loaded(client):
    response = client.get("/health")

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True