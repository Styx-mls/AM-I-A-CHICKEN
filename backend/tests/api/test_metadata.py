def test_classes(client):
    response = client.get("/classes")

    assert response.status_code == 200

    assert response.json() == {
        "classes": [
            "chicken",
            "human",
            "other",
        ]
    }


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "ChickenCNN"
    assert data["model_version"] == "v1"
    assert data["input_size"] == [128, 128]

    assert data["classes"] == [
        "chicken",
        "human",
        "other",
    ]