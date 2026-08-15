def test_predict_rejects_non_image(client):
    response = client.post(
        "/api/predict",
        files={
            "file": (
                "test.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_predict_accepts_valid_image(client, test_image):
    response = client.post(
        "/api/predict",
        files={
            "file": (
                "test.jpg",
                test_image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200


def test_predict_response_structure(client, test_image):
    response = client.post(
        "/api/predict",
        files={
            "file": (
                "test.jpg",
                test_image,
                "image/jpeg",
            )
        },
    )

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data


def test_prediction_is_valid_class(client, test_image):
    response = client.post(
        "/api/predict",
        files={
            "file": (
                "test.jpg",
                test_image,
                "image/jpeg",
            )
        },
    )

    data = response.json()

    assert data["prediction"] in [
        "chicken",
        "human",
        "other",
    ]


def test_confidence_is_valid_probability(client, test_image):
    response = client.post(
        "/api/predict",
        files={
            "file": (
                "test.jpg",
                test_image,
                "image/jpeg",
            )
        },
    )

    confidence = response.json()["confidence"]

    assert 0.0 <= confidence <= 1.0