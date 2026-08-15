from PIL import Image

from inference import (
    CLASS_NAMES,
    model_ready,
    predict_image,
)


def test_model_is_ready():
    assert model_ready() is True


def test_predict_image_returns_expected_fields():
    image = Image.new(
        "RGB",
        (128, 128),
        "white",
    )

    result = predict_image(image)

    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result


def test_predict_image_returns_valid_class():
    image = Image.new(
        "RGB",
        (128, 128),
        "white",
    )

    result = predict_image(image)

    assert result["prediction"] in CLASS_NAMES


def test_probabilities_exist_for_every_class():
    image = Image.new(
        "RGB",
        (128, 128),
        "white",
    )

    result = predict_image(image)

    assert set(result["probabilities"].keys()) == set(CLASS_NAMES)


def test_probabilities_sum_to_one():
    image = Image.new(
        "RGB",
        (128, 128),
        "white",
    )

    result = predict_image(image)

    total = sum(result["probabilities"].values())

    assert abs(total - 1.0) < 0.0001