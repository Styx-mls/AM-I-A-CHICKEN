import sys
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"

sys.path.insert(0, str(SRC_DIR))


from api import api


@pytest.fixture
def client():
    return TestClient(api)


@pytest.fixture
def test_image():
    image = Image.new(
        "RGB",
        (128, 128),
        "white"
    )

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer