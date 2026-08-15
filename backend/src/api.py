from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference import CLASS_NAMES, model_ready, predict_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


api = FastAPI(
    title="Am I a Chicken API",
    version="1.0.0"
)


api.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


@api.get("/")
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@api.get("/health")
def health():

    if not model_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    return {
        "status": "healthy",
        "model_loaded": True
    }


@api.get("/classes")
def classes():
    return {
        "classes": CLASS_NAMES
    }


@api.get("/model-info")
def model_info():
    return {
        "model": "ChickenCNN",
        "model_version": "v1",
        "input_size": [128, 128],
        "classes": CLASS_NAMES,
        "dataset_version": "processed_v1"
    }


@api.post("/api/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image"
        )

    try:
        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read image"
        )

    result = predict_image(image)

    return result