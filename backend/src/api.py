from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference import CLASS_NAMES, model_ready, predict_image


api = FastAPI(
    title="Am I a Chicken API",
    version="1.0.0"
)


api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        # Add your Vercel URL here later:
        # "https://your-app.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/")
def root():
    return {
        "message": "Am I a Chicken API",
        "docs": "/docs"
    }


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
async def predict(
    file: UploadFile = File(...)
):

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
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

    try:

        result = predict_image(image)

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )

    return result