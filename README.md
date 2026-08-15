# Am I a Chicken? 🐔

You might be asking yourself the same question I've been asking myself for months:

**Am I a chicken?**

This project was built to finally answer that pertinent question.

Upload an image and the model will classify it as one of three things:

- 🐔 Chicken
- 🧍 Human
- ❓ Other

Behind this extremely important scientific mission is a slightly more serious goal: learning how to build and operate a complete machine learning system from end to end.

## Purpose

The goal of this project isn't to build the world's greatest chicken detector.

The classifier is intentionally simple. The real goal is to use it as a playground for learning the complete machine learning lifecycle:

**Data → Training → Evaluation → Model → API → Container → Application → Deployment → Monitoring**

The project currently uses:

- **PyTorch** for the image classification model
- **FastAPI** for model inference and API endpoints
- **Docker** for containerization
- **HTML/CSS/JavaScript** for a simple frontend
- **Pytest** for automated backend testing

As the project develops, it will also explore MLOps tooling for experiment tracking, data validation, pipeline orchestration, CI/CD, deployment, and monitoring.

## How It Works

An image is uploaded through the frontend and sent to the FastAPI backend.

The backend loads the trained PyTorch model and returns:

- Predicted class
- Confidence score
- Probability for each class

The current prediction flow is:

**User → Frontend → FastAPI → PyTorch → Chicken?**

## Project Structure

```text
AM-I-A-CHICKEN/
├── .venv/
│
├── backend/
│   ├── src/
│   │   ├── api.py
│   │   ├── inference.py
│   │   └── model.py
│   │
│   ├── models/
│   │   └── best_model.pt
│   │
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── api/
│   │   │   ├── test_health.py
│   │   │   ├── test_metadata.py
│   │   │   └── test_predict.py
│   │   ├── inference/
│   │   │   └── test_inference.py
│   │   └── model/
│   │       └── test_model.py
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── training/
│   ├── core_dataset/
│   └── ...
│
├── Dockerfile
├── README.md
└── .gitignore
```

## Model

The application currently uses a small PyTorch convolutional neural network called `ChickenCNN`.

- Input: RGB image resized to `128 × 128`
- Classes: `chicken`, `human`, `other`
- Model version: `v1`

The model architecture is defined in `backend/src/model.py`.

## API

The application uses FastAPI to expose the trained model.

### `GET /health`

Checks whether the API and model are available.

### `GET /classes`

Returns the available prediction classes.

### `GET /model-info`

Returns information about the currently deployed model.

### `POST /api/predict`

Accepts an image and returns a prediction.

Example:

```json
{
  "prediction": "chicken",
  "confidence": 0.91,
  "probabilities": {
    "chicken": 0.91,
    "human": 0.02,
    "other": 0.07
  }
}
```

## Running the Project

Build the Docker image from the project root:

```bash
docker build -t am-i-a-chicken .
```

Run the container:

```bash
docker run --rm -p 8000:8000 am-i-a-chicken
```

Then open:

`http://localhost:8000`

FastAPI documentation is available at:

`http://localhost:8000/docs`

## Running Tests

Activate the project's virtual environment:

```bash
source .venv/bin/activate
```

From the project root, run:

```bash
pytest -v
```

The current backend test suite contains **18 automated tests** covering:

- API health and model availability
- API metadata
- Prediction endpoint validation
- Valid image uploads
- Prediction response structure
- Prediction class validation
- Confidence values
- Inference output structure
- Class probabilities
- Probability sanity checks
- Model input/output shapes
- Batch inference
- Trainable model parameters
- Finite model outputs

Current status:

```text
18 passed
```

## Current Status

Working:

- Dataset preparation
- CNN training
- Model inference
- FastAPI backend
- Image upload and prediction
- Docker containerization
- Basic web interface
- End-to-end inference
- Automated backend testing

Next:

- Experiment tracking
- Data and model validation
- ML pipeline orchestration
- CI/CD
- Cloud deployment
- Monitoring

## The End Goal

Eventually, this project should demonstrate the complete lifecycle of a small production-style ML application — from raw data all the way to a deployed and monitored model.

And, more importantly, nobody should ever again have to live with the uncertainty of whether or not they are a chicken.