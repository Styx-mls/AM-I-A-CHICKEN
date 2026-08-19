# Am I a Chicken?

You might be asking yourself the same question I've been asking myself for months:

**Am I a chicken?**

This project was built to finally answer that pertinent question.

Upload an image and the model will classify it as one of three things:

- Chicken
- Human
- Other

Behind this extremely important scientific mission is a more serious goal: building a complete machine learning system from raw data through training, inference, deployment, and eventually production monitoring.

---

## Purpose

The goal of this project isn't to build the world's greatest chicken detector.

The classifier is intentionally simple.

Instead, the project acts as a practical environment for learning and implementing the complete machine learning lifecycle:

**Raw Data → Validation → Preparation → Training → Evaluation → Model → API → Container → Application → Deployment → Monitoring**

Rather than hiding the ML lifecycle behind a large framework, each component is being implemented and integrated incrementally.

The project currently includes:

- PyTorch model development
- Dataset validation
- Dataset preparation
- Model training
- Model evaluation
- ZenML pipeline orchestration
- FastAPI model serving
- Docker containerization
- Browser-based inference
- Automated testing

Additional MLOps tooling will be added as the project develops.

---

## System Architecture

The project currently contains two major workflows:

1. The training pipeline
2. The inference application

### Training

```text
Raw Dataset
     |
     v
Data Validation
     |
     v
Data Preparation
     |
     v
Train / Validation / Test
     |
     v
Model Training
     |
     v
Model Evaluation
     |
     v
Trained Model
```

The workflow is orchestrated using **ZenML**.

### Inference

```text
User
 |
 v
Web Frontend
 |
 v
FastAPI
 |
 v
Image Preprocessing
 |
 v
ChickenCNN
 |
 v
Softmax Probabilities
 |
 v
Prediction
```

The training environment and inference application are intentionally separated.

Training and MLOps tooling do not need to be included in the production inference container.

---

## Technology Stack

### Machine Learning

- Python
- PyTorch
- Torchvision
- Pillow

### MLOps

- ZenML
- Deepchecks — planned
- MLflow — planned

### Backend

- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

### Infrastructure

- Docker

### Testing

- Pytest

---

## Dataset

The classifier predicts three classes:

```text
chicken
human
other
```

The original source data contains several categories.

During dataset preparation, those categories are mapped into the three final classes used by the model.

Human images are mapped to:

```text
human
```

Chicken images are mapped to:

```text
chicken
```

All remaining supported categories are mapped to:

```text
other
```

This converts the original multiclass dataset into the three-class classification problem used by the application.

### Dataset Split

The prepared dataset is split approximately into:

```text
80% training
10% validation
10% testing
```

A fixed random seed is used so dataset generation is reproducible.

---

## Dataset Preparation

Dataset preparation is handled by:

```text
training/data_preparation.py
```

The preparation stage:

1. Reads the source dataset.
2. Maps source classes into `chicken`, `human`, and `other`.
3. Balances or caps source categories where required.
4. Randomizes the dataset using a deterministic seed.
5. Splits the data into training, validation, and test sets.
6. Writes the prepared dataset structure used during training.

The goal is to keep dataset construction reproducible and separate from the actual model-training logic.

---

## Dataset Validation

Dataset validation is handled by:

```text
training/data_validation.py
```

Validation runs before preparation and training.

Its purpose is to catch problems with the source data before those problems propagate into the model.

The current validation stage checks the source dataset and verifies that the expected data is available for downstream pipeline stages.

More comprehensive validation will be introduced using **Deepchecks**.

Planned checks include:

- Class imbalance
- Duplicate samples
- Train/test leakage
- Data integrity
- Distribution changes
- Model performance issues

---

## Model

The project uses a custom convolutional neural network called:

```text
ChickenCNN
```

The model is implemented in:

```text
backend/src/model.py
```

The classifier receives RGB images resized to:

```text
128 x 128
```

and produces logits for three classes:

```text
chicken
human
other
```

### Architecture

The network contains three convolutional feature-extraction blocks.

Conceptually:

```text
Input Image
128 x 128 x 3
      |
      v
Conv2D
3 -> 32 channels
      |
      v
ReLU
      |
      v
MaxPool
      |
      v
Conv2D
32 -> 64 channels
      |
      v
ReLU
      |
      v
MaxPool
      |
      v
Conv2D
64 -> 128 channels
      |
      v
ReLU
      |
      v
MaxPool
      |
      v
Adaptive Average Pooling
      |
      v
Flatten
      |
      v
Linear Classifier
128 -> 3
      |
      v
Class Logits
```

The three output logits correspond to:

```text
chicken
human
other
```

Softmax is applied during inference to convert the logits into class probabilities.

### Why Adaptive Average Pooling?

The network uses adaptive average pooling before the final classifier.

This reduces the final convolutional feature maps into a fixed-size representation before passing them into the linear classification layer.

This keeps the classifier architecture simple and avoids requiring a large fully connected layer based directly on the spatial dimensions of the convolutional output.

---

## Training

Model training is handled by:

```text
training/train.py
```

The current training configuration uses approximately:

```text
Optimizer: Adam
Learning rate: 0.001
Batch size: 32
Input size: 128 x 128
```

Training has intentionally remained relatively lightweight while the surrounding ML infrastructure is developed.

The current focus of the project is not maximizing benchmark accuracy. It is building the complete system required to reliably train, evaluate, serve, and eventually monitor models.

---

## Class Sampling

The dataset contains different numbers of examples for each target class.

Training therefore uses a weighted sampling strategy to increase the representation of chicken images during training.

The current relative sampling weights are:

```text
chicken: 1.5
human:   1.0
other:   1.0
```

This is implemented using PyTorch's weighted sampling functionality.

The goal is to reduce the tendency of the classifier to favor more heavily represented classes.

---

## Evaluation

Model evaluation is handled by:

```text
training/evaluate.py
```

The evaluation stage operates on held-out data that is not used to train the model.

Its purpose is to measure how well the trained model generalizes beyond the training set.

Evaluation can include metrics such as:

- Overall accuracy
- Per-class performance
- Confusion matrix
- Prediction distribution

As the MLOps stack develops, evaluation metrics will also be tracked across model runs and versions.

---

## ZenML Pipeline

The machine learning workflow is orchestrated using **ZenML**.

Instead of manually running each training script independently, the project represents the workflow as a pipeline.

The current pipeline is:

```text
Data Validation
      |
      v
Data Preparation
      |
      v
Training
      |
      v
Evaluation
```

Each component is represented as an independent pipeline step.

This allows the workflow to be executed, inspected, and extended as a single ML pipeline.

### Why ZenML?

Without orchestration, the training workflow would effectively be:

```text
python data_validation.py
python data_preparation.py
python train.py
python evaluate.py
```

That works for a small experiment but becomes increasingly difficult to manage as the ML system grows.

ZenML provides a structured representation of the workflow and tracks pipeline executions.

This creates a foundation for integrating additional tooling such as:

```text
Deepchecks
MLflow
CI/CD
Model promotion
Deployment
Monitoring
```

without rewriting the entire training workflow.

### Pipeline Execution

A pipeline run executes the stages in dependency order:

```text
validation
    |
    v
preparation
    |
    v
training
    |
    v
evaluation
```

Training produces the trained model output required by evaluation.

This ensures that evaluation is associated with the model produced by the corresponding training run.

Pipeline runs and individual steps can be inspected through the ZenML dashboard.

---

## Model Artifact

The trained model used by the inference application is stored under:

```text
backend/models/
```

The current deployed model is:

```text
backend/models/best_model.pt
```

The inference application loads this model when the API starts.

Future versions of the project will introduce more explicit model versioning and promotion so that a newly trained model does not automatically replace the deployed model unless it satisfies evaluation requirements.

---

## Inference

Inference logic is implemented in:

```text
backend/src/inference.py
```

The inference pipeline is:

```text
Uploaded Image
      |
      v
Decode Image
      |
      v
Convert to RGB
      |
      v
Resize / Transform
      |
      v
PyTorch Tensor
      |
      v
ChickenCNN
      |
      v
Logits
      |
      v
Softmax
      |
      v
Class Probabilities
      |
      v
Prediction
```

The class with the highest probability is returned as the final prediction.

---

## API

The model is exposed through a FastAPI backend.

The API implementation is located in:

```text
backend/src/api.py
```

The backend exposes several endpoints.

### Health

```http
GET /health
```

Checks whether the API is running and whether the model is available.

### Classes

```http
GET /classes
```

Returns the classes supported by the classifier.

Example:

```json
{
  "classes": [
    "chicken",
    "human",
    "other"
  ]
}
```

### Model Information

```http
GET /model-info
```

Returns metadata about the currently deployed model.

### Prediction

```http
POST /api/predict
```

Accepts an uploaded image and runs model inference.

Example response:

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

---

## Frontend

The project includes a lightweight browser interface located in:

```text
frontend/
```

The frontend consists of:

```text
index.html
style.css
app.js
```

Users can select an image from their computer and submit it to the backend.

The frontend sends the image to:

```text
POST /api/predict
```

and displays the model's response.

The frontend is intentionally simple because the focus of the project is the machine learning system rather than frontend development.

---

## Docker

The inference application is containerized using Docker.

The Docker image contains the components required to serve predictions:

```text
FastAPI
PyTorch
Model architecture
Inference code
Trained model
Frontend
```

Training and MLOps tooling such as ZenML do not need to be included in the inference container.

This keeps model serving separate from model development and prevents the production image from accumulating unnecessary training dependencies.

### Build the Docker Image

From the repository root:

```bash
docker build -t am-i-a-chicken .
```

### Run the Container

```bash
docker run --rm -p 8000:8000 am-i-a-chicken
```

The application will be available at:

```text
http://localhost:8000
```

FastAPI's interactive documentation is available at:

```text
http://localhost:8000/docs
```

---

## Testing

The project includes an automated Pytest suite for the backend.

Tests are located under:

```text
backend/tests/
```

The current suite contains:

```text
18 tests
```

and covers the API, inference pipeline, and model.

### API Tests

API tests verify:

- Health endpoint behavior
- Model availability
- Metadata endpoints
- Prediction endpoint validation
- Valid image uploads
- Response structure
- Prediction classes
- Confidence values

### Inference Tests

Inference tests verify:

- Image preprocessing
- Prediction output structure
- Class probabilities
- Probability sanity checks
- Valid class outputs

### Model Tests

Model tests verify:

- Model input shape
- Model output shape
- Batch inference
- Trainable parameters
- Finite outputs

### Running Tests

Activate the project environment:

```bash
source .venv/bin/activate
```

Then run:

```bash
pytest -v
```

Current status:

```text
18 passed
```

---

## Project Structure

```text
AM-I-A-CHICKEN/
|
├── backend/
│   |
│   ├── src/
│   │   ├── api.py
│   │   ├── inference.py
│   │   └── model.py
│   |
│   ├── models/
│   │   └── best_model.pt
│   |
│   ├── tests/
│   │   ├── conftest.py
│   │   |
│   │   ├── api/
│   │   │   ├── test_health.py
│   │   │   ├── test_metadata.py
│   │   │   └── test_predict.py
│   │   |
│   │   ├── inference/
│   │   │   └── test_inference.py
│   │   |
│   │   └── model/
│   │       └── test_model.py
│   |
│   └── requirements.txt
|
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
|
├── training/
│   ├── data_validation.py
│   ├── data_preparation.py
│   ├── train.py
│   ├── evaluate.py
│   └── pipeline.py
|
├── Dockerfile
├── README.md
└── .gitignore
```

---

## Current Status

### Data

Implemented:

- Dataset ingestion
- Three-class mapping
- Train/validation/test splitting
- Dataset validation
- Dataset preparation
- Reproducible splitting

### Model

Implemented:

- Custom PyTorch CNN
- Three-class image classification
- Weighted training sampler
- Model training
- Model evaluation
- Saved model artifact

### MLOps

Implemented:

- ZenML integration
- Validation pipeline step
- Preparation pipeline step
- Training pipeline step
- Evaluation pipeline step
- End-to-end pipeline execution
- Pipeline run visualization

### Backend

Implemented:

- FastAPI application
- Model loading
- Image preprocessing
- Prediction endpoint
- Confidence scores
- Per-class probabilities
- Health endpoint
- Model metadata endpoints
- CORS configuration

### Frontend

Implemented:

- Image selection
- Image upload
- API integration
- Prediction display

### Infrastructure

Implemented:

- Docker image
- Containerized inference API
- Automated Pytest suite

---

## Next Steps

### Deepchecks

The next planned MLOps integration is **Deepchecks**.

Deepchecks will expand the existing validation system with automated checks for areas such as:

- Dataset integrity
- Duplicate samples
- Potential data leakage
- Class imbalance
- Distribution differences
- Model performance issues

Deepchecks will be integrated into the existing pipeline rather than implemented as an isolated script.

### MLflow

MLflow will be added for experiment tracking.

The goal is to track information such as:

```text
learning rate
batch size
epochs
model configuration
training loss
validation loss
accuracy
per-class metrics
model artifacts
```

This will make model runs directly comparable rather than relying on terminal output or manually recorded results.

### Model Promotion

Eventually, newly trained models should not automatically become production models.

A model promotion workflow will compare a candidate model against the currently deployed model.

Conceptually:

```text
Train Candidate
      |
      v
Evaluate Candidate
      |
      v
Compare Against Current Model
      |
      +------ worse ------> Reject
      |
      +------ better -----> Promote
```

This will allow the project to implement an:

```text
only update if better
```

model deployment policy.

### CI/CD

Future CI/CD work will automate:

- Test execution
- Pipeline validation
- Docker builds
- Deployment checks
- Model promotion

### Deployment

The FastAPI/Docker application will eventually be deployed to a cloud environment.

The production deployment will expose the same prediction API currently used locally.

### Monitoring

Production monitoring will eventually track areas such as:

- Prediction volume
- Prediction confidence
- Class distribution
- API errors
- Latency
- Data drift
- Model performance degradation

This completes the feedback loop between training and production.

---

## Planned ML Lifecycle

The eventual system should look approximately like:

```text
                    RAW DATA
                       |
                       v
                 DATA VALIDATION
                       |
                       v
                 DATA PREPARATION
                       |
                       v
                TRAIN / VAL / TEST
                       |
                       v
                    TRAIN
                       |
                       v
                   EVALUATE
                       |
                       v
              EXPERIMENT TRACKING
                       |
                       v
              CANDIDATE MODEL
                       |
                       v
             MODEL VALIDATION
                       |
             +---------+---------+
             |                   |
           REJECT              PROMOTE
                                 |
                                 v
                              DEPLOY
                                 |
                                 v
                              MONITOR
                                 |
                                 v
                         NEW DATA / SIGNALS
                                 |
                                 +----> TRAINING PIPELINE
```

---

## Why Build This?

A basic image classifier can be built in a relatively small amount of code.

A reliable machine learning system requires considerably more:

- Reproducible data preparation
- Data validation
- Training pipelines
- Experiment tracking
- Evaluation
- Testing
- Model versioning
- APIs
- Containers
- Deployment
- Monitoring

This project intentionally starts with a simple classification problem so the complexity can be focused on those engineering problems.

The classifier may be simple.

The system around it does not have to be.

And, more importantly, nobody should ever again have to live with the uncertainty of whether or not they are a chicken.
