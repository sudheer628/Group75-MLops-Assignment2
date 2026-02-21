# Cats vs Dogs Image Classification - MLOps Assignment 2

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/sudheer628/Group75-MLops-Assignment2/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/sudheer628/Group75-MLops-Assignment2/tree/master)

This is our Group75 MLOps course assignment where we built an end-to-end ML pipeline for classifying images of cats and dogs. We wanted to learn how to handle image data in an MLOps workflow, from data versioning to model deployment. This time we used CircleCI instead of GitHub Actions for CI/CD.

## What We Built

We created a binary image classifier that predicts whether an image contains a cat or a dog. We used a CNN (Convolutional Neural Network) built with PyTorch and deployed it as a REST API. The model takes a 224x224 RGB image and outputs the predicted class with confidence scores.

GitHub Repo: https://github.com/sudheer628/Group75-MLops-Assignment2

Live API (after deployment): http://136.118.140.118:8000
Docker Hub: https://hub.docker.com/r/sudheer628/cats-dogs-api

## Tech Stack

- Python 3.10+, PyTorch, torchvision
- FastAPI for the REST API
- Docker and Docker Compose
- CircleCI for CI/CD (different from Assignment-1 where we used GitHub Actions)
- DVC for data versioning
- MLflow for experiment tracking
- GCP VM for deployment

## Project Structure

```
.
├── app/                    # FastAPI application
│   ├── main.py            # API endpoints
│   ├── schemas.py         # Request/response models
│   └── config.py          # Configuration
├── src/                    # ML pipeline code
│   ├── data/              # Data loading and preprocessing
│   │   ├── prepare_data.py    # Download and split dataset
│   │   └── dataset.py         # PyTorch Dataset class
│   ├── models/            # Model architecture
│   │   └── cnn.py             # CNN model definition
│   ├── training/          # Training scripts
│   │   └── train.py           # Training loop and evaluation
│   └── inference/         # Inference code
│       └── predictor.py       # Model loading and prediction
├── tests/                  # Unit tests
├── models/                 # Trained model artifacts
├── data/                   # Dataset (tracked with DVC)
├── .circleci/             # CircleCI configuration
│   └── config.yml
├── docs/                   # Documentation
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml               # DVC pipeline definition
├── requirements.txt
└── run_pipeline.py        # Main pipeline script
```

## Dataset

We used the "Dog and Cat Classification Dataset" from Kaggle (bhavikjikadara/dog-and-cat-classification-dataset). It has around 25,000 images of cats and dogs.

We split the data into:

- Training: 80% (~20,000 images)
- Validation: 10% (~2,500 images)
- Test: 10% (~2,500 images)

The raw data is tracked with DVC so we don't have to store large files in Git.

## Model Architecture

We built a simple CNN with:

- 4 convolutional blocks (32 -> 64 -> 128 -> 256 channels)
- Batch normalization after each conv layer
- Max pooling to reduce spatial dimensions
- Global average pooling
- 2 fully connected layers with dropout
- Input: 224x224 RGB images
- Output: 2 classes (cat, dog)

## Training Results

We trained the model on CPU (no GPU available locally), so we used a subset of the data:

- Training samples: 2000
- Validation samples: 500
- Test samples: 500
- Epochs: 5

Current metrics:

- Test Accuracy: 66.6%
- Precision: 70.2%
- Recall: 66.6%
- F1 Score: 64.2%

Note: With GPU and full dataset (15-20 epochs), we expect 85-90%+ accuracy.

## How to Run Locally

### Setup

```bash
# Clone the repo
git clone https://github.com/sudheer628/Group75-MLops-Assignment2.git
cd Group75-MLops-Assignment2

# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data and Train

```bash
# Run the full pipeline (downloads data, trains model)
python run_pipeline.py --epochs 10 --no-mlflow

# Or with quick mode for testing
python run_pipeline.py --quick --no-mlflow

# Skip download if data already exists
python run_pipeline.py --skip-download --epochs 10 --no-mlflow
```

### Run Tests

```bash
pytest tests/ -v
```

### Run the API Locally

```bash
# Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Docker
docker-compose up --build
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/cat_or_dog_image.jpg"
```

Response:

```json
{
  "prediction": "cat",
  "confidence": 0.87,
  "probabilities": {
    "cat": 0.87,
    "dog": 0.13
  }
}
```

## CI/CD with CircleCI

We chose CircleCI for this assignment (instead of GitHub Actions from Assignment-1) to learn a different CI/CD tool.

Our pipeline has these jobs:

1. **test** - Run unit tests with pytest
2. **build** - Build Docker image
3. **publish** - Push image to Docker Hub
4. **deploy** - Deploy to GCP VM via SSH
5. **smoke-test** - Verify deployment is working

The pipeline triggers on every push to main branch.

See `.circleci/config.yml` for the full configuration.

## Data Versioning with DVC

We use DVC to track the raw dataset:

```bash
# Pull the data (if you have access to remote storage)
dvc pull

# Check data status
dvc status
```

The `data/raw.dvc` file tracks the raw images without storing them in Git.

## Experiment Tracking with MLflow

We use MLflow to track training experiments:

```bash
# Run training with MLflow logging
python run_pipeline.py --epochs 10

# View experiments
mlflow ui
```

MLflow logs:

- Hyperparameters (learning rate, batch size, epochs, etc.)
- Metrics (train/val loss, accuracy per epoch)
- Artifacts (model weights, confusion matrix, training curves)

## Deployment

We deploy to a GCP VM using Docker Compose. The deployment is automated through CircleCI.

See `docs/M4-DEPLOYMENT-PROCEDURE.md` for manual deployment steps.

## Monitoring

We plan to set up monitoring similar to Assignment-1:

- Prometheus metrics for API performance
- Grafana dashboards for visualization
- Logging with structured logs

See `docs/M5-MONITORING-PROCEDURE.md` for the monitoring setup guide.

## What We Learned

1. **Image data is different** - Unlike tabular data in Assignment-1, images need special handling (transforms, augmentation, batching)

2. **Deep learning needs GPUs** - Training CNNs on CPU is very slow. For production, we'd use GPU instances.

3. **CircleCI vs GitHub Actions** - Both work well for CI/CD. CircleCI has a nice UI and good Docker support.

4. **DVC for large files** - Git can't handle large datasets, so DVC is essential for ML projects.

5. **Data augmentation helps** - Random flips, rotations, and color jitter help the model generalize better.

## Differences from Assignment-1

| Aspect          | Assignment-1                       | Assignment-2   |
| --------------- | ---------------------------------- | -------------- |
| Data Type       | Tabular (CSV)                      | Images         |
| Model           | Scikit-learn (Logistic Regression) | PyTorch CNN    |
| CI/CD           | GitHub Actions                     | CircleCI       |
| Data Versioning | Git                                | DVC            |
| Dataset Size    | 303 rows                           | ~25,000 images |

## Files Description

| File                         | Description                               |
| ---------------------------- | ----------------------------------------- |
| `run_pipeline.py`            | Main script to run data prep and training |
| `src/data/prepare_data.py`   | Downloads and splits the dataset          |
| `src/data/dataset.py`        | PyTorch Dataset with transforms           |
| `src/models/cnn.py`          | CNN architecture definition               |
| `src/training/train.py`      | Training loop with evaluation             |
| `src/inference/predictor.py` | Model loading and prediction              |
| `app/main.py`                | FastAPI endpoints                         |
| `dvc.yaml`                   | DVC pipeline stages                       |
| `.circleci/config.yml`       | CircleCI pipeline config                  |

## Team

Group75 - BITS WILP MLOps Course 2025-2026
# Test deployment 2026-02-21 23:27:20
