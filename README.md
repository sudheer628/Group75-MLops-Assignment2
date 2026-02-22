# Cats vs Dogs Image Classification - MLOps Assignment 2

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/sudheer628/Group75-MLops-Assignment2/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/sudheer628/Group75-MLops-Assignment2/tree/master)

## Project Overview

This is our Group75 MLOps course assignment where we built an end-to-end ML pipeline for classifying images of cats and dogs. We wanted to learn how to handle image data in an MLOps workflow, from data versioning to model deployment. This time we used CircleCI instead of GitHub Actions for CI/CD.

We created a binary image classifier that predicts whether an image contains a cat or a dog. We used transfer learning with MobileNetV3-Small (pretrained on ImageNet) and deployed it as a REST API with a web portal for interactive demos.

## Live Demo

| Resource           | URL                                                     |
| ------------------ | ------------------------------------------------------- |
| Web Portal         | http://myprojectdemo.online                             |
| API Docs (Swagger) | http://myprojectdemo.online/docs                        |
| Health Check       | http://myprojectdemo.online/health                      |
| GitHub Repo        | https://github.com/sudheer628/Group75-MLops-Assignment2 |
| Docker Hub         | https://hub.docker.com/r/sudheer628/cats-dogs-api       |

## Tech Stack

| Category            | Tools                                 |
| ------------------- | ------------------------------------- |
| Language            | Python 3.11                           |
| ML Framework        | PyTorch, torchvision                  |
| API Framework       | FastAPI                               |
| Containerization    | Docker, Docker Compose                |
| CI/CD               | CircleCI                              |
| Data Versioning     | DVC                                   |
| Experiment Tracking | MLflow                                |
| Deployment          | GCP VM                                |
| Model               | MobileNetV3-Small (Transfer Learning) |

---

## Milestone 1: Model Development & Experiment Tracking

### Dataset

We used the "Dog and Cat Classification Dataset" from Kaggle (`bhavikjikadara/dog-and-cat-classification-dataset`). It contains around 25,000 images of cats and dogs.

We split the data into:

- Training: 80% (~20,000 images)
- Validation: 10% (~2,500 images)
- Test: 10% (~2,500 images)

### Data Versioning with DVC

We use DVC to track the large dataset without storing it in Git. The `dvc.yaml` file defines our pipeline stages:

```yaml
stages:
  prepare_data:
    cmd: python -m src.data.prepare_data
    deps:
      - src/data/prepare_data.py
    outs:
      - data/processed/train
      - data/processed/val
      - data/processed/test

  train:
    cmd: python -m src.training.train
    deps:
      - src/training/train.py
      - src/models/cnn.py
      - data/processed/train
      - data/processed/val
    outs:
      - models/best_model.pt
```

### Model Architecture

We started with a simple custom CNN but achieved only ~66% accuracy. We then switched to transfer learning with MobileNetV3-Small which gave us 98% accuracy.

**Final Architecture:**

- Backbone: MobileNetV3-Small (pretrained on ImageNet)
- Custom classifier head with dropout (0.3)
- Input: 224x224 RGB images
- Output: 2 classes (cat, dog)

### Training Strategy

We used a two-stage fine-tuning approach:

1. **Stage 1 (4 epochs):** Freeze backbone, train only the classifier head
2. **Stage 2 (remaining epochs):** Unfreeze backbone with lower learning rate

**Hyperparameters:**

- Optimizer: AdamW with weight decay (1e-4)
- Scheduler: CosineAnnealingLR
- Learning rate: 3e-4 (classifier), 3e-5 (backbone)
- Label smoothing: 0.1
- Early stopping patience: 5 epochs
- Mixed precision training (torch.cuda.amp)

**Data Augmentation:**

- RandomResizedCrop (224, scale 0.7-1.0)
- RandomHorizontalFlip
- RandomRotation (10 degrees)
- ColorJitter (brightness, contrast, saturation, hue)
- RandomErasing (15% probability)

### Training Results

We trained on a GPU VM using the full dataset:

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 98.36% |
| Precision     | 98.37% |
| Recall        | 98.36% |
| F1 Score      | 98.36% |
| Test Loss     | 0.231  |

### Experiment Tracking with MLflow

We used MLflow to track all our experiments. MLflow logs:

- Hyperparameters (learning rate, batch size, epochs, architecture)
- Metrics per epoch (train/val loss, accuracy)
- Artifacts (model weights, confusion matrix, training curves)

To view experiments locally:

```bash
mlflow ui
```

---

## Milestone 2: Model Packaging & Containerization

### FastAPI Service

We wrapped our model in a FastAPI application with these endpoints:

| Endpoint   | Method | Description                         |
| ---------- | ------ | ----------------------------------- |
| `/`        | GET    | Web portal for interactive demo     |
| `/health`  | GET    | Health check (returns model status) |
| `/predict` | POST   | Accepts image, returns prediction   |
| `/docs`    | GET    | Swagger API documentation           |

**Example API Usage:**

```bash
# Health check
curl http://myprojectdemo.online/health

# Response
{"status":"healthy","version":"1.0.0","model_loaded":true}

# Prediction
curl -X POST http://myprojectdemo.online/predict -F "file=@cat.jpg"

# Response
{"prediction":"cat","confidence":0.97,"probabilities":{"cat":0.97,"dog":0.03}}
```

### Dockerfile

We containerized the application with a multi-stage approach:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
services:
  api:
    image: sudheer628/cats-dogs-api:latest
    ports:
      - "80:8000"
      - "8000:8000"
    environment:
      - MODEL_PATH=models/best_model.pt
    restart: unless-stopped
```

---

## Milestone 3: CI Pipeline (CircleCI)

We chose CircleCI for this assignment (instead of GitHub Actions from Assignment-1) to learn a different CI/CD tool.

### Pipeline Jobs

Our CircleCI pipeline (`.circleci/config.yml`) has these jobs:

1. **test** - Run unit tests with pytest
2. **build** - Build Docker image
3. **publish** - Push image to Docker Hub
4. **deploy** - Deploy to GCP VM via SSH

### Pipeline Configuration

```yaml
version: 2.1

orbs:
  python: circleci/python@2.1.1

jobs:
  test:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - python/install-packages:
          pkg-manager: pip
          pip-dependency-file: requirements.txt
      - run:
          name: Run tests
          command: pytest tests/ -v --tb=short

  build:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - setup_remote_docker:
          docker_layer_caching: true
      - run:
          name: Build Docker image
          command: docker build -t cats-dogs-api:${CIRCLE_SHA1} .

  publish:
    docker:
      - image: cimg/python:3.11
    steps:
      - setup_remote_docker
      - run:
          name: Push to Docker Hub
          command: |
            echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
            docker push $DOCKER_USERNAME/cats-dogs-api:latest

  deploy:
    docker:
      - image: cimg/python:3.11
    steps:
      - add_ssh_keys
      - run:
          name: Deploy to GCP VM
          command: |
            ssh $VM_USER@$VM_HOST "cd ~/cats-dogs-classifier && docker-compose pull && docker-compose up -d"

workflows:
  build-test-deploy:
    jobs:
      - test
      - build:
          requires: [test]
      - publish:
          requires: [build]
          filters:
            branches:
              only: master
      - deploy:
          requires: [publish]
          filters:
            branches:
              only: master
```

### CircleCI Environment Variables

We configured these environment variables in CircleCI:

| Variable          | Description                        |
| ----------------- | ---------------------------------- |
| `DOCKER_USERNAME` | Docker Hub username                |
| `DOCKER_PASSWORD` | Docker Hub access token            |
| `VM_USER`         | GCP VM SSH username                |
| `VM_HOST`         | GCP VM IP address                  |
| `SSH_FINGERPRINT` | SSH key fingerprint for deployment |

### Unit Tests

We wrote unit tests for:

- Data preprocessing functions (image transforms)
- Model inference functions (prediction)

```bash
# Run tests locally
pytest tests/ -v
```

---

## Milestone 4: CD Pipeline & Deployment

### GCP VM Setup

We deployed to a GCP VM with the following setup:

1. Created a VM instance (e2-medium, Ubuntu 22.04)
2. Installed Docker and Docker Compose
3. Opened firewall ports 80 and 8000
4. Added SSH key for CircleCI deployment

**Firewall Rules:**

```bash
gcloud compute firewall-rules create allow-http-80 --allow tcp:80
gcloud compute firewall-rules create allow-api-8000 --allow tcp:8000
```

### Deployment Flow

On every push to master branch:

1. CircleCI runs tests
2. Builds Docker image
3. Pushes to Docker Hub
4. SSHs to GCP VM
5. Pulls latest image
6. Restarts container with docker-compose

### Domain Setup

We configured our domain `myprojectdemo.online` with an A record pointing to the GCP VM IP address.

---

## Milestone 5: Monitoring & Logging

### Request Logging

Our FastAPI application logs:

- Request timestamps
- Prediction results (class and confidence)
- Latency per request
- Errors with stack traces

### Health Check Endpoint

The `/health` endpoint returns:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Metrics Tracked

- Request count
- Prediction latency
- Model load status
- Error count

---

## Project Structure

```
Assignment-2/
├── app/                        # FastAPI application
│   ├── main.py                 # API endpoints
│   ├── config.py               # Configuration
│   ├── schemas.py              # Request/response models
│   └── static/                 # Web portal HTML
├── src/                        # ML pipeline code
│   ├── data/
│   │   ├── prepare_data.py     # Download and split dataset
│   │   └── dataset.py          # PyTorch Dataset with transforms
│   ├── models/
│   │   └── cnn.py              # Model architectures (SimpleCNN, TransferLearningCNN)
│   ├── training/
│   │   └── train.py            # Training loop with MLflow logging
│   └── inference/
│       └── predictor.py        # Model loading and prediction
├── tests/                      # Unit tests
├── models/                     # Trained model artifacts
│   ├── best_model.pt           # Model weights
│   ├── metrics.json            # Evaluation metrics
│   └── confusion_matrix.png    # Confusion matrix plot
├── data/                       # Dataset (DVC tracked)
├── docs/                       # Documentation
│   └── openapi.yaml            # OpenAPI specification
├── .circleci/
│   └── config.yml              # CircleCI pipeline
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                    # DVC pipeline definition
├── requirements.txt            # Training dependencies
├── requirements-api.txt        # API dependencies
└── run_pipeline.py             # Main pipeline script
```

---

## How to Run Locally

### Setup

```bash
# Clone the repo
git clone https://github.com/sudheer628/Group75-MLops-Assignment2.git
cd Group75-MLops-Assignment2

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data and Train

```bash
# Quick test (limited data, CPU)
python run_pipeline.py --quick --no-mlflow

# Full training with transfer learning (GPU recommended)
python run_pipeline.py --architecture mobilenet_v3_small --epochs 20 --freeze-epochs 4
```

### Run the API

```bash
# Using uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Using Docker
docker-compose up --build
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Key Files Description

| File                         | Description                                        |
| ---------------------------- | -------------------------------------------------- |
| `run_pipeline.py`            | Main script to run data prep and training          |
| `src/data/prepare_data.py`   | Downloads dataset from Kaggle and creates splits   |
| `src/data/dataset.py`        | PyTorch Dataset with augmentation transforms       |
| `src/models/cnn.py`          | Model definitions (SimpleCNN, TransferLearningCNN) |
| `src/training/train.py`      | Training loop with two-stage fine-tuning           |
| `src/inference/predictor.py` | Model loading and prediction utilities             |
| `app/main.py`                | FastAPI application with endpoints                 |
| `app/static/index.html`      | Web portal for interactive demo                    |
| `.circleci/config.yml`       | CircleCI pipeline configuration                    |
| `dvc.yaml`                   | DVC pipeline stages                                |
| `Dockerfile`                 | Container image definition                         |
| `docker-compose.yml`         | Production deployment configuration                |

---

## Differences from Assignment-1

| Aspect          | Assignment-1                       | Assignment-2                             |
| --------------- | ---------------------------------- | ---------------------------------------- |
| Data Type       | Tabular (CSV)                      | Images                                   |
| Model           | Scikit-learn (Logistic Regression) | PyTorch MobileNetV3 (Transfer Learning)  |
| CI/CD           | GitHub Actions                     | CircleCI                                 |
| Data Versioning | Git                                | DVC                                      |
| Dataset Size    | 303 rows                           | ~25,000 images                           |
| Feature Store   | Yes (engineered features)          | No (CNN extracts features automatically) |
| Training        | CPU (seconds)                      | GPU with mixed precision (minutes)       |
| Accuracy        | ~85%                               | 98.36%                                   |

---

## Note on Feature Store

In Assignment-1, we used a feature store to store engineered features from tabular data. For this image classification project, we don't use a feature store because the images themselves ARE the features. The CNN learns to extract features automatically through its convolutional layers. The raw pixel data goes through transforms (resize, normalize) and the model handles feature extraction internally.

---

## What We Learned

1. **Transfer learning is powerful** - Using a pretrained MobileNetV3 backbone gave us 98% accuracy vs ~66% with a custom CNN trained from scratch. Pretrained models have already learned useful features from ImageNet.

2. **Two-stage fine-tuning works well** - Freezing the backbone first and training only the classifier head, then unfreezing with a lower learning rate, helps prevent catastrophic forgetting.

3. **Image data is different** - Unlike tabular data in Assignment-1, images need special handling (transforms, augmentation, batching). There's no separate "feature engineering" step - the CNN does it automatically.

4. **GPU training is essential** - Training deep learning models on CPU is very slow. Mixed precision training (AMP) on GPU made training much faster.

5. **CircleCI vs GitHub Actions** - Both work well for CI/CD. CircleCI has a nice UI and good Docker support. The configuration syntax is slightly different but the concepts are the same.

6. **DVC for large files** - Git can't handle large datasets, so DVC is essential for ML projects. It tracks data versions without storing the actual files in Git.

7. **Data augmentation helps** - RandomResizedCrop, ColorJitter, and RandomErasing help the model generalize better and prevent overfitting.

8. **No feature store for images** - Unlike tabular ML, image classification doesn't need a separate feature store. The model learns features end-to-end.

---

## Team

**Group75** - BITS WILP MLOps Course 2025-2026
