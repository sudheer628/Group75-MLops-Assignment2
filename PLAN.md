# Assignment-2 Implementation Plan (MLOps)

This is our working plan for completing Assignment-2. We are following a similar approach to Assignment-1, but using different tools where possible (mainly CircleCI instead of GitHub Actions).

We will go stage by stage. After you approve this plan, we will start coding the first stage together.

---

## 1) What we need to deliver (from TASK.md)

We need to complete these 5 milestones:

- M1: Model development + experiment tracking + data/code versioning
- M2: Package model behind REST API + containerize with Docker
- M3: CI pipeline for tests + image build + image publish
- M4: CD pipeline and deployment on GCP VM
- M5: Monitoring/logging + post-deployment performance tracking

Dataset/use-case: Cats vs Dogs binary image classification (for a pet adoption platform).

---

## 2) Tools we will use

### Core stack

- Python 3.11
- PyTorch (we prefer this over TensorFlow for cleaner training loops)
- FastAPI for inference service
- Docker + Docker Compose
- pytest for tests

### MLOps tooling

- Code versioning: Git
- Data/model versioning: DVC (better than Git-LFS for large datasets)
- Experiment tracking: MLflow (self-hosted locally during dev, can use remote later)
- CI/CD: CircleCI (this is our main change from Assignment-1 where we used GitHub Actions)
- Container registry: Docker Hub
- Deployment: GCP VM with Docker Compose

### Monitoring stack

- Prometheus metrics endpoint in FastAPI
- Grafana for dashboards (optional, can use simple log-based monitoring)
- Request/response logging in app

---

## 3) Project structure

We will keep it simple and clean:

```
Assignment-2/
├── src/
│   ├── data/           # dataset download, split, preprocess
│   ├── models/         # CNN model definition
│   ├── training/       # training and evaluation scripts
│   └── inference/      # model loading and prediction utilities
├── app/
│   ├── main.py         # FastAPI application
│   ├── config.py       # app configuration
│   └── schemas.py      # request/response models
├── tests/
│   ├── test_data.py    # preprocessing tests
│   └── test_inference.py # inference tests
├── models/             # saved model artifacts (DVC tracked)
├── data/               # raw and processed data (DVC tracked)
├── .circleci/
│   └── config.yml      # CircleCI pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── dvc.yaml            # DVC pipeline definition
└── README.md
```

---

## 4) Stage-by-stage execution plan

---

## Stage M1: Model Development and Experiment Tracking

This is the coding-heavy stage. We will do this together.

### What we need to do

1. Setup project skeleton
   - Create folder structure
   - Initialize git repo
   - Create requirements.txt with pinned versions
   - Initialize DVC

2. Data pipeline
   - Download dataset using kagglehub (from dataset.py)
   - Organize into train/val/test splits (80/10/10)
   - Resize images to 224x224 RGB
   - Add data augmentation for training (random flip, rotation, color jitter)
   - Save processed data paths

3. Model development
   - Build a simple CNN baseline (we dont need anything fancy, just something that works)
   - Training script with configurable hyperparameters
   - Evaluation script (accuracy, precision, recall, F1, confusion matrix)
   - Save model as .pt file

4. Experiment tracking with MLflow
   - Setup local MLflow server
   - Log hyperparameters, metrics, and artifacts
   - Log confusion matrix and loss curves
   - Save model to MLflow

5. Data versioning with DVC
   - Track raw dataset
   - Track processed data
   - Track model artifacts
   - Create dvc.yaml pipeline

### What we will have after M1

- Working training pipeline
- Trained baseline model (.pt file)
- MLflow experiment with logged metrics
- DVC tracked data and model
- Reproducible with `dvc repro`

---

## Stage M2: Model Packaging and Containerization

Still coding-heavy. We continue together.

### What we need to do

1. FastAPI service
   - GET /health endpoint (returns service status)
   - POST /predict endpoint (accepts image, returns cat/dog prediction with confidence)
   - Load model once at startup (not on every request)

2. Request/response handling
   - Accept image upload (multipart/form-data)
   - Return JSON with prediction label and probability
   - Handle errors gracefully

3. Dockerfile
   - Use Python 3.11 slim base
   - Install dependencies
   - Copy app code
   - Expose port 8000
   - Add healthcheck

4. Local testing
   - Build and run container
   - Test with curl/Postman
   - Verify predictions work

### What we will have after M2

- Working FastAPI container
- Both endpoints tested
- Ready for CI/CD

---

## Stage M3: CI Pipeline with CircleCI

Mix of coding (tests) and configuration (CircleCI).

### What we need to do

1. Write unit tests
   - Test one preprocessing function (image resize, normalization)
   - Test one inference function (model prediction)
   - Make sure tests pass with pytest

2. CircleCI pipeline (.circleci/config.yml)
   - Job 1: Install dependencies and run pytest
   - Job 2: Build Docker image
   - Job 3: Push image to Docker Hub
   - Use CircleCI contexts for secrets (Docker Hub credentials)

3. Pipeline triggers
   - Run on every push to main branch
   - Run on pull requests

### CircleCI pipeline structure

```yaml
version: 2.1

jobs:
  test:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v

  build:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - setup_remote_docker
      - run: docker build -t $DOCKER_IMAGE:$CIRCLE_SHA1 .

  publish:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - setup_remote_docker
      - run: |
          echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
          docker build -t $DOCKER_IMAGE:latest .
          docker push $DOCKER_IMAGE:latest

workflows:
  build-test-publish:
    jobs:
      - test
      - build:
          requires:
            - test
      - publish:
          requires:
            - build
          filters:
            branches:
              only: main
```

### What we will have after M3

- Passing tests in CI
- Docker image published to Docker Hub on main branch pushes

---

## Stage M4: CD Pipeline and Deployment

This is mostly procedure/setup. We will provide step-by-step instructions.

### What we need to do

1. GCP VM setup (manual steps)
   - Create VM instance (or use existing from Assignment-1)
   - Install Docker and Docker Compose
   - Open firewall for port 80/443
   - Setup SSH keys for CircleCI

2. Deployment files
   - docker-compose.yml for production
   - Optional: nginx reverse proxy (like Assignment-1)

3. Extend CircleCI pipeline
   - Add deploy job that SSHs to VM
   - Pull latest image
   - Restart docker-compose service

4. Smoke tests
   - Call /health endpoint after deploy
   - Call /predict with a test image
   - Fail pipeline if smoke tests fail

### CircleCI deploy job (added to config.yml)

```yaml
deploy:
  docker:
    - image: cimg/python:3.11
  steps:
    - run:
        name: Deploy to GCP VM
        command: |
          ssh -o StrictHostKeyChecking=no $VM_USER@$VM_HOST << 'EOF'
            cd ~/cats-dogs-classifier
            docker-compose pull
            docker-compose up -d
          EOF

smoke-test:
  docker:
    - image: cimg/python:3.11
  steps:
    - run:
        name: Health check
        command: |
          curl -f http://$VM_HOST/health || exit 1
    - run:
        name: Prediction test
        command: |
          # test prediction endpoint
          curl -f -X POST http://$VM_HOST/predict -F "file=@test_image.jpg" || exit 1
```

### What we will have after M4

- Auto-deployment on main branch changes
- Smoke tests verify deployment success
- Live service on GCP VM

---

## Stage M5: Monitoring and Final Submission

Mostly setup and documentation.

### What we need to do

1. Add logging to FastAPI
   - Log request timestamps
   - Log prediction results (not the actual images)
   - Log errors

2. Add metrics endpoint
   - Request count
   - Request latency
   - Error count
   - Expose /metrics for Prometheus

3. Basic monitoring setup
   - Option A: Prometheus + Grafana on VM
   - Option B: Simple log-based monitoring (easier)

4. Post-deployment performance tracking
   - Run a batch of labeled test images
   - Calculate accuracy on this batch
   - Document results

5. Final documentation
   - Update README.md
   - Screenshots of MLflow, CircleCI, running service
   - Architecture diagram

### What we will have after M5

- Monitoring in place
- Post-deployment metrics documented
- Complete submission package

---

## 5) What needs coding vs what is procedure

As you requested:

| Stage | Type                     | What we do                                                   |
| ----- | ------------------------ | ------------------------------------------------------------ |
| M1    | Coding                   | Data pipeline, model training, MLflow integration, DVC setup |
| M2    | Coding                   | FastAPI app, Dockerfile, local testing                       |
| M3    | Coding + Config          | Unit tests, CircleCI config                                  |
| M4    | Procedure                | VM setup, deployment config, SSH setup                       |
| M5    | Procedure + Light coding | Logging/metrics code, monitoring setup, docs                 |

For M4 and M5, we will create the config files and scripts, but we will give you step-by-step instructions for the cloud/console actions.

---

## 6) Risks and how we handle them

- Dataset is large (~800MB): We use DVC with remote storage (Google Drive or GCS)
- Training takes time: We keep the baseline model simple, can improve later
- CircleCI free tier limits: We optimize pipeline to run only necessary jobs
- Deployment failures: We add proper health checks and rollback instructions

---

## 7) First steps after approval

Once you approve this plan, we will:

1. Create the folder structure
2. Setup requirements.txt with pinned versions
3. Initialize git and DVC
4. Build the data download and preprocessing pipeline
5. Create the baseline CNN model
6. Add training script with MLflow logging
7. Test everything locally

---

## 8) Questions before we start

1. Do you have a Kaggle account setup for downloading the dataset? (needed for kagglehub)
2. Do you have a Docker Hub account for pushing images?
3. Do you want to use the same GCP VM from Assignment-1 or create a new one?
4. For MLflow, do you want local tracking or should we setup a remote server?

---

If this plan looks good, let us know and we will start with Stage M1 Step 1.
