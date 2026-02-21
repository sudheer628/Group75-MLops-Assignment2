# M5: Monitoring and Final Submission Procedure

This document describes how to set up monitoring and prepare the final submission.

---

## Step 1: Basic Logging (Already Implemented)

The FastAPI app already logs:

- Request timestamps
- Prediction results (class and confidence)
- Latency for each request
- Errors

View logs on VM:

```bash
docker-compose logs -f api
```

---

## Step 2: Add Prometheus Metrics (Optional)

If you want to add Prometheus metrics endpoint:

### 2.1 Install prometheus-fastapi-instrumentator

Add to `requirements-api.txt`:

```
prometheus-fastapi-instrumentator==6.1.0
```

### 2.2 Update app/main.py

Add after creating the app:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

This will expose metrics at `/metrics` endpoint.

### 2.3 Setup Prometheus on VM (Optional)

```bash
# Create prometheus config
cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'cats-dogs-api'
    static_configs:
      - targets: ['api:8000']
EOF

# Add to docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

---

## Step 3: Post-Deployment Performance Tracking

### 3.1 Create test script

Create a script to test model performance on a batch of images:

```python
# test_batch.py
import requests
import os
from pathlib import Path

API_URL = "http://YOUR_VM_IP:8000/predict"
TEST_DIR = "data/processed/test"

results = {"correct": 0, "total": 0}

for class_name in ["cats", "dogs"]:
    class_dir = Path(TEST_DIR) / class_name
    expected = "cat" if class_name == "cats" else "dog"

    for img_path in list(class_dir.glob("*.jpg"))[:50]:  # Test 50 per class
        with open(img_path, "rb") as f:
            response = requests.post(API_URL, files={"file": f})

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            results["total"] += 1
            if prediction == expected:
                results["correct"] += 1

accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
print(f"Post-deployment accuracy: {accuracy:.2%}")
print(f"Tested {results['total']} images")
```

### 3.2 Run the test

```bash
python test_batch.py
```

Document the results in your submission.

---

## Step 4: Final Submission Checklist

### Code and Configuration

- [ ] All source code in GitHub repo
- [ ] requirements.txt with pinned versions
- [ ] Dockerfile and docker-compose.yml
- [ ] CircleCI config (.circleci/config.yml)
- [ ] DVC tracking for data (data/raw.dvc)

### Documentation

- [ ] README.md with project overview
- [ ] Setup instructions
- [ ] API documentation (available at /docs)

### Screenshots to Include

- [ ] MLflow experiment runs (if used)
- [ ] CircleCI pipeline (green build)
- [ ] Docker Hub repository with pushed image
- [ ] API health check response
- [ ] Sample prediction response
- [ ] GCP VM running the container

### Metrics to Report

- [ ] Model accuracy on test set
- [ ] Post-deployment accuracy
- [ ] API latency (from logs)

---

## Step 5: Prepare Submission Package

1. Export MLflow experiments (if used):

   ```bash
   mlflow experiments search --view all
   ```

2. Take screenshots of:
   - CircleCI dashboard showing successful pipeline
   - Docker Hub showing pushed images
   - API responses (health and predict)

3. Create final README.md with:
   - Project overview
   - Architecture diagram
   - How to run locally
   - How to deploy
   - Results and metrics

4. Push everything to GitHub:
   ```bash
   git add -A
   git commit -m "Final submission"
   git push
   ```
