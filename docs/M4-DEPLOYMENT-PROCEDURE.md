# M4: Deployment Procedure

This document describes the steps to set up deployment on GCP VM with CircleCI.

## Prerequisites

- GCP account with a VM instance
- Docker Hub account
- CircleCI account connected to GitHub repo

---

## Step 1: GCP VM Setup

### 1.1 Create or use existing VM

If you already have a VM from Assignment-1, you can reuse it. Otherwise:

1. Go to GCP Console > Compute Engine > VM Instances
2. Create a new instance:
   - Name: `mlops-assignment2`
   - Region: Choose one close to you
   - Machine type: `e2-medium` (2 vCPU, 4GB RAM) is enough
   - Boot disk: Ubuntu 22.04 LTS, 20GB
   - Firewall: Allow HTTP and HTTPS traffic

### 1.2 Install Docker on VM

SSH into your VM and run:

```bash
# Update packages
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io docker-compose

# Add your user to docker group
sudo usermod -aG docker $USER

# Logout and login again for group changes to take effect
exit
```

### 1.3 Open firewall port 8000

```bash
# On GCP Console: VPC Network > Firewall > Create Firewall Rule
# Name: allow-api-8000
# Targets: All instances
# Source IP ranges: 0.0.0.0/0
# Protocols and ports: tcp:8000
```

### 1.4 Create project directory on VM

```bash
mkdir -p ~/cats-dogs-classifier
cd ~/cats-dogs-classifier
```

### 1.5 Create docker-compose.yml on VM

```bash
cat > docker-compose.yml << 'EOF'
services:
  api:
    image: YOUR_DOCKERHUB_USERNAME/cats-dogs-api:latest
    container_name: cats-dogs-api
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF
```

Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username.

---

## Step 2: Docker Hub Setup

### 2.1 Create Docker Hub account

Go to https://hub.docker.com and create an account if you dont have one.

### 2.2 Create access token

1. Go to Account Settings > Security > Access Tokens
2. Create a new token with Read/Write permissions
3. Save the token (you will need it for CircleCI)

---

## Step 3: CircleCI Setup

### 3.1 Connect repository

1. Go to https://circleci.com
2. Sign in with GitHub
3. Click "Set Up Project" for `Group75-MLops-Assignment2`
4. Select "Fastest" option (use existing config)

### 3.2 Create Docker Hub context

1. Go to Organization Settings > Contexts
2. Create new context named `docker-hub`
3. Add environment variables:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub access token

### 3.3 Create GCP VM context

1. Create new context named `gcp-vm`
2. Add environment variables:
   - `VM_HOST`: Your GCP VM external IP address
   - `VM_USER`: Your VM username (usually your email prefix)

### 3.4 Add SSH key for deployment

1. Generate SSH key pair on your local machine:

   ```bash
   ssh-keygen -t ed25519 -C "circleci-deploy" -f ~/.ssh/circleci_deploy
   ```

2. Add public key to VM:

   ```bash
   # Copy content of ~/.ssh/circleci_deploy.pub
   # Add it to ~/.ssh/authorized_keys on your VM
   ```

3. Add private key to CircleCI:
   - Go to Project Settings > SSH Keys
   - Add SSH Key (paste content of ~/.ssh/circleci_deploy)
   - Copy the fingerprint

4. Update `.circleci/config.yml`:
   - Replace `${SSH_FINGERPRINT}` with the actual fingerprint

---

## Step 4: Test Deployment

### 4.1 Trigger pipeline

Push any change to master branch to trigger the pipeline:

```bash
git commit --allow-empty -m "Trigger deployment"
git push
```

### 4.2 Monitor pipeline

1. Go to CircleCI dashboard
2. Watch the pipeline progress through: test -> build -> publish -> deploy -> smoke-test

### 4.3 Verify deployment

```bash
# From your local machine
curl http://YOUR_VM_IP:8000/health

# Should return:
# {"status":"healthy","version":"1.0.0","model_loaded":true}
```

---

## Troubleshooting

### Pipeline fails at publish

- Check Docker Hub credentials in context
- Make sure access token has write permissions

### Pipeline fails at deploy

- Check SSH key is correctly added
- Verify VM_HOST and VM_USER are correct
- Make sure docker-compose.yml exists on VM

### Health check fails

- SSH into VM and check container logs:
  ```bash
  docker-compose logs
  ```
- Make sure port 8000 is open in firewall
