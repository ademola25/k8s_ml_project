# Step-by-Step Guide to Deploy and Test ML Service on Kubernetes

## Project Overview
This project demonstrates a ML service deployment on Kubernetes with the following features:
- Sentiment analysis model using scikit-learn
- FastAPI-based REST API
- Kubernetes deployment with auto-scaling
- Prometheus monitoring
- Persistent storage for model artifacts

## Prerequisites
1. Docker installed
2. Kubernetes cluster (minikube or cloud provider)
3. kubectl CLI tool
4. Python 3.8+

## Step 1: Local Development and Testing

### 1.1. Set up Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1.2. Train the Model
```bash
# Navigate to model directory
cd src/model

# Run training script
python train_script.py
```

### 1.3. Test API Locally
```bash
# Start FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# To view swagger api docs
http://localhost:8000/docs

# Test API endpoints
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing!"}'

# Check health endpoint
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics
```

## Step 2: Containerization

### 2.1. Build Docker Image
```bash
docker build -t ml-k8s-demo:latest .

# Test container locally
docker run -p 8000:8000 ml-k8s-demo:latest
```

## Push to Docker hub
```bash
docker tag ml-k8s-demo:latest docker.io/ademola25/ml-tutorial:latest

docker push docker.io/ademola25/ml-tutorial:latest

# If using podman
podman tag ml-k8s-demo:latest docker.io/ademola25/ml-tutorial:latest

podman push docker.io/ademola25/ml-tutorial:latest

```

## Step 3: Kubernetes Deployment

### 3.1. Create Persistent Volume
```bash
kubectl apply -f k8s/persistence.yaml
```

### 3.2. Create ConfigMap
```bash
kubectl apply -f k8s/configmap.yaml
```

### 3.3. Deploy Application
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 3.4. Deploy Prometheus
```bash
kubectl apply -f k8s/prometheus.yaml
```

## Step 4: Verify Deployment

### 4.1. Check Deployment Status
```bash
# Check pods
kubectl get pods

# Check services
kubectl get svc

# Check HPA
kubectl get hpa
```

### 4.2. Test Deployed Service
```bash

kubectl port-forward deployment/ml-service 8000:8000 

# Get service URL (if using minikube)
minikube service ml-service --url

# Or get LoadBalancer IP (if using cloud provider)
kubectl get svc ml-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Test prediction endpoint
curl -X POST "http://<SERVICE_URL>/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing!"}'
```

### 4.3. Monitor Metrics
```bash
# Access Prometheus dashboard
kubectl port-forward svc/prometheus 9090:9090
# Visit http://localhost:9090 in your browser
```

## Step 5: Testing Individual Components

### 5.1. Model Testing
```python
# Test model prediction
from src.model.trainer import ModelTrainer

model = ModelTrainer.load_model()
prediction = model.predict(["This is a great product!"])
print(f"Prediction: {prediction}")
```

### 5.2. API Testing
```python
# Test API endpoints using requests
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(f"Health check: {response.json()}")

# Prediction
data = {"text": "This product is amazing!"}
response = requests.post("http://localhost:8000/predict", json=data)
print(f"Prediction response: {response.json()}")
```

### 5.3. Load Testing
```bash
# Using hey tool for load testing
hey -n 1000 -c 100 -m POST -H "Content-Type: application/json" \
    -d '{"text":"This is a test"}' \
    http://<SERVICE_URL>/predict
```

## Common Issues and Troubleshooting
Check if the pods are running and ready:
```bash
kubectl get pods -l app=ml-service
```

1. If pods are not starting:
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

2. If model loading fails:
```bash
# Check if PVC is properly bound
kubectl get pvc
kubectl describe pvc model-storage
```

3. If HPA is not scaling:
```bash
kubectl describe hpa ml-service-hpa
```

4. If metrics are not showing in Prometheus:
```bash
kubectl port-forward svc/prometheus 9090:9090
# Check targets in Prometheus UI: http://localhost:9090/targets

kubectl logs -l app=prometheus

```