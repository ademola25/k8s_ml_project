# Complete MLOps Project Tutorial
## From Basics to Advanced Implementation

This comprehensive tutorial walks through a complete MLOps project, covering everything from local development to production deployment. Perfect for beginners looking to advance their MLOps skills and understand real-world machine learning systems.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Local Development Setup](#local-development-setup)
5. [Model Development](#model-development)
6. [API Development](#api-development)
7. [Containerization](#containerization)
8. [Kubernetes Deployment](#kubernetes-deployment)
9. [Monitoring Setup](#monitoring-setup)
10. [Advanced Concepts](#advanced-concepts)

## Project Overview
This project demonstrates a complete MLOps pipeline that includes model training, API development, containerization, orchestration, and monitoring. The system is built with scalability, maintainability, and production-readiness in mind.

## Technology Stack
Let's start by understanding our technology stack from the requirements.txt file:

```python
numpy==1.21.0        # For numerical computations and array operations
pandas==1.3.0        # For data manipulation and analysis
scikit-learn==0.24.2 # For machine learning algorithms and model training
nltk==3.6.3          # For natural language processing tasks
fastapi==0.68.0      # For building the REST API
uvicorn==0.15.0      # ASGI server for FastAPI
pydantic==1.8.2      # For data validation
prometheus-client==0.11.0  # For metrics and monitoring
python-jose==3.3.0        # For JWT token handling
python-multipart==0.0.5   # For handling multipart form data
```

## Project Structure
Let's examine each component of the project:

### Model Training (`src/model/train_script.py`)
This script handles the model training pipeline. Let's examine it line by line:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from trainer import ModelTrainer

def load_sample_data():
    """
    Loads and preprocesses the training data.
    - Uses pandas to read the dataset
    - Performs necessary preprocessing steps
    - Splits features and target variables
    - Returns preprocessed X (features) and y (target)
    """
    # Data loading code here
    pass

def main():
    """
    Main training pipeline that:
    1. Loads the data using load_sample_data()
    2. Initializes the ModelTrainer
    3. Performs model training
    4. Evaluates model performance
    5. Saves the trained model
    """
    # Get training data
    X, y = load_sample_data()
    
    # Initialize and train model
    trainer = ModelTrainer()
    trainer.train(X, y)
    
    # Save the trained model
    trainer.save_model()

if __name__ == "__main__":
    main()
```

Each part of this script serves a specific purpose:
1. **Imports**: Essential libraries for data handling and model training
2. **load_sample_data()**: Handles data preprocessing pipeline
3. **main()**: Orchestrates the entire training process
4. **if __name__ == "__main__"**: Standard Python idiom for script execution

```python
def main():
    # Main training pipeline
    # 1. Loads data
    # 2. Initializes model
    # 3. Performs training
    # 4. Saves the trained model
```

### Model Training Class (`src/model/trainer.py`)
The ModelTrainer class encapsulates model-related operations. Let's analyze each component:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import logging
import os

# Configure logging for the model trainer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        """
        Initialize the model pipeline with:
        1. TF-IDF Vectorizer: Converts text to numerical features
        2. Logistic Regression: Classification model for sentiment analysis
        """
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),  # Convert text to TF-IDF features
            ('classifier', LogisticRegression(max_iter=1000))  # Classification model
        ])
        
    def train(self, X, y):
        """
        Train the model with provided data
        Args:
            X: Input features (text data)
            y: Target labels (sentiment: 0 or 1)
        """
        logger.info("Starting model training...")
        self.model.fit(X, y)  # Train the entire pipeline
        logger.info("Model training completed")
        
    def save_model(self, path="models/model.joblib"):
        """
        Save the trained model to disk
        Args:
            path: Path where the model should be saved
        """
        # Convert to absolute path for reliability
        absolute_path = os.path.abspath(path)
        # Extract the directory path
        directory = os.path.dirname(absolute_path)
        # Create model directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        # Save model using joblib
        logger.info(f"Saving model to {absolute_path}")
        joblib.dump(self.model, absolute_path)
        
    @staticmethod
    def load_model(path="src/model/models/model.joblib"):
        """
        Load a trained model from disk
        Args:
            path: Path to the saved model file
        Returns:
            Loaded model object
        """
        logger.info(f"Loading model from {path}")
        return joblib.load(path)
```

Key Components Explained:
1. **Pipeline Setup**: 
   - Uses scikit-learn's Pipeline for seamless data processing
   - TF-IDF Vectorizer converts text to numerical features
   - Logistic Regression for binary classification (sentiment analysis)

2. **Training Process**:
   - Single `train()` method handles both feature extraction and model training
   - Pipeline ensures consistent preprocessing during training and inference

3. **Model Persistence**:
   - Uses joblib for efficient model serialization
   - Handles directory creation and absolute paths
   - Includes comprehensive logging for debugging

### API Service (`src/api/main.py`)
The FastAPI application that serves predictions. Let's examine each component:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from src.model.trainer import ModelTrainer
import logging

# Initialize FastAPI app with metadata
app = FastAPI(title="ML Model Serving API")

# Prometheus metrics counter
prediction_counter = Counter('model_predictions_total', 'Total number of predictions made')

class PredictionRequest(BaseModel):
    """
    Pydantic model for request validation
    Ensures that incoming requests contain the required 'text' field
    """
    text: str

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Kubernetes
    Returns 200 if the service is healthy
    Used by K8s liveness and readiness probes
    """
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    Exposes:
    - Total prediction count
    - Response times
    - Error rates
    """
    return Response(
        generate_latest().decode('utf-8'),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Main prediction endpoint
    1. Validates input using PredictionRequest model
    2. Performs prediction using loaded model
    3. Increments metrics counter
    4. Returns prediction result
    """
    try:
        prediction_counter.inc()  # Increment prediction counter
        # Add prediction logic here
        return {"prediction": "result"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Key Features Explained:
1. **Request Validation**:
   - Uses Pydantic for automatic request validation
   - Ensures input data meets expected format

2. **Monitoring Integration**:
   - Prometheus metrics for monitoring
   - Tracks prediction counts and performance

3. **Kubernetes Integration**:
   - Health check endpoint for container orchestration
   - Supports liveness and readiness probes

4. **Error Handling**:
   - Proper exception handling with HTTP status codes
   - Structured error responses

### Kubernetes Deployment (`k8s/deployment.yaml`)
The deployment configuration manages how our application runs in Kubernetes. Let's analyze each section:

```yaml
apiVersion: apps/v1  # Kubernetes API version for Deployments
kind: Deployment     # Type of resource we're creating
metadata:           # Resource identification
  name: ml-service  # Name of our deployment
  labels:           # Labels for organizing Kubernetes resources
    app: ml-service # Label to identify this application
spec:               # Deployment specification
  replicas: 3       # Number of pod replicas to maintain
  selector:         # Defines how the deployment finds pods to manage
    matchLabels:    # Must match the pod template labels
      app: ml-service
  template:         # Template for creating pods
    metadata:       # Pod metadata
      labels:       # Labels applied to pods
        app: ml-service
    spec:           # Pod specification
      containers:   # List of containers in the pod
      - name: ml-service           # Container name
        image: abonia/ml-tutorial:latest  # Docker image to use
        ports:                     # Container ports to expose
        - containerPort: 8000      # Port our FastAPI app listens on
        envFrom:                   # Environment variables from ConfigMap
        - configMapRef:
            name: ml-service-config
        volumeMounts:             # Mount points for volumes
        - name: model-storage     # Volume name
          mountPath: /models      # Where to mount in container
        resources:               # Resource limits and requests
          requests:             # Minimum resources needed
            memory: "512Mi"     # 512 megabytes of memory
            cpu: "500m"        # 500 millicpu (0.5 CPU core)
          limits:              # Maximum resources allowed
            memory: "1Gi"      # 1 gigabyte of memory
            cpu: "1000m"      # 1000 millicpu (1 CPU core)
        readinessProbe:       # Checks if pod is ready to serve traffic
          httpGet:           # Use HTTP GET request
            path: /health   # Health check endpoint
            port: 8000     # Port to check
          initialDelaySeconds: 5   # Wait before first check
          periodSeconds: 10        # Time between checks
        livenessProbe:           # Checks if pod is alive
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
      volumes:                    # Volume definitions
      - name: model-storage      # Name matches volumeMount
        persistentVolumeClaim:   # Use persistent storage
          claimName: model-storage  # Name of PVC
```

Key Components Explained:

1. **Replicas and Scaling**:
   - `replicas: 3`: Maintains 3 identical pods for high availability
   - Kubernetes automatically restarts failed pods
   - Supports horizontal scaling (can be adjusted with HPA)

2. **Container Configuration**:
   - Uses Docker image: `abonia/ml-tutorial:latest`
   - Exposes port 8000 for the FastAPI application
   - Mounts persistent storage for model files

3. **Resource Management**:
   - Requests: Minimum guaranteed resources
   - Limits: Maximum allowed resource usage
   - Prevents resource contention between pods

4. **Health Monitoring**:
   - Readiness Probe: Checks if pod can serve traffic
   - Liveness Probe: Checks if pod is healthy
   - Automatic restart of unhealthy pods

5. **Storage**:
   - Uses persistent volume for model storage
   - Ensures model data survives pod restarts

### Kubernetes Service (`k8s/service.yaml`)
The service configuration defines how our application is exposed to users. Let's examine each part:

```yaml
apiVersion: v1    # Kubernetes API version for Services
kind: Service     # Type of resource we're creating
metadata:         # Resource identification
  name: ml-service  # Service name
spec:              # Service specification
  selector:        # Defines which pods to route traffic to
    app: ml-service  # Must match pod labels
  ports:           # Port configuration
    - port: 8000        # Port exposed by the service
      targetPort: 8000  # Port on the pods
      protocol: TCP     # Network protocol
  type: LoadBalancer   # Service type for external access
```

Key Components Explained:

1. **Service Type**:
   - `LoadBalancer`: Exposes service externally through cloud provider's load balancer
   - Alternatives include:
     - `ClusterIP`: Internal-only access
     - `NodePort`: Exposes on each node's IP
     - `ExternalName`: Maps to external service

2. **Port Configuration**:
   - `port`: The port exposed by the service
   - `targetPort`: The port on the pod where the application listens
   - `protocol`: Network protocol (TCP/UDP)

3. **Pod Selection**:
   - Uses label selector to find pods
   - Automatically load balances between matching pods
   - Updates dynamically as pods come and go

### Monitoring (`k8s/prometheus.yaml`)
The Prometheus configuration sets up monitoring for our application. Let's analyze each component:

```yaml
apiVersion: v1          # Kubernetes API version
kind: ConfigMap        # Resource type for configuration
metadata:              # Resource identification
  name: prometheus-config  # ConfigMap name
data:
  prometheus.yml: |     # Prometheus configuration file
    global:            # Global settings
        scrape_interval: 15s  # How often to scrape metrics
    scrape_configs:    # List of metrics sources
      - job_name: 'ml-service'  # Name for this scrape job
        static_configs:         # Static target configuration
          - targets: ['ml-service:8000']  # Service to monitor
---
apiVersion: apps/v1    # Kubernetes API version for Deployments
kind: Deployment       # Deploy Prometheus server
metadata:
  name: prometheus     # Deployment name
spec:
  selector:           # Pod selector
    matchLabels:
      app: prometheus
  template:           # Pod template
    metadata:
      labels:
        app: prometheus
    spec:
      containers:     # Container configuration
      - name: prometheus
        image: prom/prometheus  # Official Prometheus image
        ports:
        - containerPort: 9090  # Prometheus web interface
        volumeMounts:         # Mount configuration
        - name: config
          mountPath: /etc/prometheus
      volumes:               # Volume definitions
      - name: config        # Configuration volume
        configMap:          # Use our ConfigMap
          name: prometheus-config
```

Key Components Explained:

1. **Prometheus Configuration**:
   - `scrape_interval`: Defines metric collection frequency
   - `scrape_configs`: Specifies where to collect metrics from
   - Targets our ML service endpoints

2. **Deployment Setup**:
   - Uses official Prometheus image
   - Mounts configuration from ConfigMap
   - Exposes web interface on port 9090

3. **Integration with Application**:
   - Automatically discovers and scrapes our service
   - Collects metrics from `/metrics` endpoint
   - Stores time-series data for analysis

4. **Metrics Collection**:
   - Request count and latency
   - Error rates
   - Custom model metrics

## Local Development Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Development
1. Prepare your data
2. Modify model parameters in `src/model/trainer.py`
3. Run training:
   ```bash
   python src/model/train_script.py
   ```

## API Development
1. Start the FastAPI server:
   ```bash
   uvicorn src.api.main:app --reload
   ```
2. Access the API documentation at `http://localhost:8000/docs`

## Kubernetes Deployment
1. Build and push the Docker image
2. Apply Kubernetes configurations:
   ```bash
   kubectl apply -f k8s/
   ```

## Monitoring Setup
1. Deploy Prometheus:
   ```bash
   kubectl apply -f k8s/prometheus.yaml
   ```
2. Access metrics at `/metrics` endpoint

## Advanced Concepts

### Horizontal Pod Autoscaling
The deployment uses HPA for automatic scaling based on metrics.

### Prometheus Integration
Custom metrics are exposed for:
- Request latency
- Prediction counts
- Error rates

### Best Practices
1. **Version Control**: All code and configurations are version controlled
2. **CI/CD**: Automated testing and deployment pipeline
3. **Monitoring**: Comprehensive metrics collection
4. **Scalability**: Horizontal scaling with Kubernetes
5. **Security**: JWT authentication and secure endpoints

## Troubleshooting
Common issues and solutions:
1. Model loading errors: Check model path in configuration
2. API timeout: Adjust resource limits in deployment.yaml
3. Metrics not showing: Verify Prometheus configuration

## Next Steps
1. Implement A/B testing
2. Add model versioning
3. Set up distributed training
4. Implement feature stores

This tutorial provides a complete walkthrough of setting up a production-ready ML system. For specific questions or issues, please refer to the respective documentation or raise an issue in the repository.