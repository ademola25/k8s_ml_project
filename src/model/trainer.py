import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        
    def train(self, X, y):
        """Train the model with provided data"""
        logger.info("Starting model training...")
        self.model.fit(X, y)
        logger.info("Model training completed")
        
    def save_model(self, path="models/model.joblib"):
        """Save the trained model to disk"""
        # Convert to absolute path
        absolute_path = os.path.abspath(path)
        # Extract the directory from the absolute path
        directory = os.path.dirname(absolute_path)
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        # Save the model
        logger.info(f"Saving model to {absolute_path}")
        joblib.dump(self.model, absolute_path)
        
    @staticmethod
    def load_model(path="src/model/models/model.joblib"):
        """Load a trained model from disk"""
        logger.info(f"Loading model from {path}")
        return joblib.load(path)