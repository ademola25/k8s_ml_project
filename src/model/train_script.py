import pandas as pd
from trainer import ModelTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load sample sentiment analysis data"""
    # Sample data - in production, replace with your actual dataset
    data = {
        'text': [
            "This product is amazing!",
            "Terrible experience, would not recommend",
            "Pretty good service overall",
            "Not worth the money at all",
            "Absolutely love it, best purchase ever"
        ],
        'sentiment': [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative
    }
    return pd.DataFrame(data)

def main():
    # Load data
    logger.info("Loading training data...")
    df = load_sample_data()
    
    # Initialize and train model
    trainer = ModelTrainer()
    logger.info("Training model...")
    trainer.train(df['text'], df['sentiment'])
    
    # Save the model
    trainer.save_model()
    logger.info("Model training completed and saved")

if __name__ == "__main__":
    main()