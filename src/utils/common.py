import os
import sys
import yaml
import joblib
from pathlib import Path
from src.utils.logger import logging
from src.utils.exception import CustomException
from sklearn.metrics import mean_absolute_error, mean_squared_error


def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(e, sys)

def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Created directory at: {path}")

def save_model(model, vectorizer, model_dir="models"):
    try:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "best_model.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        logging.info(f"Model saved at: {model_path}")
        logging.info(f"Vectorizer saved at: {vectorizer_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_model(model_dir="models"):
    try:
        recommendation_path = os.path.join(model_dir, "recommendation_model.pkl")
        sentiment_path = os.path.join(model_dir, "sentiment_model.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        
        if not all(os.path.exists(p) for p in [recommendation_path, sentiment_path, vectorizer_path]):
            return None, None, None
            
        recommendation_model = joblib.load(recommendation_path)
        sentiment_model = joblib.load(sentiment_path)
        vectorizer = joblib.load(vectorizer_path)
        return recommendation_model, sentiment_model, vectorizer
    except Exception as e:
        raise CustomException(e, sys)