from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from src.components.data_preprocessing import TextPreprocessor
from src.utils.logger import logging
import os

app = FastAPI(title="Review Analysis API")
preprocessor = TextPreprocessor()

# Global variables for models
recommendation_model = None
sentiment_model = None
vectorizer = None

class ReviewInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    recommendation: bool
    sentiment: int
    recommendation_confidence: float
    sentiment_confidence: float
    recommendation_probabilities: dict
    sentiment_probabilities: dict
    cleaned_text: str

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global recommendation_model, sentiment_model, vectorizer
    try:
        recommendation_model = joblib.load("models/recommendation_model.pkl")
        sentiment_model = joblib.load("models/sentiment_model.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(review: ReviewInput):
    if None in (recommendation_model, sentiment_model, vectorizer):
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please try again in a few moments."
        )
    
    try:
        # Preprocess text
        cleaned_text = preprocessor.clean_text(review.text)
        if not cleaned_text:
            raise HTTPException(
                status_code=400,
                detail="Invalid input text after cleaning"
            )
        
        # Vectorize text
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Get recommendation prediction
        rec_prob = recommendation_model.predict_proba(text_vectorized)[0]
        rec_pred = 1 if rec_prob[1] > 0.7 else 0  # Use threshold for recommendation
        
        # Get sentiment prediction
        sent_prob = sentiment_model.predict_proba(text_vectorized)[0]
        sent_pred = int(np.argmax(sent_prob) + 1)  # Add 1 since ratings are 1-5
        
        # Create probability dictionaries
        rec_probs = {
            "Not Recommended": float(rec_prob[0]),
            "Recommended": float(rec_prob[1])
        }
        
        sent_probs = {
            f"Rating {i+1}": float(p) 
            for i, p in enumerate(sent_prob)
        }
        
        return PredictionResponse(
            recommendation=bool(rec_pred),
            sentiment=sent_pred,
            recommendation_confidence=float(max(rec_prob)),
            sentiment_confidence=float(max(sent_prob)),
            recommendation_probabilities=rec_probs,
            sentiment_probabilities=sent_probs,
            cleaned_text=cleaned_text
        )
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": all([recommendation_model, sentiment_model, vectorizer])
    }