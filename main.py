from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import uvicorn
import logging
from typing import Dict, Any, Optional
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Detector API",
    description="API for detecting potential mental health issues from text",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Download NLTK data at startup
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str
    detailed: bool = False

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    condition: Optional[str] = None
    recommendation: Optional[str] = None
    processed_text: Optional[str] = None

class MentalHealthDetector:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.model = None
        
    def preprocess_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^a-zA-Z]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            tokens = nltk.word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
            return ' '.join(tokens)
        return ""
    
    def load_models(self, model_path, vectorizer_path):
        try:
            self.model = joblib.load(model_path)
            self.tfidf_vectorizer = joblib.load(vectorizer_path)
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def predict(self, text, detailed=False, threshold=0.5):
        try:
            clean_text = self.preprocess_text(text)
            
            if not clean_text:
                return {
                    'prediction': 0,
                    'probability': 0.0,
                    'risk_level': "Unable to process",
                    'processed_text': ""
                }
            
            text_tfidf = self.tfidf_vectorizer.transform([clean_text])
            
            pred_prob = self.model.predict_proba(text_tfidf)[0, 1]
            pred_class = 1 if pred_prob > threshold else 0
            
            logger.info(f"Input: {text} → Probability: {pred_prob:.3f}, Class: {pred_class}")
            
            # Risk level determination
            if pred_prob < 0.3:
                risk_level = "Low Risk"
            elif pred_prob < 0.5:
                risk_level = "Low-Moderate Risk"
            elif pred_prob < 0.7:
                risk_level = "Moderate Risk"
            elif pred_prob < 0.85:
                risk_level = "Moderate-High Risk"
            else:
                risk_level = "High Risk"
            
            result = {
                'prediction': pred_class,
                'probability': float(pred_prob),
                'risk_level': risk_level
            }
            
            if detailed:
                # Add condition and recommendation for detailed response
                if pred_class == 1:
                    if pred_prob > 0.9:
                        result['condition'] = "Severe mental health concern detected"
                    elif pred_prob > 0.7:
                        result['condition'] = "Significant mental health concern detected"
                    else:
                        result['condition'] = "Potential mental health concern detected"
                        
                    if pred_prob > 0.7:
                        result['recommendation'] = "Consider professional mental health support"
                    else:
                        result['recommendation'] = "Consider speaking with a trusted individual about your feelings"
                else:
                    result['recommendation'] = "No specific mental health recommendations at this time"
                
                result['processed_text'] = clean_text
            
            return result
        except RuntimeError as e:
            logger.error(f"Model error: {e}")
            raise HTTPException(status_code=503, detail="Model unavailable")

@lru_cache()
def get_detector():
    detector = MentalHealthDetector()
    # Paths to saved models - adjust these for your deployment
    model_paths = {
        "local": {
            "model": "models/mental_health_traditional_model.joblib",
            "vectorizer": "models/mental_health_tfidf_vectorizer.joblib"
        },
        "render": {
            "model": "/app/models/mental_health_traditional_model.joblib",
            "vectorizer": "/app/models/mental_health_tfidf_vectorizer.joblib"
        }
    }
    
    # Try loading from Render path first, then local
    if os.path.exists(model_paths["render"]["model"]):
        success = detector.load_models(
            model_paths["render"]["model"], 
            model_paths["render"]["vectorizer"]
        )
    else:
        success = detector.load_models(
            model_paths["local"]["model"], 
            model_paths["local"]["vectorizer"]
        )
    
    if not success:
        raise RuntimeError("Model loading failed")
    return detector

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TextInput, detector: MentalHealthDetector = Depends(get_detector)):
    try:
        prediction = detector.predict(input_data.text, detailed=input_data.detailed)
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.on_event("startup")
async def startup_load_model():
    # Remove model loading logic here as it's handled by get_detector
    pass

if __name__ == "__main__":
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)