import fastapi
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Union
import warnings
warnings.filterwarnings('ignore')
from pydantic import BaseModel, Field



app = fastapi.FastAPI()

class MLToxicityDetector:
    def __init__(self, model_name: str = "unitary/toxic-bert"):
        """
        Initialize the toxicity detector with a HuggingFace model.
        
        Args:
            model_name (str): Name of the pre-trained model from HuggingFace.
                            Default is "unitary/toxic-bert"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Labels for toxic-bert model
        self.labels = [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack"
        ]
    
    def analyze_text(self, text: str) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Analyze text for different types of toxic content.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Analysis results including toxicity scores and details
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        
        # Create detailed analysis
        label_scores = {
            label: float(score)
            for label, score in zip(self.labels, scores)
        }
        
        # Calculate overall toxicity score (weighted average)
        weights = {
            "toxicity": 1.0,
            "severe_toxicity": 1.5,
            "obscene": 0.8,
            "threat": 1.2,
            "insult": 0.8,
            "identity_attack": 1.2
        }
        
        weighted_score = sum(label_scores[label] * weights[label] 
                           for label in self.labels) / sum(weights.values())
        
        return {
            "overall_toxicity": float(weighted_score),
            "is_toxic": weighted_score > 0.5,
            "category_scores": label_scores
        }

# 
class ToxicityRequest(BaseModel):
    text: str = Field(default="You're completely wrong and stupid!", description="The text to analyze for toxicity")
    
@app.post("/predict_toxicity")
def predict_toxicity(request: ToxicityRequest):
    toxicity_detector = MLToxicityDetector()
    return toxicity_detector.analyze_text(request.text)
  
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7444)
