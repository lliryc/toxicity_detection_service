import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Union
import warnings
warnings.filterwarnings('ignore')

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
    
    def batch_analyze(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Analyze multiple texts efficiently in batches.
        
        Args:
            texts (List[str]): List of texts to analyze
            batch_size (int): Size of batches for processing
            
        Returns:
            List[Dict]: List of analysis results for each text
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.sigmoid(outputs.logits).cpu().numpy()
            
            for score_set in scores:
                label_scores = {
                    label: float(score)
                    for label, score in zip(self.labels, score_set)
                }
                
                weighted_score = sum(label_scores[label] * 1.0 
                                   for label in self.labels) / len(self.labels)
                
                results.append({
                    "overall_toxicity": float(weighted_score),
                    "is_toxic": weighted_score > 0.5,
                    "category_scores": label_scores
                })
        
        return results

    def get_improvement_suggestions(self, analysis: Dict) -> List[str]:
        """
        Generate suggestions for improving text based on toxicity analysis.
        
        Args:
            analysis (Dict): Analysis results from analyze_text()
            
        Returns:
            List[str]: List of improvement suggestions
        """
        suggestions = []
        
        if analysis["is_toxic"]:
            scores = analysis["category_scores"]
            
            if scores["severe_toxicity"] > 0.5:
                suggestions.append("The text contains severely toxic content that should be completely revised.")
            
            if scores["threat"] > 0.5:
                suggestions.append("Remove threatening language to make the text more appropriate.")
                
            if scores["identity_attack"] > 0.5:
                suggestions.append("Remove content that may be discriminatory or attacking specific groups.")
                
            if scores["obscene"] > 0.5:
                suggestions.append("Consider removing obscene language to make the text more appropriate.")
                
            if scores["insult"] > 0.5:
                suggestions.append("Rephrase insulting content in a more constructive way.")
                
        if not suggestions:
            suggestions.append("No specific improvements needed.")
            
        return suggestions

def main():
    # Initialize detector
    detector = MLToxicityDetector()
    
    # Example texts
    sample_texts = [
        "Let's work together to solve this problem!",
        "You're completely wrong and stupid!",
        "I respectfully disagree with your perspective.",
        "This makes me so angry I want to break something!"
    ]
    
    print("\nAnalyzing individual texts:")
    for text in sample_texts:
        print("\nText:", text)
        analysis = detector.analyze_text(text)
        print(analysis)
        print("Overall Toxicity:", f"{analysis['overall_toxicity']:.3f}")
        print("Is Toxic:", analysis['is_toxic'])
        print("Category Scores:")
        for category, score in analysis['category_scores'].items():
            print(f"  {category}: {score:.3f}")
        print("Suggestions:", detector.get_improvement_suggestions(analysis))
    
    print("\nBatch analysis example:")
    batch_results = detector.batch_analyze(sample_texts)
    for text, result in zip(sample_texts, batch_results):
        print(f"\nText: {text}")
        print(f"Overall Toxicity: {result['overall_toxicity']:.3f}")

if __name__ == "__main__":
    main()