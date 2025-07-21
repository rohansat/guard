"""
Intent Classification Module for Guard AI
Understands user commands and extracts intents for safety-related actions
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from typing import Dict, List, Tuple, Any
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class IntentClassifier(nn.Module):
    """
    Neural network for intent classification
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_intents: int = 10):
        super(IntentClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class GuardIntentClassifier:
    """
    Main intent classification system for Guard AI
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the intent classifier
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define intents for Guard AI
        self.intents = {
            0: "location_tracking",
            1: "emergency_alert",
            2: "check_in",
            3: "facial_recognition",
            4: "route_monitoring",
            5: "contact_update",
            6: "safety_check",
            7: "voice_command",
            8: "system_status",
            9: "unknown"
        }
        
        self.intent_examples = {
            "location_tracking": [
                "track my location",
                "follow me",
                "monitor where I am",
                "keep track of my position"
            ],
            "emergency_alert": [
                "help me",
                "emergency",
                "call for help",
                "alert authorities",
                "I'm in danger"
            ],
            "check_in": [
                "check in",
                "I'm safe",
                "mark me as safe",
                "update my status"
            ],
            "facial_recognition": [
                "scan faces",
                "recognize people",
                "identify who's around",
                "check for known faces"
            ],
            "route_monitoring": [
                "monitor my route",
                "watch my path",
                "track my journey",
                "follow my route"
            ],
            "contact_update": [
                "update contacts",
                "add emergency contact",
                "change trusted contacts",
                "modify contact list"
            ],
            "safety_check": [
                "how safe is this area",
                "assess safety",
                "check surroundings",
                "evaluate risk"
            ],
            "voice_command": [
                "listen to me",
                "voice activation",
                "start voice mode",
                "enable voice commands"
            ],
            "system_status": [
                "system status",
                "how are you working",
                "check system",
                "status report"
            ]
        }
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        if model_path:
            self.model = IntentClassifier(num_intents=len(self.intents))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            self.model = IntentClassifier(num_intents=len(self.intents))
            
        self.model.to(self.device)
        self.model.eval()
        
        # Context management
        self.conversation_history = []
        self.max_history_length = 10
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for intent classification
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Add Guard AI specific preprocessing
        if "guard" in text:
            text = text.replace("guard", "").strip()
            
        return text
    
    def classify_intent(self, text: str, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Classify the intent of the given text
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence for classification
            
        Returns:
            Dictionary with intent and confidence
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get intent name
            intent_name = self.intents.get(predicted_class, "unknown")
            
            # Add to conversation history
            self.add_to_history(text, intent_name, confidence)
            
            return {
                "intent": intent_name,
                "confidence": confidence,
                "text": text,
                "processed_text": processed_text,
                "is_confident": confidence >= confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "text": text,
                "processed_text": text,
                "is_confident": False
            }
    
    def extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """
        Extract entities from text based on intent
        
        Args:
            text: Input text
            intent: Classified intent
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        try:
            text_lower = text.lower()
            
            # Extract location-related entities
            if intent == "location_tracking":
                location_keywords = ["home", "work", "gym", "store", "restaurant", "park"]
                for keyword in location_keywords:
                    if keyword in text_lower:
                        entities["location"] = keyword
                        break
            
            # Extract time-related entities
            time_keywords = ["tonight", "today", "tomorrow", "morning", "afternoon", "evening", "night"]
            for keyword in time_keywords:
                if keyword in text_lower:
                    entities["time"] = keyword
                    break
            
            # Extract duration entities
            duration_patterns = ["for", "until", "during"]
            for pattern in duration_patterns:
                if pattern in text_lower:
                    # Simple duration extraction
                    words = text_lower.split()
                    try:
                        pattern_idx = words.index(pattern)
                        if pattern_idx + 1 < len(words):
                            entities["duration"] = words[pattern_idx + 1]
                    except ValueError:
                        continue
            
            # Extract contact-related entities
            if intent == "contact_update":
                contact_keywords = ["friend", "family", "emergency", "contact"]
                for keyword in contact_keywords:
                    if keyword in text_lower:
                        entities["contact_type"] = keyword
                        break
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            
        return entities
    
    def add_to_history(self, text: str, intent: str, confidence: float):
        """
        Add interaction to conversation history
        
        Args:
            text: User input text
            intent: Classified intent
            confidence: Classification confidence
        """
        self.conversation_history.append({
            "text": text,
            "intent": intent,
            "confidence": confidence,
            "timestamp": torch.tensor([torch.cuda.Event() if torch.cuda.is_available() else None])
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history.pop(0)
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get current conversation context
        
        Returns:
            Dictionary with conversation context
        """
        if not self.conversation_history:
            return {"recent_intents": [], "conversation_length": 0}
        
        recent_intents = [item["intent"] for item in self.conversation_history[-3:]]
        
        return {
            "recent_intents": recent_intents,
            "conversation_length": len(self.conversation_history),
            "last_intent": self.conversation_history[-1]["intent"] if self.conversation_history else None
        }
    
    def is_emergency_intent(self, intent: str) -> bool:
        """
        Check if intent is emergency-related
        
        Args:
            intent: Intent name
            
        Returns:
            True if emergency intent
        """
        emergency_intents = ["emergency_alert", "safety_check"]
        return intent in emergency_intents
    
    def generate_response(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Generate appropriate response based on intent and context
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            context: Conversation context
            
        Returns:
            Generated response
        """
        responses = {
            "location_tracking": "I'll track your location and keep you safe.",
            "emergency_alert": "I'm here to help. Stay calm and I'll assist you immediately.",
            "check_in": "Thank you for checking in. I'll update your status.",
            "facial_recognition": "I'll scan the area and identify people around you.",
            "route_monitoring": "I'll monitor your route and alert you to any concerns.",
            "contact_update": "I'll help you update your emergency contacts.",
            "safety_check": "I'm analyzing your surroundings for safety assessment.",
            "voice_command": "Voice commands are active. I'm listening.",
            "system_status": "All systems are operational and monitoring your safety.",
            "unknown": "I didn't understand that. Could you please rephrase?"
        }
        
        base_response = responses.get(intent, responses["unknown"])
        
        # Add context-specific modifications
        if intent == "location_tracking" and entities.get("location"):
            base_response = f"I'll track your location to {entities['location']} and keep you safe."
        
        if intent == "emergency_alert":
            base_response = "EMERGENCY MODE ACTIVATED. I'm contacting authorities and your emergency contacts immediately."
        
        return base_response 