"""
Facial Recognition Module for Guard AI
Detects and identifies known/unknown individuals for situational awareness
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import face_recognition
import pickle
import os
import logging
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Face detection result"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    face_encoding: np.ndarray
    timestamp: datetime


@dataclass
class PersonIdentity:
    """Person identity information"""
    person_id: str
    name: str
    face_encoding: np.ndarray
    trust_level: float  # 0.0 = unknown, 1.0 = trusted
    last_seen: datetime
    encounter_count: int


class MobileFaceNet(nn.Module):
    """
    Lightweight CNN for mobile facial recognition
    Optimized for edge deployment
    """
    
    def __init__(self, embedding_size: int = 128, num_classes: int = 1000):
        super(MobileFaceNet, self).__init__()
        
        self.embedding_size = embedding_size
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Depthwise separable convolutions
        self.conv2_dw = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.conv2_pw = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3_dw = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128)
        self.conv3_pw = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4_dw = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        self.conv4_pw = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5_dw = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256)
        self.conv5_pw = nn.Conv2d(256, 512, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding layer
        self.embedding = nn.Linear(512, embedding_size)
        
        # Classification head (optional)
        self.classifier = nn.Linear(embedding_size, num_classes)
        
    def forward(self, x, return_embedding: bool = True):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Depthwise separable convolutions
        x = F.relu(self.bn2(self.conv2_pw(self.conv2_dw(x))))
        x = F.relu(self.bn3(self.conv3_pw(self.conv3_dw(x))))
        x = F.relu(self.bn4(self.conv4_pw(self.conv4_dw(x))))
        x = F.relu(self.bn5(self.conv5_pw(self.conv5_dw(x))))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Embedding
        embedding = self.embedding(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        if return_embedding:
            return embedding
        else:
            # Classification
            output = self.classifier(embedding)
            return output


class FaceRecognitionSystem:
    """
    Main facial recognition system for Guard AI
    """
    
    def __init__(self, model_path: str = None, database_path: str = "face_database.pkl"):
        """
        Initialize the facial recognition system
        
        Args:
            model_path: Path to pre-trained model (optional)
            database_path: Path to face database
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize recognition model
        self.model = MobileFaceNet(embedding_size=128)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.model.to(self.device)
        self.model.eval()
        
        # Face database
        self.database_path = database_path
        self.known_faces: Dict[str, PersonIdentity] = {}
        self.load_database()
        
        # Recognition parameters
        self.face_threshold = 0.6
        self.recognition_threshold = 0.7
        self.min_face_size = 20
        self.max_face_size = 300
        
        # Suspicious behavior tracking
        self.suspicious_encounters: List[Dict[str, Any]] = []
        self.unknown_face_history: List[FaceDetection] = []
        
    def load_database(self):
        """Load face database from file"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
                logger.info(f"Loaded {len(self.known_faces)} known faces from database")
            else:
                logger.info("No existing face database found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading face database: {e}")
            self.known_faces = {}
            
    def save_database(self):
        """Save face database to file"""
        try:
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.known_faces, f)
            logger.info(f"Saved {len(self.known_faces)} faces to database")
        except Exception as e:
            logger.error(f"Error saving face database: {e}")
            
    def preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for recognition
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        face_img = cv2.resize(face_img, (112, 112))
        
        # Normalize
        face_img = face_img.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0)
        
        return face_tensor.to(self.device)
        
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face detections
        """
        detections = []
        
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
                maxSize=(self.max_face_size, self.max_face_size)
            )
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = image[y:y+h, x:x+w]
                
                # Get face encoding using face_recognition library
                face_encoding = face_recognition.face_encodings(face_img)
                
                if face_encoding:
                    detection = FaceDetection(
                        bbox=(x, y, w, h),
                        confidence=0.9,  # Default confidence for detected faces
                        face_encoding=face_encoding[0],
                        timestamp=datetime.now()
                    )
                    detections.append(detection)
                    
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            
        return detections
        
    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face from the database
        
        Args:
            face_encoding: Face encoding
            
        Returns:
            Tuple of (person_id, confidence)
        """
        best_match = None
        best_confidence = 0.0
        
        for person_id, person in self.known_faces.items():
            # Calculate distance between face encodings
            distance = face_recognition.face_distance([person.face_encoding], face_encoding)[0]
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = 1.0 - distance
            
            if confidence > best_confidence and confidence > self.recognition_threshold:
                best_confidence = confidence
                best_match = person_id
                
        return best_match, best_confidence
        
    def add_person(self, person_id: str, name: str, face_encodings: List[np.ndarray], 
                   trust_level: float = 1.0) -> bool:
        """
        Add a new person to the database
        
        Args:
            person_id: Unique identifier
            name: Person's name
            face_encodings: List of face encodings
            trust_level: Trust level (0.0-1.0)
            
        Returns:
            True if successfully added
        """
        try:
            if not face_encodings:
                return False
                
            # Use the first encoding as the primary one
            primary_encoding = face_encodings[0]
            
            person = PersonIdentity(
                person_id=person_id,
                name=name,
                face_encoding=primary_encoding,
                trust_level=trust_level,
                last_seen=datetime.now(),
                encounter_count=0
            )
            
            self.known_faces[person_id] = person
            self.save_database()
            
            logger.info(f"Added person {name} (ID: {person_id}) to database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding person to database: {e}")
            return False
            
    def update_person_encounter(self, person_id: str):
        """Update encounter statistics for a person"""
        if person_id in self.known_faces:
            person = self.known_faces[person_id]
            person.encounter_count += 1
            person.last_seen = datetime.now()
            self.save_database()
            
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image for face detection and recognition
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with detection and recognition results
        """
        results = {
            'faces_detected': 0,
            'known_faces': [],
            'unknown_faces': [],
            'suspicious_activity': False,
            'alerts': []
        }
        
        try:
            # Detect faces
            face_detections = self.detect_faces(image)
            results['faces_detected'] = len(face_detections)
            
            for detection in face_detections:
                # Recognize face
                person_id, confidence = self.recognize_face(detection.face_encoding)
                
                if person_id:
                    # Known face
                    person = self.known_faces[person_id]
                    self.update_person_encounter(person_id)
                    
                    face_info = {
                        'person_id': person_id,
                        'name': person.name,
                        'confidence': confidence,
                        'trust_level': person.trust_level,
                        'bbox': detection.bbox,
                        'timestamp': detection.timestamp
                    }
                    results['known_faces'].append(face_info)
                    
                    # Check for suspicious behavior
                    if person.trust_level < 0.5:
                        results['suspicious_activity'] = True
                        results['alerts'].append(f"Low-trust person detected: {person.name}")
                        
                else:
                    # Unknown face
                    face_info = {
                        'confidence': detection.confidence,
                        'bbox': detection.bbox,
                        'timestamp': detection.timestamp
                    }
                    results['unknown_faces'].append(face_info)
                    
                    # Track unknown face
                    self.unknown_face_history.append(detection)
                    
                    # Check for multiple unknown faces
                    recent_unknown = [
                        f for f in self.unknown_face_history 
                        if (datetime.now() - f.timestamp).seconds < 300  # Last 5 minutes
                    ]
                    
                    if len(recent_unknown) >= 3:
                        results['suspicious_activity'] = True
                        results['alerts'].append("Multiple unknown faces detected in short time")
                        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            results['error'] = str(e)
            
        return results
        
    def get_proximity_analysis(self, current_faces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze proximity and potential threats
        
        Args:
            current_faces: List of currently detected faces
            
        Returns:
            Proximity analysis results
        """
        analysis = {
            'total_people': len(current_faces),
            'known_people': len([f for f in current_faces if 'person_id' in f]),
            'unknown_people': len([f for f in current_faces if 'person_id' not in f]),
            'threat_level': 'low',
            'recommendations': []
        }
        
        # Calculate threat level
        unknown_ratio = analysis['unknown_people'] / max(analysis['total_people'], 1)
        
        if unknown_ratio > 0.7:
            analysis['threat_level'] = 'high'
            analysis['recommendations'].append("High number of unknown people detected")
        elif unknown_ratio > 0.3:
            analysis['threat_level'] = 'medium'
            analysis['recommendations'].append("Moderate number of unknown people")
        else:
            analysis['threat_level'] = 'low'
            
        # Check for low-trust individuals
        low_trust_faces = [f for f in current_faces if f.get('trust_level', 1.0) < 0.5]
        if low_trust_faces:
            analysis['threat_level'] = 'high'
            analysis['recommendations'].append("Low-trust individuals detected")
            
        return analysis
        
    def export_database_summary(self) -> Dict[str, Any]:
        """Export a summary of the face database"""
        summary = {
            'total_people': len(self.known_faces),
            'trusted_people': len([p for p in self.known_faces.values() if p.trust_level >= 0.8]),
            'low_trust_people': len([p for p in self.known_faces.values() if p.trust_level < 0.5]),
            'recent_encounters': [],
            'database_size_mb': os.path.getsize(self.database_path) / (1024 * 1024) if os.path.exists(self.database_path) else 0
        }
        
        # Get recent encounters
        recent_people = sorted(
            self.known_faces.values(),
            key=lambda p: p.last_seen,
            reverse=True
        )[:10]
        
        for person in recent_people:
            summary['recent_encounters'].append({
                'name': person.name,
                'person_id': person.person_id,
                'trust_level': person.trust_level,
                'encounter_count': person.encounter_count,
                'last_seen': person.last_seen.isoformat()
            })
            
        return summary 