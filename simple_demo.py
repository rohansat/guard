#!/usr/bin/env python3
"""
Simple Guard AI Demo
A simplified demonstration of Guard AI capabilities without loading pre-trained models
"""

import asyncio
import json
import time
from datetime import datetime
import numpy as np

# Add the current directory to Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp.intent_classifier import GuardIntentClassifier
from src.nlp.voice_processor import VoiceProcessor
from src.behavioral.pattern_learner import BehavioralAnalyzer
from src.facial_recognition.face_detector import FaceRecognitionSystem
from src.emergency.response_system import EmergencyResponseSystem, Contact
from src.sensor_fusion.fusion_engine import SensorFusionEngine
from src.utils.config import ConfigManager


class SimpleGuardAI:
    """
    Simplified Guard AI system for demonstration
    """
    
    def __init__(self, user_id: str = "demo_user"):
        """Initialize simplified Guard AI"""
        self.user_id = user_id
        print(f"🤖 Initializing Guard AI for user: {user_id}")
        
        # Initialize configuration
        self.config_manager = ConfigManager("config.yaml")
        self.config = self.config_manager.get_config()
        
        # Initialize modules without loading pre-trained models
        self._initialize_modules()
        
        print("✅ Guard AI initialized successfully!")
        
    def _initialize_modules(self):
        """Initialize AI modules without loading models"""
        try:
            # Initialize NLP modules
            self.intent_classifier = GuardIntentClassifier()
            self.voice_processor = VoiceProcessor()
            
            # Initialize behavioral analysis (without model loading)
            self.behavioral_analyzer = BehavioralAnalyzer(user_id=self.user_id)
            
            # Initialize facial recognition (without model loading)
            self.face_recognition = FaceRecognitionSystem()
            
            # Initialize emergency response (without model loading)
            self.emergency_system = EmergencyResponseSystem(user_id=self.user_id)
            
            # Initialize sensor fusion (without model loading)
            self.sensor_fusion = SensorFusionEngine()
            
            print("✅ All AI modules initialized")
            
        except Exception as e:
            print(f"❌ Error initializing modules: {e}")
            raise
    
    def test_voice_processing(self):
        """Test voice processing capabilities"""
        print("\n🎤 Testing Voice Processing...")
        
        # Test intent classification
        test_commands = [
            "Guard AI, I need help",
            "What's my current location?",
            "Add John as an emergency contact",
            "Is it safe to walk here?",
            "Activate emergency mode"
        ]
        
        for command in test_commands:
            print(f"\n📝 Processing: '{command}'")
            try:
                intent_result = self.intent_classifier.classify_intent(command)
                print(f"   Intent: {intent_result['intent']}")
                print(f"   Confidence: {intent_result['confidence']:.2f}")
                
                # Generate response
                response = self.intent_classifier.generate_response(
                    intent_result['intent'],
                    intent_result.get('entities', {}),
                    self.intent_classifier.get_context()
                )
                print(f"   Response: {response}")
                
            except Exception as e:
                print(f"   Error: {e}")
    
    def test_behavioral_analysis(self):
        """Test behavioral analysis capabilities"""
        print("\n🧠 Testing Behavioral Analysis...")
        
        # Simulate location data
        locations = [
            {"lat": 37.7749, "lon": -122.4194, "timestamp": datetime.now()},
            {"lat": 37.7849, "lon": -122.4094, "timestamp": datetime.now()},
            {"lat": 37.7949, "lon": -122.3994, "timestamp": datetime.now()},
        ]
        
        # Simulate activity data
        activities = [
            {"activity": "walking", "confidence": 0.9, "duration": 300},
            {"activity": "running", "confidence": 0.8, "duration": 180},
            {"activity": "standing", "confidence": 0.7, "duration": 60},
        ]
        
        print("📍 Processing location data...")
        for loc in locations:
            try:
                result = self.behavioral_analyzer.add_location_data(
                    lat=loc["lat"], 
                    lon=loc["lon"], 
                    timestamp=loc["timestamp"]
                )
                print(f"   Location added: {loc['lat']:.4f}, {loc['lon']:.4f}")
            except Exception as e:
                print(f"   Error: {e}")
        
        print("🏃 Processing activity data...")
        for activity in activities:
            try:
                result = self.behavioral_analyzer.add_activity_data(
                    activity_type=activity["activity"],
                    confidence=activity["confidence"],
                    duration=activity["duration"]
                )
                print(f"   Activity: {activity['activity']} ({activity['confidence']:.1f})")
            except Exception as e:
                print(f"   Error: {e}")
        
        # Get behavioral summary
        try:
            summary = self.behavioral_analyzer.get_behavioral_summary()
            print(f"📊 Behavioral Summary: {len(summary.get('locations', []))} locations, {len(summary.get('activities', []))} activities")
        except Exception as e:
            print(f"   Error getting summary: {e}")
    
    def test_facial_recognition(self):
        """Test facial recognition capabilities"""
        print("\n👤 Testing Facial Recognition...")
        
        # Create a dummy image (black image)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            # Process the dummy image
            result = self.face_recognition.process_image(dummy_image)
            print(f"   Faces detected: {result.get('num_faces', 0)}")
            print(f"   Known faces: {result.get('known_faces', [])}")
            print(f"   Unknown faces: {result.get('unknown_faces', [])}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    def test_emergency_system(self):
        """Test emergency response system"""
        print("\n🚨 Testing Emergency Response System...")
        
        # Add a test contact
        try:
            test_contact = Contact(
                contact_id="test_001",
                name="Test Contact",
                phone="+1234567890",
                relationship="friend",
                priority=1,
                email="test@example.com",
                notification_preferences={"sms": True, "call": True, "email": False}
            )
            
            success = self.emergency_system.add_contact(test_contact)
            print(f"   Contact added: {success}")
            
            # Get emergency summary
            summary = self.emergency_system.get_emergency_summary()
            print(f"   Emergency contacts: {len(summary.get('contacts', []))}")
            print(f"   Active events: {len(summary.get('active_events', []))}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    def test_sensor_fusion(self):
        """Test sensor fusion capabilities"""
        print("\n🔍 Testing Sensor Fusion...")
        
        # Simulate sensor data
        sensor_data = {
            "location": {"lat": 37.7749, "lon": -122.4194, "accuracy": 10.0},
            "accelerometer": {"x": 0.1, "y": 0.2, "z": 9.8},
            "microphone": {"volume": 0.3, "frequency": 1000},
            "camera": {"faces_detected": 2, "motion_detected": True}
        }
        
        try:
            # Process sensor data
            result = self.sensor_fusion.process_sensor_data(sensor_data)
            print(f"   Threat level: {result.get('threat_level', 'unknown')}")
            print(f"   Risk factors: {result.get('risk_factors', [])}")
            print(f"   Recommendations: {result.get('recommendations', [])}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    def run_demo(self):
        """Run the complete demo"""
        print("\n" + "="*60)
        print("🚀 GUARD AI DEMO - Voice-Activated Personal Safety Assistant")
        print("="*60)
        
        # Test each component
        self.test_voice_processing()
        self.test_behavioral_analysis()
        self.test_facial_recognition()
        self.test_emergency_system()
        self.test_sensor_fusion()
        
        print("\n" + "="*60)
        print("✅ Demo completed successfully!")
        print("="*60)
        
        # Show system capabilities
        print("\n🎯 Guard AI Capabilities:")
        print("   • Voice command processing and intent recognition")
        print("   • Behavioral pattern learning and anomaly detection")
        print("   • Facial recognition and person identification")
        print("   • Emergency response and contact management")
        print("   • Multi-sensor data fusion and threat assessment")
        print("   • Real-time safety monitoring and alerts")
        
        print("\n🔧 Next Steps:")
        print("   • Train models with real data for better accuracy")
        print("   • Integrate with actual sensors and cameras")
        print("   • Deploy to mobile device or edge computing platform")
        print("   • Configure emergency contacts and notification preferences")
        print("   • Customize safety thresholds and response protocols")


async def main():
    """Main demo function"""
    try:
        # Initialize Guard AI
        guard_ai = SimpleGuardAI(user_id="demo_user")
        
        # Run the demo
        guard_ai.run_demo()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 