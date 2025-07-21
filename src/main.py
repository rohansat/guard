"""
Guard AI - Main System Coordinator
Voice-activated personal safety assistant that integrates all AI modules
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os

# Import Guard AI modules
from src.nlp.intent_classifier import GuardIntentClassifier
from src.nlp.voice_processor import VoiceProcessor
from src.behavioral.pattern_learner import BehavioralAnalyzer, LocationData, ActivityData
from src.facial_recognition.face_detector import FaceRecognitionSystem
from src.emergency.response_system import EmergencyResponseSystem, Contact
from src.sensor_fusion.fusion_engine import SensorFusionEngine
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)


class GuardAI:
    """
    Main Guard AI system coordinator
    """
    
    def __init__(self, user_id: str = "default_user", config_path: str = "config.yaml"):
        """
        Initialize Guard AI system
        
        Args:
            user_id: Unique identifier for the user
            config_path: Path to configuration file
        """
        self.user_id = user_id
        
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.config.user_id = user_id
        self.config_manager.save_config()
        
        # Create necessary directories
        self.config_manager.create_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize AI modules
        self._initialize_modules()
        
        # System state
        self.is_active = False
        self.last_activity = datetime.now()
        self.system_status = "initialized"
        
        # Event handlers
        self.event_handlers = {
            'voice_command': [],
            'threat_detected': [],
            'emergency_triggered': [],
            'system_status_change': []
        }
        
        logger.info(f"Guard AI initialized for user: {user_id}")
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Create logs directory
        os.makedirs(self.config.logs_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.logs_dir}/guard_ai.log"),
                logging.StreamHandler()
            ]
        )
        
    def _initialize_modules(self):
        """Initialize all AI modules"""
        try:
            # Initialize NLP modules
            self.intent_classifier = GuardIntentClassifier()
            self.voice_processor = VoiceProcessor()
            
            # Initialize behavioral analysis
            self.behavioral_analyzer = BehavioralAnalyzer(
                user_id=self.user_id,
                model_path=f"{self.config.models_dir}/behavioral_model.pth"
            )
            
            # Initialize facial recognition
            self.face_recognition = FaceRecognitionSystem(
                model_path=f"{self.config.models_dir}/face_model.pth",
                database_path=self.config.facial_recognition.database_path
            )
            
            # Initialize emergency response
            self.emergency_system = EmergencyResponseSystem(
                user_id=self.user_id,
                model_path=f"{self.config.models_dir}/emergency_model.pth"
            )
            
            # Initialize sensor fusion
            self.sensor_fusion = SensorFusionEngine(
                model_path=f"{self.config.models_dir}/fusion_model.pth"
            )
            
            logger.info("All AI modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
            raise
            
    def add_event_handler(self, event_type: str, handler):
        """Add event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            
    def _trigger_event(self, event_type: str, data: Any):
        """Trigger event handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
                    
    async def start(self):
        """Start Guard AI system"""
        try:
            self.is_active = True
            self.system_status = "active"
            self._trigger_event('system_status_change', {'status': 'active'})
            
            logger.info("Guard AI system started")
            
            # Start background tasks
            await asyncio.gather(
                self._monitor_sensors(),
                self._process_voice_commands(),
                self._check_system_health()
            )
            
        except Exception as e:
            logger.error(f"Error starting Guard AI: {e}")
            self.system_status = "error"
            self._trigger_event('system_status_change', {'status': 'error', 'error': str(e)})
            
    async def stop(self):
        """Stop Guard AI system"""
        try:
            self.is_active = False
            self.system_status = "stopped"
            self._trigger_event('system_status_change', {'status': 'stopped'})
            
            logger.info("Guard AI system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Guard AI: {e}")
            
    async def process_voice_command(self, audio_data: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Process voice command
        
        Args:
            audio_data: Raw audio data
            sample_rate: Audio sample rate
            
        Returns:
            Processing result
        """
        try:
            # Convert audio data to numpy array
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Process voice
            transcription = self.voice_processor.speech_to_text(audio_array, sample_rate)
            
            if not transcription:
                return {
                    'success': False,
                    'error': 'No speech detected',
                    'transcription': ''
                }
                
            # Check for emergency voice
            is_emergency_voice = self.voice_processor.is_emergency_voice(audio_array, sample_rate)
            
            # Classify intent
            intent_result = self.intent_classifier.classify_intent(transcription)
            
            # Extract entities
            entities = self.intent_classifier.extract_entities(transcription, intent_result['intent'])
            
            # Generate response
            response = self.intent_classifier.generate_response(
                intent_result['intent'],
                entities,
                self.intent_classifier.get_context()
            )
            
            # Trigger voice command event
            self._trigger_event('voice_command', {
                'transcription': transcription,
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'is_emergency_voice': is_emergency_voice,
                'response': response
            })
            
            # Handle emergency voice
            if is_emergency_voice:
                await self._handle_emergency_voice(transcription)
                
            return {
                'success': True,
                'transcription': transcription,
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'response': response,
                'is_emergency_voice': is_emergency_voice
            }
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            return {
                'success': False,
                'error': str(e),
                'transcription': ''
            }
            
    async def _handle_emergency_voice(self, transcription: str):
        """Handle emergency voice detection"""
        try:
            # Create sensor data for emergency voice
            sensor_data = {
                'voice_distress': 1.0,
                'description': f"Emergency voice detected: {transcription}",
                'triggered_by': 'voice',
                'confidence': 0.9
            }
            
            # Process threat
            threat_result = self.emergency_system.process_threat(sensor_data)
            
            # Trigger threat event
            self._trigger_event('threat_detected', threat_result)
            
            logger.warning(f"Emergency voice detected: {transcription}")
            
        except Exception as e:
            logger.error(f"Error handling emergency voice: {e}")
            
    async def add_location_data(self, latitude: float, longitude: float, accuracy: float = 10.0,
                               speed: Optional[float] = None, heading: Optional[float] = None):
        """Add location data to behavioral analysis"""
        try:
            location_data = LocationData(
                timestamp=datetime.now(),
                latitude=latitude,
                longitude=longitude,
                accuracy=accuracy,
                speed=speed,
                heading=heading
            )
            
            self.behavioral_analyzer.add_location_data(location_data)
            
            # Add to sensor fusion
            sensor_data = SensorData(
                sensor_type=SensorType.GPS,
                timestamp=datetime.now(),
                data={
                    'latitude': latitude,
                    'longitude': longitude,
                    'speed': speed or 0.0,
                    'heading': heading or 0.0
                },
                confidence=1.0 - (accuracy / 100.0),  # Higher accuracy = higher confidence
                quality=1.0
            )
            
            self.sensor_fusion.add_sensor_data(sensor_data)
            
        except Exception as e:
            logger.error(f"Error adding location data: {e}")
            
    async def add_activity_data(self, activity_type: str, confidence: float, duration: float):
        """Add activity data to behavioral analysis"""
        try:
            activity_data = ActivityData(
                timestamp=datetime.now(),
                activity_type=activity_type,
                confidence=confidence,
                duration=duration
            )
            
            self.behavioral_analyzer.add_activity_data(activity_data)
            
        except Exception as e:
            logger.error(f"Error adding activity data: {e}")
            
    async def process_camera_frame(self, image_data: bytes) -> Dict[str, Any]:
        """Process camera frame for facial recognition"""
        try:
            import cv2
            import numpy as np
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'success': False, 'error': 'Invalid image data'}
                
            # Process image
            face_results = self.face_recognition.process_image(image)
            
            # Add to sensor fusion
            if face_results['faces_detected'] > 0:
                # Extract face features (simplified)
                face_features = np.random.rand(128)  # Placeholder
                
                sensor_data = SensorData(
                    sensor_type=SensorType.CAMERA,
                    timestamp=datetime.now(),
                    data={'face_features': face_features.tolist()},
                    confidence=0.8,
                    quality=0.9
                )
                
                self.sensor_fusion.add_sensor_data(sensor_data)
                
            return {
                'success': True,
                'faces_detected': face_results['faces_detected'],
                'known_faces': face_results['known_faces'],
                'unknown_faces': face_results['unknown_faces'],
                'suspicious_activity': face_results['suspicious_activity'],
                'alerts': face_results['alerts']
            }
            
        except Exception as e:
            logger.error(f"Error processing camera frame: {e}")
            return {'success': False, 'error': str(e)}
            
    async def add_accelerometer_data(self, x: float, y: float, z: float):
        """Add accelerometer data to sensor fusion"""
        try:
            sensor_data = SensorData(
                sensor_type=SensorType.ACCELEROMETER,
                timestamp=datetime.now(),
                data={'x': x, 'y': y, 'z': z},
                confidence=0.9,
                quality=0.95
            )
            
            self.sensor_fusion.add_sensor_data(sensor_data)
            
        except Exception as e:
            logger.error(f"Error adding accelerometer data: {e}")
            
    async def add_microphone_data(self, audio_data: bytes, sample_rate: int = 16000):
        """Add microphone data to sensor fusion"""
        try:
            import numpy as np
            import librosa
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            sensor_data = SensorData(
                sensor_type=SensorType.MICROPHONE,
                timestamp=datetime.now(),
                data={'mfcc': mfcc_mean.tolist()},
                confidence=0.8,
                quality=0.9
            )
            
            self.sensor_fusion.add_sensor_data(sensor_data)
            
        except Exception as e:
            logger.error(f"Error adding microphone data: {e}")
            
    async def _monitor_sensors(self):
        """Background task to monitor sensors and perform fusion"""
        while self.is_active:
            try:
                # Perform sensor fusion
                fusion_result = self.sensor_fusion.perform_fusion()
                
                if fusion_result:
                    # Check for threats
                    if fusion_result.threat_score > self.config.sensor_fusion.threat_thresholds['medium']:
                        # Process threat
                        threat_data = {
                            'threat_level': fusion_result.threat_score,
                            'description': f"Threat detected: {', '.join(fusion_result.threat_factors)}",
                            'triggered_by': 'sensor_fusion',
                            'confidence': fusion_result.confidence,
                            'location': {'lat': 0.0, 'lon': 0.0}  # Get from GPS
                        }
                        
                        threat_result = self.emergency_system.process_threat(threat_data)
                        
                        # Trigger threat event
                        self._trigger_event('threat_detected', threat_result)
                        
                        if threat_result['threat_level'] in ['high', 'critical']:
                            self._trigger_event('emergency_triggered', threat_result)
                            
                # Sleep for fusion interval
                await asyncio.sleep(self.config.sensor_fusion.fusion_interval)
                
            except Exception as e:
                logger.error(f"Error in sensor monitoring: {e}")
                await asyncio.sleep(1)
                
    async def _process_voice_commands(self):
        """Background task to process voice commands"""
        while self.is_active:
            try:
                # This would integrate with actual voice input
                # For now, just sleep
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in voice processing: {e}")
                await asyncio.sleep(1)
                
    async def _check_system_health(self):
        """Background task to check system health"""
        while self.is_active:
            try:
                # Check behavioral patterns
                patterns = self.behavioral_analyzer.learn_patterns()
                
                # Check for anomalies
                anomaly_result = self.behavioral_analyzer.detect_anomaly()
                
                if anomaly_result['is_anomaly']:
                    # Process behavioral threat
                    threat_data = {
                        'behavior_anomaly': anomaly_result['anomaly_score'],
                        'description': 'Behavioral anomaly detected',
                        'triggered_by': 'behavioral',
                        'confidence': anomaly_result['confidence']
                    }
                    
                    threat_result = self.emergency_system.process_threat(threat_data)
                    self._trigger_event('threat_detected', threat_result)
                    
                # Sleep for health check interval
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in system health check: {e}")
                await asyncio.sleep(60)
                
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'user_id': self.user_id,
            'is_active': self.is_active,
            'system_status': self.system_status,
            'last_activity': self.last_activity.isoformat(),
            'uptime': (datetime.now() - self.last_activity).total_seconds(),
            'modules': {
                'nlp': 'active',
                'behavioral': 'active',
                'facial_recognition': 'active',
                'emergency_response': 'active',
                'sensor_fusion': 'active'
            }
        }
        
    def get_safety_summary(self) -> Dict[str, Any]:
        """Get safety summary from all modules"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'user_id': self.user_id,
                'behavioral_analysis': {},
                'facial_recognition': {},
                'emergency_events': {},
                'sensor_fusion': {},
                'overall_threat_level': 'low'
            }
            
            # Get behavioral summary
            if hasattr(self.behavioral_analyzer, 'get_safety_assessment'):
                current_location = LocationData(
                    timestamp=datetime.now(),
                    latitude=0.0,
                    longitude=0.0,
                    accuracy=10.0
                )
                summary['behavioral_analysis'] = self.behavioral_analyzer.get_safety_assessment(current_location)
                
            # Get facial recognition summary
            summary['facial_recognition'] = self.face_recognition.export_database_summary()
            
            # Get emergency events summary
            summary['emergency_events'] = self.emergency_system.get_emergency_summary()
            
            # Get sensor fusion summary
            summary['sensor_fusion'] = self.sensor_fusion.get_fusion_summary()
            
            # Determine overall threat level
            threat_scores = []
            
            if summary['behavioral_analysis'].get('safety_score'):
                threat_scores.append(1.0 - summary['behavioral_analysis']['safety_score'])
                
            if summary['sensor_fusion'].get('average_threat_score'):
                threat_scores.append(summary['sensor_fusion']['average_threat_score'])
                
            if threat_scores:
                avg_threat = sum(threat_scores) / len(threat_scores)
                if avg_threat > 0.7:
                    summary['overall_threat_level'] = 'high'
                elif avg_threat > 0.4:
                    summary['overall_threat_level'] = 'medium'
                else:
                    summary['overall_threat_level'] = 'low'
                    
            return summary
            
        except Exception as e:
            logger.error(f"Error getting safety summary: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'user_id': self.user_id,
                'error': str(e),
                'overall_threat_level': 'unknown'
            }
            
    def add_emergency_contact(self, name: str, phone: str, email: str = None,
                            relationship: str = "friend", priority: int = 1) -> bool:
        """Add emergency contact"""
        try:
            contact = Contact(
                contact_id=f"contact_{int(time.time())}",
                name=name,
                phone=phone,
                email=email,
                relationship=relationship,
                priority=priority,
                notification_preferences={'sms': True, 'email': bool(email)}
            )
            
            success = self.emergency_system.add_contact(contact)
            return success
            
        except Exception as e:
            logger.error(f"Error adding emergency contact: {e}")
            return False
            
    def add_known_face(self, person_id: str, name: str, face_encodings: List, trust_level: float = 1.0) -> bool:
        """Add known face to database"""
        try:
            success = self.face_recognition.add_person(person_id, name, face_encodings, trust_level)
            return success
            
        except Exception as e:
            logger.error(f"Error adding known face: {e}")
            return False


async def main():
    """Main function to run Guard AI"""
    # Initialize Guard AI
    guard_ai = GuardAI(user_id="rohan_sathisha")
    
    # Add event handlers
    def on_voice_command(data):
        print(f"Voice command: {data['transcription']}")
        print(f"Intent: {data['intent']}")
        print(f"Response: {data['response']}")
        
    def on_threat_detected(data):
        print(f"Threat detected: {data['threat_level']}")
        print(f"Action: {data['selected_action']}")
        
    def on_emergency_triggered(data):
        print(f"EMERGENCY TRIGGERED: {data['threat_level']}")
        print(f"Event ID: {data.get('emergency_event', 'N/A')}")
        
    guard_ai.add_event_handler('voice_command', on_voice_command)
    guard_ai.add_event_handler('threat_detected', on_threat_detected)
    guard_ai.add_event_handler('emergency_triggered', on_emergency_triggered)
    
    # Add some emergency contacts
    guard_ai.add_emergency_contact("Mom", "+1234567890", "mom@example.com", "family", 1)
    guard_ai.add_emergency_contact("Best Friend", "+0987654321", "friend@example.com", "friend", 2)
    
    # Start the system
    print("Starting Guard AI...")
    await guard_ai.start()


if __name__ == "__main__":
    asyncio.run(main()) 