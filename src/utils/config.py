"""
Configuration Management for Guard AI
Centralized configuration for all system components
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class NLPConfig:
    """NLP module configuration"""
    model_name: str = "bert-base-uncased"
    max_sequence_length: int = 128
    confidence_threshold: float = 0.7
    voice_activity_threshold: float = 0.5
    min_audio_length: float = 0.5
    max_audio_length: float = 30.0


@dataclass
class BehavioralConfig:
    """Behavioral analysis configuration"""
    sequence_length: int = 24  # hours
    anomaly_threshold: float = 0.7
    min_pattern_confidence: float = 0.8
    data_retention_days: int = 30
    learning_rate: float = 0.001
    hidden_size: int = 128


@dataclass
class FacialRecognitionConfig:
    """Facial recognition configuration"""
    face_threshold: float = 0.6
    recognition_threshold: float = 0.7
    min_face_size: int = 20
    max_face_size: int = 300
    embedding_size: int = 128
    database_path: str = "face_database.pkl"


@dataclass
class EmergencyConfig:
    """Emergency response configuration"""
    threat_thresholds: Dict[str, float] = None
    response_protocols: Dict[str, list] = None
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.1
    state_size: int = 15
    
    def __post_init__(self):
        if self.threat_thresholds is None:
            self.threat_thresholds = {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'critical': 0.9
            }
        if self.response_protocols is None:
            self.response_protocols = {
                'low': ['monitor'],
                'medium': ['monitor', 'track_location'],
                'high': ['monitor', 'track_location', 'alert_contacts'],
                'critical': ['monitor', 'track_location', 'alert_contacts', 'call_emergency']
            }


@dataclass
class SensorFusionConfig:
    """Sensor fusion configuration"""
    buffer_size: int = 100
    fusion_interval: float = 1.0  # seconds
    hidden_dim: int = 256
    num_layers: int = 4
    threat_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.threat_thresholds is None:
            self.threat_thresholds = {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'critical': 0.9
            }


@dataclass
class SystemConfig:
    """Main system configuration"""
    user_id: str
    device: str = "auto"  # auto, cpu, cuda
    log_level: str = "INFO"
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    
    # Component configurations
    nlp: NLPConfig = None
    behavioral: BehavioralConfig = None
    facial_recognition: FacialRecognitionConfig = None
    emergency: EmergencyConfig = None
    sensor_fusion: SensorFusionConfig = None
    
    def __post_init__(self):
        if self.nlp is None:
            self.nlp = NLPConfig()
        if self.behavioral is None:
            self.behavioral = BehavioralConfig()
        if self.facial_recognition is None:
            self.facial_recognition = FacialRecognitionConfig()
        if self.emergency is None:
            self.emergency = EmergencyConfig()
        if self.sensor_fusion is None:
            self.sensor_fusion = SensorFusionConfig()


class ConfigManager:
    """
    Configuration manager for Guard AI
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        self.load_config()
        
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                # Create system config from data
                self.config = self._create_config_from_dict(config_data)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                # Create default configuration
                self.config = SystemConfig(user_id="default_user")
                self.save_config()
                logger.info(f"Created default configuration at {self.config_path}")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Fallback to default config
            self.config = SystemConfig(user_id="default_user")
            
    def save_config(self):
        """Save configuration to file"""
        try:
            if self.config is None:
                logger.warning("No configuration to save")
                return
                
            # Convert config to dictionary
            config_dict = asdict(self.config)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save to file
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
            logger.info(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> SystemConfig:
        """Create SystemConfig from dictionary"""
        # Extract main config
        main_config = {
            'user_id': config_data.get('user_id', 'default_user'),
            'device': config_data.get('device', 'auto'),
            'log_level': config_data.get('log_level', 'INFO'),
            'data_dir': config_data.get('data_dir', 'data'),
            'models_dir': config_data.get('models_dir', 'models'),
            'logs_dir': config_data.get('logs_dir', 'logs')
        }
        
        # Create component configs
        nlp_data = config_data.get('nlp', {})
        nlp_config = NLPConfig(**nlp_data)
        
        behavioral_data = config_data.get('behavioral', {})
        behavioral_config = BehavioralConfig(**behavioral_data)
        
        facial_data = config_data.get('facial_recognition', {})
        facial_config = FacialRecognitionConfig(**facial_data)
        
        emergency_data = config_data.get('emergency', {})
        emergency_config = EmergencyConfig(**emergency_data)
        
        sensor_data = config_data.get('sensor_fusion', {})
        sensor_config = SensorFusionConfig(**sensor_data)
        
        return SystemConfig(
            **main_config,
            nlp=nlp_config,
            behavioral=behavioral_config,
            facial_recognition=facial_config,
            emergency=emergency_config,
            sensor_fusion=sensor_config
        )
        
    def get_config(self) -> SystemConfig:
        """Get current configuration"""
        return self.config
        
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        try:
            if self.config is None:
                logger.warning("No configuration to update")
                return
                
            # Update main config
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
            # Update component configs
            for component_name, component_updates in updates.items():
                if hasattr(self.config, component_name):
                    component = getattr(self.config, component_name)
                    if component is not None:
                        for key, value in component_updates.items():
                            if hasattr(component, key):
                                setattr(component, key, value)
                                
            # Save updated config
            self.save_config()
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            
    def get_device(self) -> str:
        """Get device configuration"""
        if self.config is None:
            return "cpu"
            
        device = self.config.device
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return device
            
    def create_directories(self):
        """Create necessary directories"""
        if self.config is None:
            return
            
        directories = [
            self.config.data_dir,
            self.config.models_dir,
            self.config.logs_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info("Created necessary directories")
        
    def validate_config(self) -> bool:
        """Validate configuration"""
        if self.config is None:
            logger.error("No configuration loaded")
            return False
            
        try:
            # Validate main config
            if not self.config.user_id:
                logger.error("User ID is required")
                return False
                
            # Validate component configs
            if self.config.nlp.confidence_threshold < 0 or self.config.nlp.confidence_threshold > 1:
                logger.error("NLP confidence threshold must be between 0 and 1")
                return False
                
            if self.config.behavioral.anomaly_threshold < 0 or self.config.behavioral.anomaly_threshold > 1:
                logger.error("Behavioral anomaly threshold must be between 0 and 1")
                return False
                
            if self.config.facial_recognition.recognition_threshold < 0 or self.config.facial_recognition.recognition_threshold > 1:
                logger.error("Facial recognition threshold must be between 0 and 1")
                return False
                
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
            
    def export_config(self, filepath: str) -> bool:
        """Export configuration to file"""
        try:
            if self.config is None:
                logger.warning("No configuration to export")
                return False
                
            config_dict = asdict(self.config)
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            logger.info(f"Exported configuration to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
            
    def import_config(self, filepath: str) -> bool:
        """Import configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
                
            self.config = self._create_config_from_dict(config_data)
            self.save_config()
            
            logger.info(f"Imported configuration from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False 