"""
Multimodal Sensor Fusion Engine for Guard AI
Combines data from multiple sensors for comprehensive threat assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
import asyncio
from enum import Enum
import os

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Sensor type enumeration"""
    GPS = "gps"
    ACCELEROMETER = "accelerometer"
    MICROPHONE = "microphone"
    CAMERA = "camera"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"


@dataclass
class SensorData:
    """Generic sensor data structure"""
    sensor_type: SensorType
    timestamp: datetime
    data: Dict[str, Any]
    confidence: float
    quality: float  # 0-1, data quality indicator


@dataclass
class FusionResult:
    """Result of sensor fusion"""
    timestamp: datetime
    threat_score: float
    confidence: float
    contributing_sensors: List[SensorType]
    threat_factors: List[str]
    recommendations: List[str]
    fused_features: np.ndarray


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for sensor fusion
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return output, attention_weights


class SensorFusionTransformer(nn.Module):
    """
    Transformer-based model for multimodal sensor fusion
    """
    
    def __init__(self, sensor_dims: Dict[SensorType, int], hidden_dim: int = 256, num_layers: int = 4):
        super(SensorFusionTransformer, self).__init__()
        
        self.sensor_dims = sensor_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Sensor-specific encoders
        self.sensor_encoders = nn.ModuleDict()
        for sensor_type, dim in sensor_dims.items():
            self.sensor_encoders[sensor_type.value] = nn.Sequential(
                nn.Linear(dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
            
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 10, hidden_dim))  # Max 10 sensors
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output heads
        self.threat_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.confidence_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sensor_data: Dict[SensorType, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the fusion model
        
        Args:
            sensor_data: Dictionary of sensor data tensors
            
        Returns:
            Tuple of (threat_score, confidence)
        """
        # Encode each sensor's data
        encoded_sensors = []
        sensor_mask = []
        
        for sensor_type in SensorType:
            if sensor_type in sensor_data:
                encoded = self.sensor_encoders[sensor_type.value](sensor_data[sensor_type])
                encoded_sensors.append(encoded)
                sensor_mask.append(True)
            else:
                # Pad with zeros for missing sensors
                encoded_sensors.append(torch.zeros(1, self.hidden_dim))
                sensor_mask.append(False)
                
        # Stack encoded sensors
        sensor_sequence = torch.stack(encoded_sensors, dim=1)  # [batch, num_sensors, hidden_dim]
        
        # Add positional encoding
        sensor_sequence = sensor_sequence + self.pos_encoding[:, :sensor_sequence.size(1), :]
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            sensor_sequence = transformer_layer(sensor_sequence)
            
        # Global average pooling
        pooled = torch.mean(sensor_sequence, dim=1)
        
        # Get outputs
        threat_score = self.threat_classifier(pooled)
        confidence = self.confidence_regressor(pooled)
        
        return threat_score, confidence


class SensorFusionEngine:
    """
    Main sensor fusion engine for Guard AI
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the sensor fusion engine
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define sensor dimensions
        self.sensor_dims = {
            SensorType.GPS: 4,  # lat, lon, speed, heading
            SensorType.ACCELEROMETER: 3,  # x, y, z
            SensorType.MICROPHONE: 13,  # MFCC features
            SensorType.CAMERA: 128,  # Face recognition features
            SensorType.GYROSCOPE: 3,  # x, y, z
            SensorType.MAGNETOMETER: 3  # x, y, z
        }
        
        # Initialize fusion model
        self.fusion_model = SensorFusionTransformer(self.sensor_dims)
        
        if model_path and os.path.exists(model_path):
            self.fusion_model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.fusion_model.to(self.device)
        self.fusion_model.eval()
        
        # Sensor data buffers
        self.sensor_buffers: Dict[SensorType, List[SensorData]] = {
            sensor_type: [] for sensor_type in SensorType
        }
        
        # Fusion parameters
        self.buffer_size = 100  # Number of data points to keep per sensor
        self.fusion_interval = 1.0  # seconds
        self.last_fusion_time = datetime.now()
        
        # Threat assessment parameters
        self.threat_thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
        
        # Historical fusion results
        self.fusion_history: List[FusionResult] = []
        
    def add_sensor_data(self, sensor_data: SensorData):
        """
        Add new sensor data to the buffer
        
        Args:
            sensor_data: Sensor data to add
        """
        try:
            # Add to buffer
            self.sensor_buffers[sensor_data.sensor_type].append(sensor_data)
            
            # Maintain buffer size
            if len(self.sensor_buffers[sensor_data.sensor_type]) > self.buffer_size:
                self.sensor_buffers[sensor_data.sensor_type].pop(0)
                
        except Exception as e:
            logger.error(f"Error adding sensor data: {e}")
            
    def preprocess_gps_data(self, gps_data: List[SensorData]) -> torch.Tensor:
        """Preprocess GPS sensor data"""
        if not gps_data:
            return torch.zeros(1, self.sensor_dims[SensorType.GPS])
            
        # Use most recent data point
        latest_data = gps_data[-1].data
        
        features = [
            latest_data.get('latitude', 0.0),
            latest_data.get('longitude', 0.0),
            latest_data.get('speed', 0.0),
            latest_data.get('heading', 0.0)
        ]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
    def preprocess_accelerometer_data(self, accel_data: List[SensorData]) -> torch.Tensor:
        """Preprocess accelerometer data"""
        if not accel_data:
            return torch.zeros(1, self.sensor_dims[SensorType.ACCELEROMETER])
            
        # Use most recent data point
        latest_data = accel_data[-1].data
        
        features = [
            latest_data.get('x', 0.0),
            latest_data.get('y', 0.0),
            latest_data.get('z', 0.0)
        ]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
    def preprocess_microphone_data(self, mic_data: List[SensorData]) -> torch.Tensor:
        """Preprocess microphone data"""
        if not mic_data:
            return torch.zeros(1, self.sensor_dims[SensorType.MICROPHONE])
            
        # Use most recent data point
        latest_data = mic_data[-1].data
        
        # Extract MFCC features or other audio features
        mfcc_features = latest_data.get('mfcc', [0.0] * 13)
        if len(mfcc_features) < 13:
            mfcc_features.extend([0.0] * (13 - len(mfcc_features)))
            
        return torch.tensor(mfcc_features[:13], dtype=torch.float32).unsqueeze(0)
        
    def preprocess_camera_data(self, camera_data: List[SensorData]) -> torch.Tensor:
        """Preprocess camera data"""
        if not camera_data:
            return torch.zeros(1, self.sensor_dims[SensorType.CAMERA])
            
        # Use most recent data point
        latest_data = camera_data[-1].data
        
        # Extract face recognition features
        face_features = latest_data.get('face_features', [0.0] * 128)
        if len(face_features) < 128:
            face_features.extend([0.0] * (128 - len(face_features)))
            
        return torch.tensor(face_features[:128], dtype=torch.float32).unsqueeze(0)
        
    def preprocess_gyroscope_data(self, gyro_data: List[SensorData]) -> torch.Tensor:
        """Preprocess gyroscope data"""
        if not gyro_data:
            return torch.zeros(1, self.sensor_dims[SensorType.GYROSCOPE])
            
        # Use most recent data point
        latest_data = gyro_data[-1].data
        
        features = [
            latest_data.get('x', 0.0),
            latest_data.get('y', 0.0),
            latest_data.get('z', 0.0)
        ]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
    def preprocess_magnetometer_data(self, mag_data: List[SensorData]) -> torch.Tensor:
        """Preprocess magnetometer data"""
        if not mag_data:
            return torch.zeros(1, self.sensor_dims[SensorType.MAGNETOMETER])
            
        # Use most recent data point
        latest_data = mag_data[-1].data
        
        features = [
            latest_data.get('x', 0.0),
            latest_data.get('y', 0.0),
            latest_data.get('z', 0.0)
        ]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
    def should_perform_fusion(self) -> bool:
        """Check if it's time to perform sensor fusion"""
        current_time = datetime.now()
        time_since_last = (current_time - self.last_fusion_time).total_seconds()
        return time_since_last >= self.fusion_interval
        
    def perform_fusion(self) -> Optional[FusionResult]:
        """
        Perform multimodal sensor fusion
        
        Returns:
            Fusion result if successful, None otherwise
        """
        try:
            if not self.should_perform_fusion():
                return None
                
            # Preprocess all sensor data
            sensor_tensors = {}
            
            sensor_tensors[SensorType.GPS] = self.preprocess_gps_data(
                self.sensor_buffers[SensorType.GPS]
            )
            sensor_tensors[SensorType.ACCELEROMETER] = self.preprocess_accelerometer_data(
                self.sensor_buffers[SensorType.ACCELEROMETER]
            )
            sensor_tensors[SensorType.MICROPHONE] = self.preprocess_microphone_data(
                self.sensor_buffers[SensorType.MICROPHONE]
            )
            sensor_tensors[SensorType.CAMERA] = self.preprocess_camera_data(
                self.sensor_buffers[SensorType.CAMERA]
            )
            sensor_tensors[SensorType.GYROSCOPE] = self.preprocess_gyroscope_data(
                self.sensor_buffers[SensorType.GYROSCOPE]
            )
            sensor_tensors[SensorType.MAGNETOMETER] = self.preprocess_magnetometer_data(
                self.sensor_buffers[SensorType.MAGNETOMETER]
            )
            
            # Move to device
            sensor_tensors = {k: v.to(self.device) for k, v in sensor_tensors.items()}
            
            # Perform fusion
            with torch.no_grad():
                threat_score, confidence = self.fusion_model(sensor_tensors)
                
            threat_score = threat_score.item()
            confidence = confidence.item()
            
            # Determine contributing sensors
            contributing_sensors = [
                sensor_type for sensor_type in SensorType
                if self.sensor_buffers[sensor_type]
            ]
            
            # Analyze threat factors
            threat_factors = self._analyze_threat_factors(sensor_tensors)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(threat_score, threat_factors)
            
            # Create fusion result
            result = FusionResult(
                timestamp=datetime.now(),
                threat_score=threat_score,
                confidence=confidence,
                contributing_sensors=contributing_sensors,
                threat_factors=threat_factors,
                recommendations=recommendations,
                fused_features=torch.cat(list(sensor_tensors.values()), dim=1).cpu().numpy()
            )
            
            # Update history
            self.fusion_history.append(result)
            self.last_fusion_time = datetime.now()
            
            # Keep only recent history
            if len(self.fusion_history) > 1000:
                self.fusion_history = self.fusion_history[-1000:]
                
            return result
            
        except Exception as e:
            logger.error(f"Error in sensor fusion: {e}")
            return None
            
    def _analyze_threat_factors(self, sensor_tensors: Dict[SensorType, torch.Tensor]) -> List[str]:
        """Analyze threat factors from sensor data"""
        threat_factors = []
        
        try:
            # GPS analysis
            if SensorType.GPS in sensor_tensors:
                gps_data = sensor_tensors[SensorType.GPS].cpu().numpy()[0]
                speed = gps_data[2]
                
                if speed > 20:  # High speed
                    threat_factors.append("high_speed_movement")
                    
            # Accelerometer analysis
            if SensorType.ACCELEROMETER in sensor_tensors:
                accel_data = sensor_tensors[SensorType.ACCELEROMETER].cpu().numpy()[0]
                acceleration_magnitude = np.sqrt(np.sum(accel_data**2))
                
                if acceleration_magnitude > 15:  # High acceleration
                    threat_factors.append("sudden_movement")
                    
            # Microphone analysis
            if SensorType.MICROPHONE in sensor_tensors:
                mic_data = sensor_tensors[SensorType.MICROPHONE].cpu().numpy()[0]
                energy = np.mean(mic_data**2)
                
                if energy > 0.1:  # High audio energy
                    threat_factors.append("high_audio_activity")
                    
            # Camera analysis
            if SensorType.CAMERA in sensor_tensors:
                camera_data = sensor_tensors[SensorType.CAMERA].cpu().numpy()[0]
                face_detected = np.any(camera_data != 0)
                
                if face_detected:
                    threat_factors.append("people_detected")
                    
        except Exception as e:
            logger.error(f"Error analyzing threat factors: {e}")
            
        return threat_factors
        
    def _generate_recommendations(self, threat_score: float, threat_factors: List[str]) -> List[str]:
        """Generate recommendations based on threat assessment"""
        recommendations = []
        
        if threat_score > self.threat_thresholds['critical']:
            recommendations.append("Immediate action required - activate emergency protocols")
            recommendations.append("Contact emergency services")
            
        elif threat_score > self.threat_thresholds['high']:
            recommendations.append("High threat detected - increase monitoring")
            recommendations.append("Alert emergency contacts")
            
        elif threat_score > self.threat_thresholds['medium']:
            recommendations.append("Moderate threat - continue monitoring")
            recommendations.append("Stay alert and aware of surroundings")
            
        else:
            recommendations.append("Low threat level - normal operation")
            
        # Factor-specific recommendations
        if "high_speed_movement" in threat_factors:
            recommendations.append("Monitor for potential vehicle-related incidents")
            
        if "sudden_movement" in threat_factors:
            recommendations.append("Check for falls or sudden impacts")
            
        if "high_audio_activity" in threat_factors:
            recommendations.append("Monitor for distress sounds or loud noises")
            
        if "people_detected" in threat_factors:
            recommendations.append("Verify identity of detected individuals")
            
        return recommendations
        
    def get_fusion_summary(self) -> Dict[str, Any]:
        """Get summary of fusion results"""
        summary = {
            'total_fusions': len(self.fusion_history),
            'average_threat_score': 0.0,
            'average_confidence': 0.0,
            'threat_level_distribution': {},
            'sensor_usage': {},
            'recent_threat_factors': [],
            'active_sensors': []
        }
        
        if self.fusion_history:
            # Calculate averages
            threat_scores = [r.threat_score for r in self.fusion_history]
            confidences = [r.confidence for r in self.fusion_history]
            
            summary['average_threat_score'] = np.mean(threat_scores)
            summary['average_confidence'] = np.mean(confidences)
            
            # Threat level distribution
            for result in self.fusion_history:
                if result.threat_score > self.threat_thresholds['critical']:
                    level = 'critical'
                elif result.threat_score > self.threat_thresholds['high']:
                    level = 'high'
                elif result.threat_score > self.threat_thresholds['medium']:
                    level = 'medium'
                else:
                    level = 'low'
                    
                summary['threat_level_distribution'][level] = summary['threat_level_distribution'].get(level, 0) + 1
                
            # Sensor usage
            for sensor_type in SensorType:
                usage_count = sum(1 for r in self.fusion_history if sensor_type in r.contributing_sensors)
                summary['sensor_usage'][sensor_type.value] = usage_count
                
            # Recent threat factors
            recent_results = self.fusion_history[-10:]  # Last 10 results
            for result in recent_results:
                summary['recent_threat_factors'].extend(result.threat_factors)
                
            # Active sensors
            summary['active_sensors'] = [
                sensor_type.value for sensor_type in SensorType
                if self.sensor_buffers[sensor_type]
            ]
            
        return summary
        
    def clear_buffers(self):
        """Clear all sensor buffers"""
        for sensor_type in SensorType:
            self.sensor_buffers[sensor_type].clear()
            
    def export_fusion_data(self, filepath: str) -> bool:
        """Export fusion history to file"""
        try:
            export_data = []
            
            for result in self.fusion_history:
                result_data = {
                    'timestamp': result.timestamp.isoformat(),
                    'threat_score': result.threat_score,
                    'confidence': result.confidence,
                    'contributing_sensors': [s.value for s in result.contributing_sensors],
                    'threat_factors': result.threat_factors,
                    'recommendations': result.recommendations
                }
                export_data.append(result_data)
                
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Exported fusion data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting fusion data: {e}")
            return False 