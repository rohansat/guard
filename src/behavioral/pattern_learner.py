"""
Behavioral Pattern Learning Module for Guard AI
Learns user's normal behavior patterns and detects anomalies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LocationData:
    """Location data point"""
    timestamp: datetime
    latitude: float
    longitude: float
    accuracy: float
    speed: Optional[float] = None
    heading: Optional[float] = None


@dataclass
class ActivityData:
    """Activity data point"""
    timestamp: datetime
    activity_type: str  # walking, running, driving, stationary
    confidence: float
    duration: float


@dataclass
class BehavioralPattern:
    """User's behavioral pattern"""
    user_id: str
    pattern_type: str  # daily_routine, weekly_pattern, location_pattern
    data: Dict[str, Any]
    confidence: float
    last_updated: datetime


class LSTMPatternLearner(nn.Module):
    """
    LSTM-based neural network for learning behavioral patterns
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPatternLearner, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Anomaly score
        
    def forward(self, x, hidden=None):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x, hidden)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Final classification
        out = self.dropout(pooled)
        anomaly_score = torch.sigmoid(self.fc(out))
        
        return anomaly_score, (hidden, cell)


class BehavioralAnalyzer:
    """
    Main behavioral analysis system for Guard AI
    """
    
    def __init__(self, user_id: str, model_path: str = None):
        """
        Initialize the behavioral analyzer
        
        Args:
            user_id: Unique identifier for the user
            model_path: Path to pre-trained model (optional)
        """
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize LSTM model
        self.input_size = 8  # lat, lon, speed, heading, hour, day_of_week, activity_type, duration
        self.model = LSTMPatternLearner(input_size=self.input_size)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.model.to(self.device)
        self.model.eval()
        
        # Data storage
        self.location_history: List[LocationData] = []
        self.activity_history: List[ActivityData] = []
        self.behavioral_patterns: List[BehavioralPattern] = []
        
        # Pattern detection parameters
        self.sequence_length = 24  # hours
        self.anomaly_threshold = 0.7
        self.min_pattern_confidence = 0.8
        
        # Activity types
        self.activity_types = ['walking', 'running', 'driving', 'stationary', 'cycling']
        
    def add_location_data(self, location_data: LocationData):
        """
        Add new location data point
        
        Args:
            location_data: Location data point
        """
        self.location_history.append(location_data)
        
        # Keep only last 30 days of data
        cutoff_date = datetime.now() - timedelta(days=30)
        self.location_history = [
            loc for loc in self.location_history 
            if loc.timestamp > cutoff_date
        ]
        
    def add_activity_data(self, activity_data: ActivityData):
        """
        Add new activity data point
        
        Args:
            activity_data: Activity data point
        """
        self.activity_history.append(activity_data)
        
        # Keep only last 30 days of data
        cutoff_date = datetime.now() - timedelta(days=30)
        self.activity_history = [
            act for act in self.activity_history 
            if act.timestamp > cutoff_date
        ]
        
    def extract_features(self, location_data: LocationData, activity_data: Optional[ActivityData] = None) -> np.ndarray:
        """
        Extract features from location and activity data
        
        Args:
            location_data: Location data point
            activity_data: Activity data point (optional)
            
        Returns:
            Feature vector
        """
        features = []
        
        # Location features
        features.extend([
            location_data.latitude,
            location_data.longitude,
            location_data.speed or 0.0,
            location_data.heading or 0.0
        ])
        
        # Time features
        features.extend([
            location_data.timestamp.hour / 24.0,  # Normalized hour
            location_data.timestamp.weekday() / 7.0,  # Normalized day of week
        ])
        
        # Activity features
        if activity_data:
            activity_encoding = [1.0 if activity_data.activity_type == act_type else 0.0 
                               for act_type in self.activity_types]
            features.extend(activity_encoding)
            features.append(activity_data.duration / 3600.0)  # Duration in hours
        else:
            features.extend([0.0] * len(self.activity_types))
            features.append(0.0)
            
        return np.array(features)
        
    def create_sequence(self, current_time: datetime, hours_back: int = 24) -> torch.Tensor:
        """
        Create a sequence of features for the LSTM model
        
        Args:
            current_time: Current timestamp
            hours_back: Number of hours to look back
            
        Returns:
            Tensor of shape (sequence_length, input_size)
        """
        sequence = []
        start_time = current_time - timedelta(hours=hours_back)
        
        # Group data by hour
        for i in range(hours_back):
            target_time = start_time + timedelta(hours=i)
            
            # Find closest location data
            closest_location = None
            min_diff = float('inf')
            
            for loc in self.location_history:
                time_diff = abs((loc.timestamp - target_time).total_seconds())
                if time_diff < min_diff:
                    min_diff = time_diff
                    closest_location = loc
                    
            # Find closest activity data
            closest_activity = None
            min_diff = float('inf')
            
            for act in self.activity_history:
                time_diff = abs((act.timestamp - target_time).total_seconds())
                if time_diff < min_diff:
                    min_diff = time_diff
                    closest_activity = act
                    
            if closest_location:
                features = self.extract_features(closest_location, closest_activity)
            else:
                # Use zero features if no data available
                features = np.zeros(self.input_size)
                
            sequence.append(features)
            
        return torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
    def detect_anomaly(self, current_time: datetime = None) -> Dict[str, Any]:
        """
        Detect behavioral anomalies
        
        Args:
            current_time: Current timestamp (defaults to now)
            
        Returns:
            Dictionary with anomaly detection results
        """
        if current_time is None:
            current_time = datetime.now()
            
        try:
            # Create feature sequence
            sequence = self.create_sequence(current_time)
            sequence = sequence.to(self.device)
            
            # Get anomaly score
            with torch.no_grad():
                anomaly_score, _ = self.model(sequence)
                anomaly_score = anomaly_score.item()
                
            # Determine if anomaly
            is_anomaly = anomaly_score > self.anomaly_threshold
            
            # Get current location and activity
            current_location = self.location_history[-1] if self.location_history else None
            current_activity = self.activity_history[-1] if self.activity_history else None
            
            return {
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'confidence': anomaly_score,
                'current_location': current_location,
                'current_activity': current_activity,
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
            
    def learn_patterns(self) -> List[BehavioralPattern]:
        """
        Learn behavioral patterns from historical data
        
        Returns:
            List of learned patterns
        """
        patterns = []
        
        try:
            # Daily routine pattern
            daily_pattern = self._learn_daily_routine()
            if daily_pattern:
                patterns.append(daily_pattern)
                
            # Location pattern
            location_pattern = self._learn_location_pattern()
            if location_pattern:
                patterns.append(location_pattern)
                
            # Activity pattern
            activity_pattern = self._learn_activity_pattern()
            if activity_pattern:
                patterns.append(activity_pattern)
                
        except Exception as e:
            logger.error(f"Error learning patterns: {e}")
            
        return patterns
        
    def _learn_daily_routine(self) -> Optional[BehavioralPattern]:
        """Learn daily routine patterns"""
        if len(self.location_history) < 7:  # Need at least a week of data
            return None
            
        # Group by hour of day
        hourly_data = {}
        for loc in self.location_history:
            hour = loc.timestamp.hour
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append(loc)
            
        # Calculate average locations for each hour
        routine_data = {}
        for hour, locations in hourly_data.items():
            if len(locations) >= 3:  # Need at least 3 data points
                avg_lat = np.mean([loc.latitude for loc in locations])
                avg_lon = np.mean([loc.longitude for loc in locations])
                routine_data[hour] = {
                    'latitude': avg_lat,
                    'longitude': avg_lon,
                    'frequency': len(locations)
                }
                
        if routine_data:
            confidence = min(1.0, len(routine_data) / 24.0)
            return BehavioralPattern(
                user_id=self.user_id,
                pattern_type="daily_routine",
                data=routine_data,
                confidence=confidence,
                last_updated=datetime.now()
            )
            
        return None
        
    def _learn_location_pattern(self) -> Optional[BehavioralPattern]:
        """Learn location-based patterns"""
        if len(self.location_history) < 10:
            return None
            
        # Cluster locations to find frequent places
        locations = np.array([[loc.latitude, loc.longitude] for loc in self.location_history])
        
        # Simple clustering using distance threshold
        clusters = []
        visited = set()
        
        for i, loc in enumerate(locations):
            if i in visited:
                continue
                
            cluster = [i]
            visited.add(i)
            
            for j, other_loc in enumerate(locations):
                if j in visited:
                    continue
                    
                # Calculate distance (simplified)
                distance = np.sqrt((loc[0] - other_loc[0])**2 + (loc[1] - other_loc[1])**2)
                if distance < 0.01:  # ~1km threshold
                    cluster.append(j)
                    visited.add(j)
                    
            if len(cluster) >= 3:  # Minimum cluster size
                clusters.append(cluster)
                
        # Calculate cluster centers and frequencies
        location_data = {}
        for i, cluster in enumerate(clusters):
            cluster_locations = [locations[j] for j in cluster]
            center_lat = np.mean([loc[0] for loc in cluster_locations])
            center_lon = np.mean([loc[1] for loc in cluster_locations])
            
            location_data[f"cluster_{i}"] = {
                'latitude': center_lat,
                'longitude': center_lon,
                'frequency': len(cluster),
                'visits': [self.location_history[j].timestamp for j in cluster]
            }
            
        if location_data:
            confidence = min(1.0, len(location_data) / 10.0)
            return BehavioralPattern(
                user_id=self.user_id,
                pattern_type="location_pattern",
                data=location_data,
                confidence=confidence,
                last_updated=datetime.now()
            )
            
        return None
        
    def _learn_activity_pattern(self) -> Optional[BehavioralPattern]:
        """Learn activity-based patterns"""
        if len(self.activity_history) < 10:
            return None
            
        # Group activities by type and time
        activity_data = {}
        
        for activity in self.activity_history:
            activity_type = activity.activity_type
            hour = activity.timestamp.hour
            
            if activity_type not in activity_data:
                activity_data[activity_type] = {}
                
            if hour not in activity_data[activity_type]:
                activity_data[activity_type][hour] = []
                
            activity_data[activity_type][hour].append(activity.duration)
            
        # Calculate average durations
        pattern_data = {}
        for activity_type, hourly_data in activity_data.items():
            pattern_data[activity_type] = {}
            for hour, durations in hourly_data.items():
                if len(durations) >= 2:  # Need at least 2 data points
                    pattern_data[activity_type][hour] = {
                        'avg_duration': np.mean(durations),
                        'frequency': len(durations)
                    }
                    
        if pattern_data:
            confidence = min(1.0, len(pattern_data) / len(self.activity_types))
            return BehavioralPattern(
                user_id=self.user_id,
                pattern_type="activity_pattern",
                data=pattern_data,
                confidence=confidence,
                last_updated=datetime.now()
            )
            
        return None
        
    def get_safety_assessment(self, current_location: LocationData) -> Dict[str, Any]:
        """
        Assess safety based on current location and learned patterns
        
        Args:
            current_location: Current location data
            
        Returns:
            Safety assessment dictionary
        """
        assessment = {
            'safety_score': 0.5,  # Default neutral score
            'risk_factors': [],
            'recommendations': [],
            'confidence': 0.0
        }
        
        try:
            # Check if location is in known safe areas
            if self.behavioral_patterns:
                for pattern in self.behavioral_patterns:
                    if pattern.pattern_type == "location_pattern":
                        for cluster_name, cluster_data in pattern.data.items():
                            cluster_lat = cluster_data['latitude']
                            cluster_lon = cluster_data['longitude']
                            
                            # Calculate distance to known location
                            distance = np.sqrt(
                                (current_location.latitude - cluster_lat)**2 + 
                                (current_location.longitude - cluster_lon)**2
                            )
                            
                            if distance < 0.01:  # Within 1km of known location
                                assessment['safety_score'] += 0.2
                                assessment['confidence'] += 0.3
                                break
                                
            # Check time-based safety
            current_hour = current_location.timestamp.hour
            if current_hour < 6 or current_hour > 22:  # Late night/early morning
                assessment['safety_score'] -= 0.3
                assessment['risk_factors'].append('late_night_hours')
                assessment['recommendations'].append('Consider staying in well-lit areas')
                
            # Normalize safety score
            assessment['safety_score'] = max(0.0, min(1.0, assessment['safety_score']))
            assessment['confidence'] = min(1.0, assessment['confidence'])
            
        except Exception as e:
            logger.error(f"Error in safety assessment: {e}")
            
        return assessment 