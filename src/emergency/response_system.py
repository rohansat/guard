"""
Emergency Response System for Guard AI
Makes intelligent decisions about emergency responses using reinforcement learning
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
import aiohttp
from enum import Enum
import os

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseAction(Enum):
    """Emergency response actions"""
    MONITOR = "monitor"
    ALERT_CONTACTS = "alert_contacts"
    CALL_EMERGENCY = "call_emergency"
    ACTIVATE_SOS = "activate_sos"
    TRACK_LOCATION = "track_location"
    RECORD_AUDIO = "record_audio"
    RECORD_VIDEO = "record_video"


@dataclass
class EmergencyEvent:
    """Emergency event data"""
    event_id: str
    timestamp: datetime
    threat_level: ThreatLevel
    location: Dict[str, float]  # lat, lon
    description: str
    confidence: float
    triggered_by: str  # voice, location, behavior, facial, sensor
    actions_taken: List[ResponseAction]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class Contact:
    """Emergency contact information"""
    contact_id: str
    name: str
    phone: str
    relationship: str
    priority: int  # 1 = highest priority
    email: Optional[str] = None
    notification_preferences: Dict[str, bool] = None  # sms, call, email


class PolicyNetwork(nn.Module):
    """
    Policy network for reinforcement learning-based decision making
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Return action probabilities
        return F.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """
    Value network for estimating state values
    """
    
    def __init__(self, state_size: int, hidden_size: int = 128):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class EmergencyResponseSystem:
    """
    Main emergency response system for Guard AI
    """
    
    def __init__(self, user_id: str, model_path: str = None):
        """
        Initialize the emergency response system
        
        Args:
            user_id: Unique identifier for the user
            model_path: Path to pre-trained model (optional)
        """
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State and action spaces
        self.state_size = 15  # threat_level, location_risk, time_risk, behavior_anomaly, etc.
        self.action_size = len(ResponseAction)
        
        # Initialize networks
        self.policy_network = PolicyNetwork(self.state_size, self.action_size)
        self.value_network = ValueNetwork(self.state_size)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_network.load_state_dict(checkpoint['policy'])
            self.value_network.load_state_dict(checkpoint['value'])
            
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        
        # Emergency contacts
        self.emergency_contacts: List[Contact] = []
        self.load_contacts()
        
        # Emergency events
        self.emergency_events: List[EmergencyEvent] = []
        self.active_events: List[EmergencyEvent] = []
        
        # Response thresholds
        self.thresholds = {
            ThreatLevel.LOW: 0.3,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.7,
            ThreatLevel.CRITICAL: 0.9
        }
        
        # Response protocols
        self.response_protocols = {
            ThreatLevel.LOW: [ResponseAction.MONITOR],
            ThreatLevel.MEDIUM: [ResponseAction.MONITOR, ResponseAction.TRACK_LOCATION],
            ThreatLevel.HIGH: [ResponseAction.MONITOR, ResponseAction.TRACK_LOCATION, ResponseAction.ALERT_CONTACTS],
            ThreatLevel.CRITICAL: [ResponseAction.MONITOR, ResponseAction.TRACK_LOCATION, ResponseAction.ALERT_CONTACTS, ResponseAction.CALL_EMERGENCY]
        }
        
        # Learning parameters
        self.learning_rate = 0.001
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
        
    def load_contacts(self):
        """Load emergency contacts from file"""
        try:
            contacts_file = f"contacts_{self.user_id}.json"
            if os.path.exists(contacts_file):
                with open(contacts_file, 'r') as f:
                    contacts_data = json.load(f)
                    
                for contact_data in contacts_data:
                    contact = Contact(
                        contact_id=contact_data['contact_id'],
                        name=contact_data['name'],
                        phone=contact_data['phone'],
                        email=contact_data.get('email'),
                        relationship=contact_data['relationship'],
                        priority=contact_data['priority'],
                        notification_preferences=contact_data.get('notification_preferences', {})
                    )
                    self.emergency_contacts.append(contact)
                    
                logger.info(f"Loaded {len(self.emergency_contacts)} emergency contacts")
            else:
                logger.info("No emergency contacts found")
                
        except Exception as e:
            logger.error(f"Error loading contacts: {e}")
            
    def save_contacts(self):
        """Save emergency contacts to file"""
        try:
            contacts_file = f"contacts_{self.user_id}.json"
            contacts_data = []
            
            for contact in self.emergency_contacts:
                contact_data = {
                    'contact_id': contact.contact_id,
                    'name': contact.name,
                    'phone': contact.phone,
                    'email': contact.email,
                    'relationship': contact.relationship,
                    'priority': contact.priority,
                    'notification_preferences': contact.notification_preferences
                }
                contacts_data.append(contact_data)
                
            with open(contacts_file, 'w') as f:
                json.dump(contacts_data, f, indent=2)
                
            logger.info(f"Saved {len(self.emergency_contacts)} emergency contacts")
            
        except Exception as e:
            logger.error(f"Error saving contacts: {e}")
            
    def add_contact(self, contact: Contact) -> bool:
        """Add a new emergency contact"""
        try:
            self.emergency_contacts.append(contact)
            self.save_contacts()
            return True
        except Exception as e:
            logger.error(f"Error adding contact: {e}")
            return False
            
    def remove_contact(self, contact_id: str) -> bool:
        """Remove an emergency contact"""
        try:
            self.emergency_contacts = [c for c in self.emergency_contacts if c.contact_id != contact_id]
            self.save_contacts()
            return True
        except Exception as e:
            logger.error(f"Error removing contact: {e}")
            return False
            
    def create_state_vector(self, threat_data: Dict[str, Any]) -> torch.Tensor:
        """
        Create state vector from threat data
        
        Args:
            threat_data: Dictionary containing threat information
            
        Returns:
            State tensor
        """
        state = []
        
        # Threat level (one-hot encoded)
        threat_level = threat_data.get('threat_level', ThreatLevel.LOW)
        threat_encoding = [1.0 if threat_level == level else 0.0 for level in ThreatLevel]
        state.extend(threat_encoding)
        
        # Location risk (0-1)
        state.append(threat_data.get('location_risk', 0.0))
        
        # Time risk (0-1)
        state.append(threat_data.get('time_risk', 0.0))
        
        # Behavior anomaly score (0-1)
        state.append(threat_data.get('behavior_anomaly', 0.0))
        
        # Voice distress score (0-1)
        state.append(threat_data.get('voice_distress', 0.0))
        
        # Unknown faces count (normalized)
        unknown_faces = threat_data.get('unknown_faces', 0)
        state.append(min(unknown_faces / 10.0, 1.0))
        
        # Low trust individuals (normalized)
        low_trust = threat_data.get('low_trust_individuals', 0)
        state.append(min(low_trust / 5.0, 1.0))
        
        # Previous emergency events (normalized)
        recent_events = len([e for e in self.emergency_events 
                           if (datetime.now() - e.timestamp).hours < 24])
        state.append(min(recent_events / 10.0, 1.0))
        
        # User response time (0-1, higher is worse)
        response_time = threat_data.get('user_response_time', 0.0)
        state.append(min(response_time / 300.0, 1.0))  # 5 minutes max
        
        # Environmental factors
        state.append(threat_data.get('poor_lighting', 0.0))
        state.append(threat_data.get('isolated_area', 0.0))
        state.append(threat_data.get('high_crime_area', 0.0))
        
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
    def select_action(self, state: torch.Tensor, training: bool = False) -> ResponseAction:
        """
        Select action using policy network
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        state = state.to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_network(state)
            
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            action_idx = np.random.randint(0, self.action_size)
        else:
            # Exploitation: best action
            action_idx = torch.argmax(action_probs).item()
            
        return list(ResponseAction)[action_idx]
        
    def assess_threat(self, sensor_data: Dict[str, Any]) -> ThreatLevel:
        """
        Assess threat level from sensor data
        
        Args:
            sensor_data: Data from various sensors
            
        Returns:
            Threat level
        """
        threat_score = 0.0
        
        # Voice distress
        if sensor_data.get('voice_distress', False):
            threat_score += 0.4
            
        # Behavior anomaly
        anomaly_score = sensor_data.get('behavior_anomaly', 0.0)
        threat_score += anomaly_score * 0.3
            
        # Location risk
        location_risk = sensor_data.get('location_risk', 0.0)
        threat_score += location_risk * 0.2
            
        # Time risk (late night/early morning)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            threat_score += 0.2
            
        # Unknown faces
        unknown_faces = sensor_data.get('unknown_faces', 0)
        threat_score += min(unknown_faces * 0.1, 0.3)
            
        # Low trust individuals
        low_trust = sensor_data.get('low_trust_individuals', 0)
        threat_score += min(low_trust * 0.15, 0.4)
            
        # Determine threat level
        if threat_score >= self.thresholds[ThreatLevel.CRITICAL]:
            return ThreatLevel.CRITICAL
        elif threat_score >= self.thresholds[ThreatLevel.HIGH]:
            return ThreatLevel.HIGH
        elif threat_score >= self.thresholds[ThreatLevel.MEDIUM]:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
            
    def create_emergency_event(self, threat_data: Dict[str, Any]) -> EmergencyEvent:
        """
        Create a new emergency event
        
        Args:
            threat_data: Threat assessment data
            
        Returns:
            Emergency event
        """
        event_id = f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        event = EmergencyEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            threat_level=threat_data['threat_level'],
            location=threat_data.get('location', {'lat': 0.0, 'lon': 0.0}),
            description=threat_data.get('description', 'Unknown threat'),
            confidence=threat_data.get('confidence', 0.5),
            triggered_by=threat_data.get('triggered_by', 'unknown'),
            actions_taken=[]
        )
        
        self.emergency_events.append(event)
        self.active_events.append(event)
        
        return event
        
    async def execute_response(self, event: EmergencyEvent, action: ResponseAction) -> bool:
        """
        Execute emergency response action
        
        Args:
            event: Emergency event
            action: Response action to execute
            
        Returns:
            True if action was successful
        """
        try:
            if action == ResponseAction.MONITOR:
                logger.info(f"Monitoring emergency event {event.event_id}")
                return True
                
            elif action == ResponseAction.TRACK_LOCATION:
                logger.info(f"Tracking location for event {event.event_id}")
                # Implement location tracking
                return True
                
            elif action == ResponseAction.ALERT_CONTACTS:
                success = await self._alert_contacts(event)
                return success
                
            elif action == ResponseAction.CALL_EMERGENCY:
                success = await self._call_emergency_services(event)
                return success
                
            elif action == ResponseAction.ACTIVATE_SOS:
                success = await self._activate_sos(event)
                return success
                
            elif action == ResponseAction.RECORD_AUDIO:
                logger.info(f"Recording audio for event {event.event_id}")
                # Implement audio recording
                return True
                
            elif action == ResponseAction.RECORD_VIDEO:
                logger.info(f"Recording video for event {event.event_id}")
                # Implement video recording
                return True
                
            else:
                logger.warning(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing response {action}: {e}")
            return False
            
    async def _alert_contacts(self, event: EmergencyEvent) -> bool:
        """Alert emergency contacts"""
        try:
            # Sort contacts by priority
            sorted_contacts = sorted(self.emergency_contacts, key=lambda c: c.priority)
            
            for contact in sorted_contacts[:3]:  # Alert top 3 contacts
                message = self._create_alert_message(event, contact)
                
                if contact.notification_preferences.get('sms', True):
                    await self._send_sms(contact.phone, message)
                    
                if contact.notification_preferences.get('email', False) and contact.email:
                    await self._send_email(contact.email, message)
                    
            logger.info(f"Alerted contacts for event {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error alerting contacts: {e}")
            return False
            
    async def _call_emergency_services(self, event: EmergencyEvent) -> bool:
        """Call emergency services"""
        try:
            # This would integrate with emergency services API
            # For now, just log the action
            logger.info(f"Calling emergency services for event {event.event_id}")
            
            # Simulate emergency call
            emergency_data = {
                'event_id': event.event_id,
                'location': event.location,
                'threat_level': event.threat_level.value,
                'description': event.description,
                'timestamp': event.timestamp.isoformat()
            }
            
            logger.info(f"Emergency call data: {emergency_data}")
            return True
            
        except Exception as e:
            logger.error(f"Error calling emergency services: {e}")
            return False
            
    async def _activate_sos(self, event: EmergencyEvent) -> bool:
        """Activate SOS mode"""
        try:
            # Implement SOS activation
            logger.info(f"Activating SOS for event {event.event_id}")
            
            # This would trigger immediate emergency protocols
            # - Continuous location tracking
            # - Audio/video recording
            # - Emergency services notification
            # - Contact all emergency contacts
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating SOS: {e}")
            return False
            
    def _create_alert_message(self, event: EmergencyEvent, contact: Contact) -> str:
        """Create alert message for contact"""
        message = f"EMERGENCY ALERT - {contact.name}\n\n"
        message += f"Guard AI has detected a {event.threat_level.value} level threat.\n"
        message += f"Location: {event.location.get('lat', 0):.4f}, {event.location.get('lon', 0):.4f}\n"
        message += f"Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"Description: {event.description}\n\n"
        message += "Please check on the user immediately."
        
        return message
        
    async def _send_sms(self, phone: str, message: str) -> bool:
        """Send SMS message (placeholder)"""
        # This would integrate with SMS service
        logger.info(f"Sending SMS to {phone}: {message[:50]}...")
        return True
        
    async def _send_email(self, email: str, message: str) -> bool:
        """Send email message (placeholder)"""
        # This would integrate with email service
        logger.info(f"Sending email to {email}: {message[:50]}...")
        return True
        
    def process_threat(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process threat data and determine response
        
        Args:
            sensor_data: Data from various sensors
            
        Returns:
            Response decision and actions
        """
        try:
            # Assess threat level
            threat_level = self.assess_threat(sensor_data)
            
            # Create threat data
            threat_data = {
                'threat_level': threat_level,
                'location_risk': sensor_data.get('location_risk', 0.0),
                'time_risk': sensor_data.get('time_risk', 0.0),
                'behavior_anomaly': sensor_data.get('behavior_anomaly', 0.0),
                'voice_distress': sensor_data.get('voice_distress', 0.0),
                'unknown_faces': sensor_data.get('unknown_faces', 0),
                'low_trust_individuals': sensor_data.get('low_trust_individuals', 0),
                'user_response_time': sensor_data.get('user_response_time', 0.0),
                'poor_lighting': sensor_data.get('poor_lighting', 0.0),
                'isolated_area': sensor_data.get('isolated_area', 0.0),
                'high_crime_area': sensor_data.get('high_crime_area', 0.0),
                'location': sensor_data.get('location', {'lat': 0.0, 'lon': 0.0}),
                'description': sensor_data.get('description', 'Unknown threat'),
                'confidence': sensor_data.get('confidence', 0.5),
                'triggered_by': sensor_data.get('triggered_by', 'unknown')
            }
            
            # Create state vector
            state = self.create_state_vector(threat_data)
            
            # Select action
            action = self.select_action(state)
            
            # Get recommended actions based on threat level
            recommended_actions = self.response_protocols.get(threat_level, [])
            
            # Create emergency event if threat is significant
            event = None
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                event = self.create_emergency_event(threat_data)
                
            return {
                'threat_level': threat_level.value,
                'selected_action': action.value,
                'recommended_actions': [a.value for a in recommended_actions],
                'emergency_event': event.event_id if event else None,
                'confidence': threat_data['confidence'],
                'state_vector': state.cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"Error processing threat: {e}")
            return {
                'threat_level': ThreatLevel.LOW.value,
                'selected_action': ResponseAction.MONITOR.value,
                'error': str(e)
            }
            
    def resolve_event(self, event_id: str, resolution_notes: str = "") -> bool:
        """
        Resolve an emergency event
        
        Args:
            event_id: Event ID to resolve
            resolution_notes: Notes about resolution
            
        Returns:
            True if event was resolved
        """
        try:
            for event in self.active_events:
                if event.event_id == event_id:
                    event.resolved = True
                    event.resolution_time = datetime.now()
                    
                    # Move from active to resolved
                    self.active_events.remove(event)
                    
                    logger.info(f"Resolved emergency event {event_id}")
                    return True
                    
            logger.warning(f"Event {event_id} not found in active events")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving event: {e}")
            return False
            
    def get_emergency_summary(self) -> Dict[str, Any]:
        """Get summary of emergency events"""
        summary = {
            'total_events': len(self.emergency_events),
            'active_events': len(self.active_events),
            'resolved_events': len([e for e in self.emergency_events if e.resolved]),
            'threat_level_distribution': {},
            'recent_events': [],
            'contacts_count': len(self.emergency_contacts)
        }
        
        # Threat level distribution
        for event in self.emergency_events:
            level = event.threat_level.value
            summary['threat_level_distribution'][level] = summary['threat_level_distribution'].get(level, 0) + 1
            
        # Recent events (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_events = [e for e in self.emergency_events if e.timestamp > week_ago]
        
        for event in recent_events[-10:]:  # Last 10 events
            summary['recent_events'].append({
                'event_id': event.event_id,
                'threat_level': event.threat_level.value,
                'timestamp': event.timestamp.isoformat(),
                'resolved': event.resolved,
                'description': event.description
            })
            
        return summary 