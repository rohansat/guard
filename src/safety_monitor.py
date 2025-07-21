"""
Safety Monitoring System for Guard AI
Handles location tracking, close circle management, and automated alerts
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from geopy.distance import geodesic
import sqlite3
import os

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts that can be sent"""
    LOCATION_UPDATE = "location_update"
    SAFETY_CHECK = "safety_check"
    USER_HOME = "user_home"
    USER_MISSING = "user_missing"
    EMERGENCY = "emergency"
    TRACKING_START = "tracking_start"
    TRACKING_STOP = "tracking_stop"


class UpdateFrequency(Enum):
    """Update frequency options"""
    HOURLY = "hourly"
    EVERY_30_MIN = "30_min"
    EVERY_15_MIN = "15_min"
    REAL_TIME = "real_time"


@dataclass
class CloseCircleMember:
    """Member of user's close circle"""
    member_id: str
    name: str
    phone: str
    email: Optional[str] = None
    relationship: str = "friend"
    notification_preferences: Dict[str, bool] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.notification_preferences is None:
            self.notification_preferences = {
                "sms": True,
                "email": True,
                "push": True
            }


@dataclass
class LocationPoint:
    """Location data point"""
    timestamp: datetime
    latitude: float
    longitude: float
    accuracy: float
    speed: Optional[float] = None
    heading: Optional[float] = None
    activity: Optional[str] = None


@dataclass
class TrackingSession:
    """Active tracking session"""
    session_id: str
    user_id: str
    start_time: datetime
    expected_end_time: Optional[datetime] = None
    update_frequency: UpdateFrequency = UpdateFrequency.HOURLY
    close_circle_members: List[str] = None  # member IDs
    is_active: bool = True
    last_location: Optional[LocationPoint] = None
    last_alert_sent: Optional[datetime] = None
    safety_checks_enabled: bool = True
    auto_shutdown_hours: int = 6  # Auto shutdown after 6 hours if no "home" signal
    
    def __post_init__(self):
        if self.close_circle_members is None:
            self.close_circle_members = []


class SafetyMonitor:
    """
    Main safety monitoring system for Guard AI
    """
    
    def __init__(self, user_id: str, db_path: str = "safety_monitor.db"):
        """
        Initialize safety monitoring system
        
        Args:
            user_id: Unique identifier for the user
            db_path: Path to SQLite database
        """
        self.user_id = user_id
        self.db_path = db_path
        
        # Current tracking session
        self.current_session: Optional[TrackingSession] = None
        
        # Close circle members
        self.close_circle: Dict[str, CloseCircleMember] = {}
        
        # Home location (to detect when user returns)
        self.home_location: Optional[LocationPoint] = None
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_close_circle()
        self._load_home_location()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        logger.info(f"Safety Monitor initialized for user: {user_id}")
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS close_circle (
                    member_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    email TEXT,
                    relationship TEXT,
                    notification_preferences TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracking_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    expected_end_time TEXT,
                    update_frequency TEXT,
                    close_circle_members TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    last_location TEXT,
                    last_alert_sent TEXT,
                    safety_checks_enabled BOOLEAN DEFAULT 1,
                    auto_shutdown_hours INTEGER DEFAULT 6
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS location_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    accuracy REAL,
                    speed REAL,
                    heading REAL,
                    activity TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts_sent (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    alert_type TEXT NOT NULL,
                    recipient_id TEXT,
                    message TEXT,
                    timestamp TEXT NOT NULL,
                    status TEXT DEFAULT 'sent'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS home_location (
                    user_id TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    radius REAL DEFAULT 100,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _load_close_circle(self):
        """Load close circle members from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM close_circle WHERE is_active = 1')
            rows = cursor.fetchall()
            
            for row in rows:
                member = CloseCircleMember(
                    member_id=row[0],
                    name=row[1],
                    phone=row[2],
                    email=row[3],
                    relationship=row[4],
                    notification_preferences=json.loads(row[5]) if row[5] else {},
                    is_active=bool(row[6])
                )
                self.close_circle[member.member_id] = member
            
            conn.close()
            logger.info(f"Loaded {len(self.close_circle)} close circle members")
            
        except Exception as e:
            logger.error(f"Error loading close circle: {e}")
    
    def _load_home_location(self):
        """Load home location from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM home_location WHERE user_id = ?', (self.user_id,))
            row = cursor.fetchone()
            
            if row:
                self.home_location = LocationPoint(
                    timestamp=datetime.fromisoformat(row[4]),
                    latitude=row[1],
                    longitude=row[2],
                    accuracy=row[3]
                )
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading home location: {e}")
    
    def set_home_location(self, latitude: float, longitude: float, radius: float = 100):
        """Set user's home location"""
        try:
            self.home_location = LocationPoint(
                timestamp=datetime.now(),
                latitude=latitude,
                longitude=longitude,
                accuracy=radius
            )
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO home_location 
                (user_id, latitude, longitude, radius, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (self.user_id, latitude, longitude, radius, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Home location set: {latitude}, {longitude}")
            
        except Exception as e:
            logger.error(f"Error setting home location: {e}")
    
    def add_close_circle_member(self, member: CloseCircleMember) -> bool:
        """Add member to close circle"""
        try:
            self.close_circle[member.member_id] = member
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO close_circle 
                (member_id, name, phone, email, relationship, notification_preferences, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                member.member_id, member.name, member.phone, member.email,
                member.relationship, json.dumps(member.notification_preferences), member.is_active
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added close circle member: {member.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding close circle member: {e}")
            return False
    
    def remove_close_circle_member(self, member_id: str) -> bool:
        """Remove member from close circle"""
        try:
            if member_id in self.close_circle:
                del self.close_circle[member_id]
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE close_circle SET is_active = 0 WHERE member_id = ?', (member_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Removed close circle member: {member_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing close circle member: {e}")
            return False
    
    def start_tracking(self, 
                      update_frequency: UpdateFrequency = UpdateFrequency.HOURLY,
                      close_circle_members: List[str] = None,
                      expected_duration_hours: Optional[int] = None,
                      safety_checks: bool = True) -> str:
        """
        Start location tracking session
        
        Args:
            update_frequency: How often to send updates
            close_circle_members: List of member IDs to notify
            expected_duration_hours: Expected duration in hours
            safety_checks: Enable safety checks
            
        Returns:
            Session ID
        """
        try:
            session_id = f"session_{int(time.time())}"
            
            expected_end_time = None
            if expected_duration_hours:
                expected_end_time = datetime.now() + timedelta(hours=expected_duration_hours)
            
            # Use all close circle members if none specified
            if close_circle_members is None:
                close_circle_members = list(self.close_circle.keys())
            
            self.current_session = TrackingSession(
                session_id=session_id,
                user_id=self.user_id,
                start_time=datetime.now(),
                expected_end_time=expected_end_time,
                update_frequency=update_frequency,
                close_circle_members=close_circle_members,
                safety_checks_enabled=safety_checks
            )
            
            # Save to database
            self._save_tracking_session()
            
            # Start monitoring
            if not self.is_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
                self.is_monitoring = True
            
            # Send initial alert
            self._send_alert_to_close_circle(
                AlertType.TRACKING_START,
                f"Guard AI is now tracking {self.user_id}'s location. Updates will be sent {update_frequency.value}."
            )
            
            logger.info(f"Started tracking session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting tracking: {e}")
            return None
    
    def stop_tracking(self, reason: str = "user_requested") -> bool:
        """Stop current tracking session"""
        try:
            if self.current_session:
                self.current_session.is_active = False
                self._save_tracking_session()
                
                # Send final alert
                self._send_alert_to_close_circle(
                    AlertType.TRACKING_STOP,
                    f"{self.user_id} has stopped location tracking. Reason: {reason}"
                )
                
                logger.info(f"Stopped tracking session: {self.current_session.session_id}")
                self.current_session = None
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error stopping tracking: {e}")
            return False
    
    def add_location_point(self, latitude: float, longitude: float, 
                          accuracy: float = 10.0, speed: Optional[float] = None,
                          heading: Optional[float] = None, activity: Optional[str] = None):
        """Add a new location point"""
        try:
            location = LocationPoint(
                timestamp=datetime.now(),
                latitude=latitude,
                longitude=longitude,
                accuracy=accuracy,
                speed=speed,
                heading=heading,
                activity=activity
            )
            
            if self.current_session:
                self.current_session.last_location = location
                self._save_location_point(location)
                
                # Check if user is home
                if self._is_at_home(location):
                    self.stop_tracking("user_returned_home")
                    return
            
            logger.debug(f"Added location point: {latitude}, {longitude}")
            
        except Exception as e:
            logger.error(f"Error adding location point: {e}")
    
    def _is_at_home(self, location: LocationPoint) -> bool:
        """Check if user is at home location"""
        if not self.home_location:
            return False
        
        distance = geodesic(
            (location.latitude, location.longitude),
            (self.home_location.latitude, self.home_location.longitude)
        ).meters
        
        return distance <= self.home_location.accuracy
    
    def _save_tracking_session(self):
        """Save tracking session to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO tracking_sessions 
                (session_id, user_id, start_time, expected_end_time, update_frequency,
                 close_circle_members, is_active, last_location, last_alert_sent,
                 safety_checks_enabled, auto_shutdown_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_session.session_id,
                self.current_session.user_id,
                self.current_session.start_time.isoformat(),
                self.current_session.expected_end_time.isoformat() if self.current_session.expected_end_time else None,
                self.current_session.update_frequency.value,
                json.dumps(self.current_session.close_circle_members),
                self.current_session.is_active,
                json.dumps(asdict(self.current_session.last_location)) if self.current_session.last_location else None,
                self.current_session.last_alert_sent.isoformat() if self.current_session.last_alert_sent else None,
                self.current_session.safety_checks_enabled,
                self.current_session.auto_shutdown_hours
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving tracking session: {e}")
    
    def _save_location_point(self, location: LocationPoint):
        """Save location point to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO location_history 
                (session_id, timestamp, latitude, longitude, accuracy, speed, heading, activity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_session.session_id,
                location.timestamp.isoformat(),
                location.latitude,
                location.longitude,
                location.accuracy,
                location.speed,
                location.heading,
                location.activity
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving location point: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                if self.current_session and self.current_session.is_active:
                    await self._check_and_send_updates()
                    await self._check_safety_conditions()
                    await self._check_auto_shutdown()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_and_send_updates(self):
        """Check if it's time to send updates"""
        if not self.current_session or not self.current_session.last_location:
            return
        
        now = datetime.now()
        last_alert = self.current_session.last_alert_sent
        
        # Determine if we should send update based on frequency
        should_send = False
        
        if self.current_session.update_frequency == UpdateFrequency.REAL_TIME:
            should_send = True
        elif self.current_session.update_frequency == UpdateFrequency.EVERY_15_MIN:
            should_send = not last_alert or (now - last_alert).total_seconds() >= 900
        elif self.current_session.update_frequency == UpdateFrequency.EVERY_30_MIN:
            should_send = not last_alert or (now - last_alert).total_seconds() >= 1800
        elif self.current_session.update_frequency == UpdateFrequency.HOURLY:
            should_send = not last_alert or (now - last_alert).total_seconds() >= 3600
        
        if should_send:
            await self._send_location_update()
    
    async def _send_location_update(self):
        """Send location update to close circle"""
        if not self.current_session or not self.current_session.last_location:
            return
        
        location = self.current_session.last_location
        
        # Get nearby landmarks (simplified)
        nearby_landmarks = self._get_nearby_landmarks(location.latitude, location.longitude)
        
        message = (
            f"📍 Location Update for {self.user_id}:\n"
            f"Time: {location.timestamp.strftime('%H:%M')}\n"
            f"Location: {location.latitude:.4f}, {location.longitude:.4f}\n"
            f"Activity: {location.activity or 'Unknown'}\n"
            f"Nearby: {nearby_landmarks}\n"
            f"Status: Safe ✅"
        )
        
        self._send_alert_to_close_circle(AlertType.LOCATION_UPDATE, message)
        self.current_session.last_alert_sent = datetime.now()
        self._save_tracking_session()
    
    def _get_nearby_landmarks(self, lat: float, lon: float) -> str:
        """Get nearby landmarks (simplified implementation)"""
        # In a real implementation, this would use Google Places API or similar
        return "City center area"
    
    async def _check_safety_conditions(self):
        """Check for safety concerns"""
        if not self.current_session or not self.current_session.safety_checks_enabled:
            return
        
        # Add safety checks here (e.g., unusual movement patterns, long stops in unfamiliar areas)
        pass
    
    async def _check_auto_shutdown(self):
        """Check if auto-shutdown conditions are met"""
        if not self.current_session:
            return
        
        now = datetime.now()
        session_duration = now - self.current_session.start_time
        
        # Auto shutdown after specified hours
        if session_duration.total_seconds() > (self.current_session.auto_shutdown_hours * 3600):
            logger.info("Auto-shutdown triggered due to time limit")
            self.stop_tracking("auto_shutdown_time_limit")
    
    def _send_alert_to_close_circle(self, alert_type: AlertType, message: str):
        """Send alert to close circle members"""
        try:
            for member_id in self.current_session.close_circle_members:
                if member_id in self.close_circle:
                    member = self.close_circle[member_id]
                    
                    # Send based on preferences
                    if member.notification_preferences.get("sms", True):
                        self._send_sms(member.phone, message)
                    
                    if member.notification_preferences.get("email", True) and member.email:
                        self._send_email(member.email, message)
                    
                    # Log alert
                    self._log_alert(alert_type, member_id, message)
            
            logger.info(f"Sent {alert_type.value} alert to {len(self.current_session.close_circle_members)} members")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _send_sms(self, phone: str, message: str):
        """Send SMS (placeholder implementation)"""
        # In real implementation, use Twilio or similar service
        logger.info(f"SMS to {phone}: {message[:50]}...")
    
    def _send_email(self, email: str, message: str):
        """Send email (placeholder implementation)"""
        # In real implementation, use SMTP or email service
        logger.info(f"Email to {email}: {message[:50]}...")
    
    def _log_alert(self, alert_type: AlertType, recipient_id: str, message: str):
        """Log alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts_sent 
                (session_id, alert_type, recipient_id, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                self.current_session.session_id,
                alert_type.value,
                recipient_id,
                message,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
    
    def get_tracking_status(self) -> Dict[str, Any]:
        """Get current tracking status"""
        if not self.current_session:
            return {"is_tracking": False}
        
        return {
            "is_tracking": self.current_session.is_active,
            "session_id": self.current_session.session_id,
            "start_time": self.current_session.start_time.isoformat(),
            "update_frequency": self.current_session.update_frequency.value,
            "close_circle_members": len(self.current_session.close_circle_members),
            "last_location": asdict(self.current_session.last_location) if self.current_session.last_location else None,
            "last_alert_sent": self.current_session.last_alert_sent.isoformat() if self.current_session.last_alert_sent else None
        }
    
    def get_close_circle_summary(self) -> Dict[str, Any]:
        """Get close circle summary"""
        return {
            "total_members": len(self.close_circle),
            "active_members": len([m for m in self.close_circle.values() if m.is_active]),
            "members": [asdict(member) for member in self.close_circle.values()]
        } 