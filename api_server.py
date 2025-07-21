"""
Guard AI API Server
REST API for the voice-activated personal safety assistant
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import logging
import uvicorn
from datetime import datetime

from src.main import GuardAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Guard AI API",
    description="Voice-activated personal safety assistant API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Guard AI instance
guard_ai: Optional[GuardAI] = None


# Pydantic models for API requests/responses
class VoiceCommandRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    sample_rate: int = 16000


class LocationDataRequest(BaseModel):
    latitude: float
    longitude: float
    accuracy: float = 10.0
    speed: Optional[float] = None
    heading: Optional[float] = None


class ActivityDataRequest(BaseModel):
    activity_type: str
    confidence: float
    duration: float


class CameraFrameRequest(BaseModel):
    image_data: str  # Base64 encoded image


class SensorDataRequest(BaseModel):
    accelerometer: Optional[Dict[str, float]] = None
    microphone: Optional[str] = None  # Base64 encoded audio
    gyroscope: Optional[Dict[str, float]] = None
    magnetometer: Optional[Dict[str, float]] = None


class EmergencyContactRequest(BaseModel):
    name: str
    phone: str
    email: Optional[str] = None
    relationship: str = "friend"
    priority: int = 1


class FaceDataRequest(BaseModel):
    person_id: str
    name: str
    face_encodings: List[List[float]]
    trust_level: float = 1.0


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Guard AI on startup"""
    global guard_ai
    try:
        guard_ai = GuardAI(user_id="api_user")
        logger.info("Guard AI initialized for API server")
    except Exception as e:
        logger.error(f"Error initializing Guard AI: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global guard_ai
    if guard_ai:
        await guard_ai.stop()
        logger.info("Guard AI stopped")


@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint"""
    return APIResponse(
        success=True,
        message="Guard AI API is running",
        data={"version": "1.0.0", "status": "active"}
    )


@app.get("/status", response_model=APIResponse)
async def get_status():
    """Get system status"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    status = guard_ai.get_system_status()
    return APIResponse(
        success=True,
        message="System status retrieved",
        data=status
    )


@app.get("/safety-summary", response_model=APIResponse)
async def get_safety_summary():
    """Get comprehensive safety summary"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    summary = guard_ai.get_safety_summary()
    return APIResponse(
        success=True,
        message="Safety summary retrieved",
        data=summary
    )


@app.post("/voice-command", response_model=APIResponse)
async def process_voice_command(request: VoiceCommandRequest):
    """Process voice command"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        import base64
        audio_data = base64.b64decode(request.audio_data)
        
        result = await guard_ai.process_voice_command(audio_data, request.sample_rate)
        
        return APIResponse(
            success=result['success'],
            message=result.get('response', 'Voice command processed'),
            data=result
        )
    except Exception as e:
        logger.error(f"Error processing voice command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/location", response_model=APIResponse)
async def add_location_data(request: LocationDataRequest):
    """Add location data"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        await guard_ai.add_location_data(
            request.latitude,
            request.longitude,
            request.accuracy,
            request.speed,
            request.heading
        )
        
        return APIResponse(
            success=True,
            message="Location data added successfully"
        )
    except Exception as e:
        logger.error(f"Error adding location data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/activity", response_model=APIResponse)
async def add_activity_data(request: ActivityDataRequest):
    """Add activity data"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        await guard_ai.add_activity_data(
            request.activity_type,
            request.confidence,
            request.duration
        )
        
        return APIResponse(
            success=True,
            message="Activity data added successfully"
        )
    except Exception as e:
        logger.error(f"Error adding activity data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/camera", response_model=APIResponse)
async def process_camera_frame(request: CameraFrameRequest):
    """Process camera frame"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        import base64
        image_data = base64.b64decode(request.image_data)
        
        result = await guard_ai.process_camera_frame(image_data)
        
        return APIResponse(
            success=result['success'],
            message="Camera frame processed",
            data=result
        )
    except Exception as e:
        logger.error(f"Error processing camera frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sensors", response_model=APIResponse)
async def add_sensor_data(request: SensorDataRequest):
    """Add sensor data"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        # Add accelerometer data
        if request.accelerometer:
            await guard_ai.add_accelerometer_data(
                request.accelerometer['x'],
                request.accelerometer['y'],
                request.accelerometer['z']
            )
        
        # Add microphone data
        if request.microphone:
            import base64
            audio_data = base64.b64decode(request.microphone)
            await guard_ai.add_microphone_data(audio_data)
        
        # Add gyroscope data (if implemented)
        if request.gyroscope:
            # TODO: Implement gyroscope data processing
            pass
        
        # Add magnetometer data (if implemented)
        if request.magnetometer:
            # TODO: Implement magnetometer data processing
            pass
        
        return APIResponse(
            success=True,
            message="Sensor data added successfully"
        )
    except Exception as e:
        logger.error(f"Error adding sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emergency-contacts", response_model=APIResponse)
async def add_emergency_contact(request: EmergencyContactRequest):
    """Add emergency contact"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        success = guard_ai.add_emergency_contact(
            request.name,
            request.phone,
            request.email,
            request.relationship,
            request.priority
        )
        
        if success:
            return APIResponse(
                success=True,
                message="Emergency contact added successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to add emergency contact")
    except Exception as e:
        logger.error(f"Error adding emergency contact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/faces", response_model=APIResponse)
async def add_known_face(request: FaceDataRequest):
    """Add known face"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        # Convert face encodings to numpy arrays
        import numpy as np
        face_encodings = [np.array(encoding) for encoding in request.face_encodings]
        
        success = guard_ai.add_known_face(
            request.person_id,
            request.name,
            face_encodings,
            request.trust_level
        )
        
        if success:
            return APIResponse(
                success=True,
                message="Known face added successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to add known face")
    except Exception as e:
        logger.error(f"Error adding known face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emergency-contacts", response_model=APIResponse)
async def get_emergency_contacts():
    """Get emergency contacts"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        # This would need to be implemented in GuardAI class
        # For now, return empty list
        contacts = []
        
        return APIResponse(
            success=True,
            message="Emergency contacts retrieved",
            data={"contacts": contacts}
        )
    except Exception as e:
        logger.error(f"Error getting emergency contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faces", response_model=APIResponse)
async def get_known_faces():
    """Get known faces summary"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        summary = guard_ai.get_safety_summary()
        face_summary = summary['facial_recognition']
        
        return APIResponse(
            success=True,
            message="Known faces summary retrieved",
            data=face_summary
        )
    except Exception as e:
        logger.error(f"Error getting known faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start", response_model=APIResponse)
async def start_system():
    """Start Guard AI system"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        # Start the system in background
        asyncio.create_task(guard_ai.start())
        
        return APIResponse(
            success=True,
            message="Guard AI system started"
        )
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop", response_model=APIResponse)
async def stop_system():
    """Stop Guard AI system"""
    global guard_ai
    if not guard_ai:
        raise HTTPException(status_code=500, detail="Guard AI not initialized")
    
    try:
        await guard_ai.stop()
        
        return APIResponse(
            success=True,
            message="Guard AI system stopped"
        )
    except Exception as e:
        logger.error(f"Error stopping system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint"""
    global guard_ai
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "guard_ai_initialized": guard_ai is not None
    }
    
    if guard_ai:
        try:
            system_status = guard_ai.get_system_status()
            health_status["system_status"] = system_status['system_status']
            health_status["is_active"] = system_status['is_active']
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
    
    return APIResponse(
        success=health_status["status"] == "healthy",
        message="Health check completed",
        data=health_status
    )


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 