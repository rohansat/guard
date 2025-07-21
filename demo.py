"""
Guard AI Demo Script
Demonstrates the voice-activated personal safety assistant
"""

import asyncio
import numpy as np
import time
from datetime import datetime
import logging

from src.main import GuardAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_voice_commands(guard_ai):
    """Demo voice command processing"""
    print("\n=== Voice Command Demo ===")
    
    # Simulate voice commands
    voice_commands = [
        "Guard, track my location",
        "Help me, I'm in danger",
        "Check in, I'm safe",
        "Scan faces around me",
        "How safe is this area?",
        "System status"
    ]
    
    for command in voice_commands:
        print(f"\nUser: {command}")
        
        # Simulate audio data (random bytes)
        audio_data = np.random.randn(16000).astype(np.float32).tobytes()
        
        # Process voice command
        result = await guard_ai.process_voice_command(audio_data)
        
        if result['success']:
            print(f"Guard AI: {result['response']}")
            print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        else:
            print(f"Error: {result['error']}")
            
        await asyncio.sleep(1)


async def demo_location_tracking(guard_ai):
    """Demo location tracking and behavioral analysis"""
    print("\n=== Location Tracking Demo ===")
    
    # Simulate location data over time
    locations = [
        (37.7749, -122.4194, 0.0),  # Home
        (37.7849, -122.4094, 5.0),  # Moving
        (37.7949, -122.3994, 10.0),  # Further away
        (37.8049, -122.3894, 15.0),  # Even further
        (37.8149, -122.3794, 20.0),  # Far from home
    ]
    
    for i, (lat, lon, speed) in enumerate(locations):
        print(f"\nLocation {i+1}: ({lat:.4f}, {lon:.4f}) - Speed: {speed} m/s")
        
        # Add location data
        await guard_ai.add_location_data(lat, lon, accuracy=5.0, speed=speed)
        
        # Add activity data
        activity_type = "walking" if speed > 0 else "stationary"
        await guard_ai.add_activity_data(activity_type, confidence=0.9, duration=60.0)
        
        # Get safety assessment
        summary = guard_ai.get_safety_summary()
        threat_level = summary['overall_threat_level']
        print(f"Threat Level: {threat_level}")
        
        await asyncio.sleep(2)


async def demo_facial_recognition(guard_ai):
    """Demo facial recognition"""
    print("\n=== Facial Recognition Demo ===")
    
    # Simulate camera frames
    for i in range(3):
        print(f"\nCamera Frame {i+1}:")
        
        # Simulate image data (random bytes)
        image_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8).tobytes()
        
        # Process camera frame
        result = await guard_ai.process_camera_frame(image_data)
        
        if result['success']:
            print(f"Faces detected: {result['faces_detected']}")
            print(f"Known faces: {len(result['known_faces'])}")
            print(f"Unknown faces: {len(result['unknown_faces'])}")
            
            if result['suspicious_activity']:
                print("⚠️  Suspicious activity detected!")
                for alert in result['alerts']:
                    print(f"Alert: {alert}")
        else:
            print(f"Error: {result['error']}")
            
        await asyncio.sleep(1)


async def demo_sensor_fusion(guard_ai):
    """Demo sensor fusion"""
    print("\n=== Sensor Fusion Demo ===")
    
    # Simulate sensor data
    for i in range(5):
        print(f"\nSensor Data {i+1}:")
        
        # Add accelerometer data
        accel_x = np.random.uniform(-10, 10)
        accel_y = np.random.uniform(-10, 10)
        accel_z = np.random.uniform(-10, 10)
        await guard_ai.add_accelerometer_data(accel_x, accel_y, accel_z)
        
        # Add microphone data
        audio_data = np.random.randn(16000).astype(np.float32).tobytes()
        await guard_ai.add_microphone_data(audio_data)
        
        # Add location data
        lat = 37.7749 + np.random.uniform(-0.01, 0.01)
        lon = -122.4194 + np.random.uniform(-0.01, 0.01)
        await guard_ai.add_location_data(lat, lon)
        
        print(f"Accelerometer: ({accel_x:.2f}, {accel_y:.2f}, {accel_z:.2f})")
        print(f"Location: ({lat:.4f}, {lon:.4f})")
        
        # Get fusion summary
        summary = guard_ai.get_safety_summary()
        fusion_summary = summary['sensor_fusion']
        
        if fusion_summary.get('total_fusions', 0) > 0:
            avg_threat = fusion_summary.get('average_threat_score', 0.0)
            print(f"Average threat score: {avg_threat:.3f}")
            
            recent_factors = fusion_summary.get('recent_threat_factors', [])
            if recent_factors:
                print(f"Recent threat factors: {recent_factors}")
        
        await asyncio.sleep(2)


async def demo_emergency_response(guard_ai):
    """Demo emergency response system"""
    print("\n=== Emergency Response Demo ===")
    
    # Simulate emergency scenarios
    scenarios = [
        {
            'name': 'High-speed movement',
            'location': (37.7749, -122.4194),
            'speed': 25.0,
            'description': 'User moving at high speed'
        },
        {
            'name': 'Late night activity',
            'location': (37.7849, -122.4094),
            'speed': 0.0,
            'description': 'User active late at night'
        },
        {
            'name': 'Unknown area',
            'location': (37.8149, -122.3794),
            'speed': 5.0,
            'description': 'User in unfamiliar area'
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        # Add location data
        lat, lon = scenario['location']
        await guard_ai.add_location_data(lat, lon, speed=scenario['speed'])
        
        # Get emergency summary
        summary = guard_ai.get_safety_summary()
        emergency_summary = summary['emergency_events']
        
        print(f"Active events: {emergency_summary.get('active_events', 0)}")
        print(f"Total events: {emergency_summary.get('total_events', 0)}")
        
        # Check threat level
        threat_level = summary['overall_threat_level']
        print(f"Overall threat level: {threat_level}")
        
        if threat_level in ['high', 'critical']:
            print("🚨 HIGH THREAT DETECTED - Emergency protocols activated!")
        
        await asyncio.sleep(3)


async def demo_system_status(guard_ai):
    """Demo system status and monitoring"""
    print("\n=== System Status Demo ===")
    
    # Get system status
    status = guard_ai.get_system_status()
    print(f"\nSystem Status: {status['system_status']}")
    print(f"Active: {status['is_active']}")
    print(f"Uptime: {status['uptime']:.1f} seconds")
    
    # Get comprehensive safety summary
    summary = guard_ai.get_safety_summary()
    print(f"\nSafety Summary:")
    print(f"Overall threat level: {summary['overall_threat_level']}")
    print(f"Behavioral analysis: {summary['behavioral_analysis'].get('safety_score', 'N/A')}")
    print(f"Facial recognition - Total people: {summary['facial_recognition'].get('total_people', 0)}")
    print(f"Emergency events - Active: {summary['emergency_events'].get('active_events', 0)}")
    print(f"Sensor fusion - Total fusions: {summary['sensor_fusion'].get('total_fusions', 0)}")


async def main():
    """Main demo function"""
    print("🚀 Guard AI - Voice-Activated Personal Safety Assistant")
    print("=" * 60)
    
    # Initialize Guard AI
    print("Initializing Guard AI...")
    guard_ai = GuardAI(user_id="demo_user")
    
    # Add emergency contacts
    guard_ai.add_emergency_contact("Demo Contact", "+1234567890", "demo@example.com")
    
    # Start the system
    print("Starting Guard AI system...")
    
    # Run demos
    try:
        await demo_voice_commands(guard_ai)
        await demo_location_tracking(guard_ai)
        await demo_facial_recognition(guard_ai)
        await demo_sensor_fusion(guard_ai)
        await demo_emergency_response(guard_ai)
        await demo_system_status(guard_ai)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
    finally:
        # Stop the system
        print("\nStopping Guard AI system...")
        await guard_ai.stop()
        print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main()) 