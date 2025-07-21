# Guard AI - Voice-Activated Personal Safety Assistant

Guard AI is a comprehensive personal safety system that combines multiple AI models to provide real-time protection and monitoring. The system uses voice commands, location tracking, behavioral analysis, and facial recognition to ensure user safety.

## 🚀 Features

### Core Capabilities
- **Voice Command Processing**: Natural language understanding for hands-free operation
- **Behavioral Pattern Learning**: LSTM-based anomaly detection for user behavior
- **Facial Recognition**: Mobile-optimized CNN for identifying known/unknown individuals
- **Multimodal Sensor Fusion**: Transformer-based fusion of GPS, accelerometer, microphone, and camera data
- **Emergency Response System**: Reinforcement learning-based decision making for threat response
- **Real-time Monitoring**: Continuous safety assessment and threat detection

### Safety Features
- **Location Tracking**: Monitor user location and detect unusual patterns
- **Activity Recognition**: Classify user activities (walking, running, driving, stationary)
- **Proximity Awareness**: Detect and identify people in the user's vicinity
- **Emergency Alerts**: Automatic notification of trusted contacts and emergency services
- **Voice Distress Detection**: Identify emergency situations through voice analysis
- **SOS Activation**: Immediate emergency protocols when critical threats are detected

## 🏗️ Architecture Overview

The Guard AI system consists of several specialized AI models working together:

### 1. Natural Language Processing (NLP) Model
- **Purpose**: Voice command understanding and conversational AI
- **Features**: Intent recognition, context awareness, natural dialogue
- **Technology**: BERT-based transformer models with custom intent classification

### 2. Behavioral Pattern Learning Model
- **Purpose**: Learn user's normal behavior patterns and detect anomalies
- **Features**: Location tracking, activity classification, anomaly detection
- **Technology**: LSTM networks with attention mechanisms and self-attention

### 3. Facial Recognition Model
- **Purpose**: Identify known/unknown individuals and detect suspicious behavior
- **Features**: Face classification, proximity awareness, threat assessment
- **Technology**: MobileFaceNet CNN optimized for edge deployment

### 4. Multimodal Sensor Fusion Model
- **Purpose**: Combine data from multiple sensors for comprehensive threat assessment
- **Features**: GPS, accelerometer, microphone, camera data integration
- **Technology**: Transformer-based fusion models with multi-head attention

### 5. Emergency Response Decision Model
- **Purpose**: Make intelligent decisions about when and how to respond to threats
- **Features**: Risk assessment, escalation protocols, automated responses
- **Technology**: Reinforcement learning with policy and value networks

## 📁 Project Structure

```
guard_ai/
├── models/                 # Trained model files
├── data/                   # Training and test datasets
├── src/                    # Source code
│   ├── nlp/               # NLP model components
│   │   ├── intent_classifier.py
│   │   └── voice_processor.py
│   ├── behavioral/        # Behavioral analysis models
│   │   └── pattern_learner.py
│   ├── facial_recognition/ # Face recognition models
│   │   └── face_detector.py
│   ├── sensor_fusion/     # Multimodal fusion models
│   │   └── fusion_engine.py
│   ├── emergency/         # Emergency response models
│   │   └── response_system.py
│   └── utils/             # Utility functions
│       └── config.py
├── training/              # Training scripts
│   └── train_all_models.py
├── evaluation/            # Model evaluation tools
├── deployment/            # Deployment configurations
├── demo.py               # Interactive demo script
├── api_server.py         # FastAPI REST server
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd guard_ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

The system will automatically create a default configuration file (`config.yaml`) on first run. You can customize settings for each module:

```yaml
user_id: "your_user_id"
device: "auto"  # auto, cpu, cuda
log_level: "INFO"

nlp:
  model_name: "bert-base-uncased"
  confidence_threshold: 0.7

behavioral:
  anomaly_threshold: 0.7
  sequence_length: 24

facial_recognition:
  recognition_threshold: 0.7
  database_path: "face_database.pkl"

emergency:
  threat_thresholds:
    low: 0.3
    medium: 0.5
    high: 0.7
    critical: 0.9

sensor_fusion:
  fusion_interval: 1.0
  buffer_size: 100
```

### 3. Training Models

Train all AI models using synthetic data:

```bash
python training/train_all_models.py
```

This will train:
- NLP intent classification model
- Behavioral pattern learning model
- Facial recognition model
- Emergency response model
- Sensor fusion model

### 4. Running the System

#### Interactive Demo
```bash
python demo.py
```

#### API Server
```bash
python api_server.py
```
The API server will be available at `http://localhost:8000`

#### Direct Usage
```python
import asyncio
from src.main import GuardAI

async def main():
    # Initialize Guard AI
    guard_ai = GuardAI(user_id="your_user_id")
    
    # Add emergency contacts
    guard_ai.add_emergency_contact("Mom", "+1234567890", "mom@example.com")
    
    # Start the system
    await guard_ai.start()
    
    # Process voice command
    audio_data = b"..."  # Your audio data
    result = await guard_ai.process_voice_command(audio_data)
    
    # Add location data
    await guard_ai.add_location_data(37.7749, -122.4194)
    
    # Get safety summary
    summary = guard_ai.get_safety_summary()
    print(f"Threat level: {summary['overall_threat_level']}")

asyncio.run(main())
```

## 📡 API Endpoints

The FastAPI server provides the following endpoints:

- `GET /` - API status
- `GET /status` - System status
- `GET /safety-summary` - Comprehensive safety assessment
- `POST /voice-command` - Process voice commands
- `POST /location` - Add location data
- `POST /activity` - Add activity data
- `POST /camera` - Process camera frames
- `POST /sensors` - Add sensor data
- `POST /emergency-contacts` - Add emergency contacts
- `POST /faces` - Add known faces
- `GET /health` - Health check

## 🎯 Use Cases

### College Students
- Safe navigation around campus
- Emergency alerts during late-night activities
- Contact trusted friends/family in emergencies

### Solo Travelers
- Location tracking in unfamiliar areas
- Facial recognition for safety assessment
- Emergency response in foreign countries

### General Safety
- Daily routine monitoring
- Anomaly detection in behavior patterns
- Proactive threat assessment

## 🔧 Customization

### Adding New Intents
Extend the NLP model by adding new intents in `src/nlp/intent_classifier.py`:

```python
self.intents = {
    # ... existing intents ...
    10: "custom_intent"
}

self.intent_examples["custom_intent"] = [
    "custom command example",
    "another example"
]
```

### Custom Emergency Protocols
Modify emergency response protocols in `src/emergency/response_system.py`:

```python
self.response_protocols = {
    ThreatLevel.LOW: [ResponseAction.MONITOR],
    ThreatLevel.MEDIUM: [ResponseAction.MONITOR, ResponseAction.TRACK_LOCATION],
    ThreatLevel.HIGH: [ResponseAction.MONITOR, ResponseAction.TRACK_LOCATION, ResponseAction.ALERT_CONTACTS],
    ThreatLevel.CRITICAL: [ResponseAction.MONITOR, ResponseAction.TRACK_LOCATION, ResponseAction.ALERT_CONTACTS, ResponseAction.CALL_EMERGENCY]
}
```

### Sensor Integration
Add new sensors by extending the sensor fusion engine in `src/sensor_fusion/fusion_engine.py`:

```python
class SensorType(Enum):
    # ... existing sensors ...
    NEW_SENSOR = "new_sensor"
```

## 🔒 Privacy and Security

- **On-device Processing**: All sensitive data processing happens locally
- **Encrypted Storage**: User data is encrypted at rest
- **Consent-based**: Explicit user consent required for data collection
- **Minimal Retention**: Models designed to minimize data retention
- **Secure Communication**: End-to-end encryption for all communications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the demo scripts

## 🔮 Future Enhancements

- **Mobile App**: Native iOS/Android applications
- **Cloud Integration**: Optional cloud backup and sync
- **Advanced Analytics**: Detailed safety insights and reports
- **Integration APIs**: Connect with smart home and IoT devices
- **Multi-language Support**: Internationalization for global use
