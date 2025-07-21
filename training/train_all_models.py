"""
Training Script for Guard AI Models
Trains all AI models using synthetic and real-world data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
import os
import json
from datetime import datetime, timedelta
import random

# Import Guard AI modules
from src.nlp.intent_classifier import IntentClassifier
from src.behavioral.pattern_learner import LSTMPatternLearner
from src.facial_recognition.face_detector import MobileFaceNet
from src.emergency.response_system import PolicyNetwork, ValueNetwork
from src.sensor_fusion.fusion_engine import SensorFusionTransformer
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)


class GuardAITrainer:
    """
    Comprehensive trainer for all Guard AI models
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Setup device
        self.device = torch.device(self.config_manager.get_device())
        
        # Create models directory
        os.makedirs(self.config.models_dir, exist_ok=True)
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.validation_split = 0.2
        
        logger.info(f"Guard AI Trainer initialized on device: {self.device}")
        
    def generate_synthetic_data(self) -> Dict[str, Any]:
        """Generate synthetic training data"""
        logger.info("Generating synthetic training data...")
        
        data = {
            'nlp': self._generate_nlp_data(),
            'behavioral': self._generate_behavioral_data(),
            'facial': self._generate_facial_data(),
            'emergency': self._generate_emergency_data(),
            'sensor_fusion': self._generate_sensor_fusion_data()
        }
        
        logger.info("Synthetic data generation completed")
        return data
        
    def _generate_nlp_data(self) -> Dict[str, Any]:
        """Generate NLP training data"""
        # Intent examples for Guard AI
        intents = {
            'location_tracking': [
                "Guard, track my location",
                "Follow me",
                "Monitor where I am",
                "Keep track of my position",
                "Watch my location"
            ],
            'emergency_alert': [
                "Help me",
                "Emergency",
                "Call for help",
                "Alert authorities",
                "I'm in danger",
                "SOS",
                "Need help now"
            ],
            'check_in': [
                "Check in",
                "I'm safe",
                "Mark me as safe",
                "Update my status",
                "All good"
            ],
            'facial_recognition': [
                "Scan faces",
                "Recognize people",
                "Identify who's around",
                "Check for known faces",
                "Who's nearby"
            ],
            'route_monitoring': [
                "Monitor my route",
                "Watch my path",
                "Track my journey",
                "Follow my route"
            ],
            'contact_update': [
                "Update contacts",
                "Add emergency contact",
                "Change trusted contacts",
                "Modify contact list"
            ],
            'safety_check': [
                "How safe is this area",
                "Assess safety",
                "Check surroundings",
                "Evaluate risk"
            ],
            'voice_command': [
                "Listen to me",
                "Voice activation",
                "Start voice mode",
                "Enable voice commands"
            ],
            'system_status': [
                "System status",
                "How are you working",
                "Check system",
                "Status report"
            ]
        }
        
        # Generate training data
        texts = []
        labels = []
        
        for intent, examples in intents.items():
            for example in examples:
                texts.append(example)
                labels.append(list(intents.keys()).index(intent))
                
                # Add variations
                for _ in range(5):
                    variation = self._add_noise_to_text(example)
                    texts.append(variation)
                    labels.append(list(intents.keys()).index(intent))
                    
        return {
            'texts': texts,
            'labels': labels,
            'num_classes': len(intents)
        }
        
    def _generate_behavioral_data(self) -> Dict[str, Any]:
        """Generate behavioral training data"""
        # Generate 30 days of synthetic behavioral data
        num_days = 30
        data_points_per_day = 24  # Hourly data points
        
        timestamps = []
        locations = []
        activities = []
        anomaly_labels = []
        
        for day in range(num_days):
            for hour in range(data_points_per_day):
                timestamp = datetime.now() - timedelta(days=num_days-day, hours=hour)
                timestamps.append(timestamp)
                
                # Generate normal location patterns
                if 6 <= hour <= 22:  # Daytime
                    lat = 37.7749 + random.uniform(-0.01, 0.01)  # San Francisco area
                    lon = -122.4194 + random.uniform(-0.01, 0.01)
                    activity = random.choice(['walking', 'stationary', 'driving'])
                else:  # Nighttime
                    lat = 37.7749 + random.uniform(-0.005, 0.005)  # Home area
                    lon = -122.4194 + random.uniform(-0.005, 0.005)
                    activity = 'stationary'
                    
                locations.append([lat, lon])
                activities.append(activity)
                
                # Generate anomaly labels (mostly normal, some anomalies)
                is_anomaly = random.random() < 0.05  # 5% anomaly rate
                anomaly_labels.append(1.0 if is_anomaly else 0.0)
                
        return {
            'timestamps': timestamps,
            'locations': locations,
            'activities': activities,
            'anomaly_labels': anomaly_labels
        }
        
    def _generate_facial_data(self) -> Dict[str, Any]:
        """Generate facial recognition training data"""
        # Generate synthetic face embeddings
        num_people = 50
        embeddings_per_person = 10
        
        face_embeddings = []
        person_labels = []
        
        for person_id in range(num_people):
            # Generate base embedding for person
            base_embedding = np.random.randn(128)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            
            for _ in range(embeddings_per_person):
                # Add noise to base embedding
                noise = np.random.randn(128) * 0.1
                embedding = base_embedding + noise
                embedding = embedding / np.linalg.norm(embedding)
                
                face_embeddings.append(embedding)
                person_labels.append(person_id)
                
        return {
            'embeddings': face_embeddings,
            'labels': person_labels,
            'num_people': num_people
        }
        
    def _generate_emergency_data(self) -> Dict[str, Any]:
        """Generate emergency response training data"""
        # Generate state-action pairs for reinforcement learning
        num_episodes = 1000
        states = []
        actions = []
        rewards = []
        
        for episode in range(num_episodes):
            # Generate random state
            state = np.random.rand(15)  # 15-dimensional state
            
            # Generate random action
            action = random.randint(0, 6)  # 7 possible actions
            
            # Generate reward based on state and action
            if state[0] > 0.8:  # High threat
                reward = -10 if action < 3 else 5  # Prefer strong actions
            elif state[0] > 0.5:  # Medium threat
                reward = -5 if action < 2 else 2  # Prefer moderate actions
            else:  # Low threat
                reward = -2 if action > 4 else 1  # Prefer light actions
                
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }
        
    def _generate_sensor_fusion_data(self) -> Dict[str, Any]:
        """Generate sensor fusion training data"""
        # Generate multimodal sensor data
        num_samples = 1000
        
        gps_data = []
        accelerometer_data = []
        microphone_data = []
        camera_data = []
        threat_labels = []
        
        for _ in range(num_samples):
            # Generate GPS data
            lat = random.uniform(37.7, 37.8)
            lon = random.uniform(-122.5, -122.4)
            speed = random.uniform(0, 30)
            heading = random.uniform(0, 360)
            gps_data.append([lat, lon, speed, heading])
            
            # Generate accelerometer data
            accel_x = random.uniform(-20, 20)
            accel_y = random.uniform(-20, 20)
            accel_z = random.uniform(-20, 20)
            accelerometer_data.append([accel_x, accel_y, accel_z])
            
            # Generate microphone data (MFCC features)
            mfcc = np.random.randn(13)
            microphone_data.append(mfcc)
            
            # Generate camera data (face features)
            face_features = np.random.randn(128)
            camera_data.append(face_features)
            
            # Generate threat label based on sensor data
            threat_score = 0.0
            
            # High speed = higher threat
            if speed > 25:
                threat_score += 0.3
                
            # High acceleration = higher threat
            accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            if accel_magnitude > 15:
                threat_score += 0.2
                
            # High audio energy = higher threat
            audio_energy = np.mean(mfcc**2)
            if audio_energy > 0.1:
                threat_score += 0.2
                
            # Random noise
            threat_score += random.uniform(-0.1, 0.1)
            threat_score = max(0.0, min(1.0, threat_score))
            
            threat_labels.append(threat_score)
            
        return {
            'gps_data': gps_data,
            'accelerometer_data': accelerometer_data,
            'microphone_data': microphone_data,
            'camera_data': camera_data,
            'threat_labels': threat_labels
        }
        
    def _add_noise_to_text(self, text: str) -> str:
        """Add noise to text for data augmentation"""
        words = text.split()
        
        # Random word replacement
        if random.random() < 0.3:
            synonyms = {
                'track': ['follow', 'monitor', 'watch'],
                'location': ['position', 'whereabouts', 'place'],
                'help': ['assist', 'aid', 'support'],
                'emergency': ['urgent', 'critical', 'danger'],
                'safe': ['secure', 'protected', 'okay']
            }
            
            for i, word in enumerate(words):
                if word.lower() in synonyms:
                    words[i] = random.choice(synonyms[word.lower()])
                    
        # Random word insertion
        if random.random() < 0.2:
            filler_words = ['please', 'now', 'quickly', 'right']
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, random.choice(filler_words))
            
        return ' '.join(words)
        
    def train_nlp_model(self, data: Dict[str, Any]):
        """Train NLP intent classification model"""
        logger.info("Training NLP model...")
        
        try:
            # Prepare data
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Tokenize texts
            encodings = tokenizer(
                data['texts'],
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            
            labels = torch.tensor(data['labels'])
            
            # Split data
            num_train = int(len(data['texts']) * (1 - self.validation_split))
            
            train_encodings = {
                'input_ids': encodings['input_ids'][:num_train],
                'attention_mask': encodings['attention_mask'][:num_train]
            }
            train_labels = labels[:num_train]
            
            val_encodings = {
                'input_ids': encodings['input_ids'][num_train:],
                'attention_mask': encodings['attention_mask'][num_train:]
            }
            val_labels = labels[num_train:]
            
            # Create model
            model = IntentClassifier(num_intents=data['num_classes'])
            model.to(self.device)
            
            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(self.num_epochs):
                model.train()
                optimizer.zero_grad()
                
                outputs = model(
                    train_encodings['input_ids'].to(self.device),
                    train_encodings['attention_mask'].to(self.device)
                )
                
                loss = criterion(outputs, train_labels.to(self.device))
                loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(
                            val_encodings['input_ids'].to(self.device),
                            val_encodings['attention_mask'].to(self.device)
                        )
                        val_loss = criterion(val_outputs, val_labels.to(self.device))
                        
                    logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                    
            # Save model
            model_path = f"{self.config.models_dir}/nlp_model.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"NLP model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error training NLP model: {e}")
            
    def train_behavioral_model(self, data: Dict[str, Any]):
        """Train behavioral pattern learning model"""
        logger.info("Training behavioral model...")
        
        try:
            # Prepare data
            locations = np.array(data['locations'])
            activities = data['activities']
            labels = np.array(data['anomaly_labels'])
            
            # Encode activities
            activity_encoding = {
                'walking': [1, 0, 0, 0, 0],
                'running': [0, 1, 0, 0, 0],
                'driving': [0, 0, 1, 0, 0],
                'stationary': [0, 0, 0, 1, 0],
                'cycling': [0, 0, 0, 0, 1]
            }
            
            # Create feature sequences
            sequence_length = 24  # 24 hours
            features = []
            targets = []
            
            for i in range(sequence_length, len(locations)):
                sequence_features = []
                
                for j in range(sequence_length):
                    idx = i - sequence_length + j
                    lat, lon = locations[idx]
                    activity = activities[idx]
                    
                    # Normalize features
                    hour = data['timestamps'][idx].hour / 24.0
                    day_of_week = data['timestamps'][idx].weekday() / 7.0
                    
                    feature = [
                        lat, lon,  # Location
                        hour, day_of_week,  # Time
                        *activity_encoding.get(activity, [0, 0, 0, 0, 0])  # Activity
                    ]
                    
                    sequence_features.append(feature)
                    
                features.append(sequence_features)
                targets.append(labels[i])
                
            # Convert to tensors
            features = torch.tensor(features, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
            
            # Split data
            num_train = int(len(features) * (1 - self.validation_split))
            
            train_features = features[:num_train]
            train_targets = targets[:num_train]
            val_features = features[num_train:]
            val_targets = targets[num_train:]
            
            # Create model
            model = LSTMPatternLearner(input_size=9)  # lat, lon, hour, day, 5 activities
            model.to(self.device)
            
            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.BCELoss()
            
            # Training loop
            for epoch in range(self.num_epochs):
                model.train()
                optimizer.zero_grad()
                
                outputs, _ = model(train_features.to(self.device))
                loss = criterion(outputs, train_targets.to(self.device))
                loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs, _ = model(val_features.to(self.device))
                        val_loss = criterion(val_outputs, val_targets.to(self.device))
                        
                    logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                    
            # Save model
            model_path = f"{self.config.models_dir}/behavioral_model.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Behavioral model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error training behavioral model: {e}")
            
    def train_facial_model(self, data: Dict[str, Any]):
        """Train facial recognition model"""
        logger.info("Training facial recognition model...")
        
        try:
            # Prepare data
            embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
            labels = torch.tensor(data['labels'], dtype=torch.long)
            
            # Split data
            num_train = int(len(embeddings) * (1 - self.validation_split))
            
            train_embeddings = embeddings[:num_train]
            train_labels = labels[:num_train]
            val_embeddings = embeddings[num_train:]
            val_labels = labels[num_train:]
            
            # Create model
            model = MobileFaceNet(embedding_size=128, num_classes=data['num_people'])
            model.to(self.device)
            
            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(self.num_epochs):
                model.train()
                optimizer.zero_grad()
                
                outputs = model(train_embeddings.to(self.device), return_embedding=False)
                loss = criterion(outputs, train_labels.to(self.device))
                loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(val_embeddings.to(self.device), return_embedding=False)
                        val_loss = criterion(val_outputs, val_labels.to(self.device))
                        
                    logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                    
            # Save model
            model_path = f"{self.config.models_dir}/face_model.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Facial recognition model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error training facial recognition model: {e}")
            
    def train_emergency_model(self, data: Dict[str, Any]):
        """Train emergency response model"""
        logger.info("Training emergency response model...")
        
        try:
            # Prepare data
            states = torch.tensor(data['states'], dtype=torch.float32)
            actions = torch.tensor(data['actions'], dtype=torch.long)
            rewards = torch.tensor(data['rewards'], dtype=torch.float32)
            
            # Split data
            num_train = int(len(states) * (1 - self.validation_split))
            
            train_states = states[:num_train]
            train_actions = actions[:num_train]
            train_rewards = rewards[:num_train]
            val_states = states[num_train:]
            val_actions = actions[num_train:]
            val_rewards = rewards[num_train:]
            
            # Create models
            policy_network = PolicyNetwork(state_size=15, action_size=7)
            value_network = ValueNetwork(state_size=15)
            
            policy_network.to(self.device)
            value_network.to(self.device)
            
            # Training setup
            policy_optimizer = optim.Adam(policy_network.parameters(), lr=self.learning_rate)
            value_optimizer = optim.Adam(value_network.parameters(), lr=self.learning_rate)
            
            # Training loop
            for epoch in range(self.num_epochs):
                policy_network.train()
                value_network.train()
                
                # Policy training
                policy_optimizer.zero_grad()
                action_probs = policy_network(train_states.to(self.device))
                log_probs = torch.log(action_probs + 1e-8)
                selected_log_probs = log_probs.gather(1, train_actions.unsqueeze(1).to(self.device))
                policy_loss = -(selected_log_probs * train_rewards.unsqueeze(1).to(self.device)).mean()
                policy_loss.backward()
                policy_optimizer.step()
                
                # Value training
                value_optimizer.zero_grad()
                values = value_network(train_states.to(self.device))
                value_loss = nn.MSELoss()(values.squeeze(), train_rewards.to(self.device))
                value_loss.backward()
                value_optimizer.step()
                
                # Validation
                if epoch % 10 == 0:
                    policy_network.eval()
                    value_network.eval()
                    with torch.no_grad():
                        val_action_probs = policy_network(val_states.to(self.device))
                        val_values = value_network(val_states.to(self.device))
                        val_policy_loss = -(torch.log(val_action_probs + 1e-8).gather(1, val_actions.unsqueeze(1).to(self.device)) * val_rewards.unsqueeze(1).to(self.device)).mean()
                        val_value_loss = nn.MSELoss()(val_values.squeeze(), val_rewards.to(self.device))
                        
                    logger.info(f"Epoch {epoch}: Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
                    
            # Save models
            checkpoint = {
                'policy': policy_network.state_dict(),
                'value': value_network.state_dict()
            }
            model_path = f"{self.config.models_dir}/emergency_model.pth"
            torch.save(checkpoint, model_path)
            logger.info(f"Emergency response model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error training emergency response model: {e}")
            
    def train_sensor_fusion_model(self, data: Dict[str, Any]):
        """Train sensor fusion model"""
        logger.info("Training sensor fusion model...")
        
        try:
            from src.sensor_fusion.fusion_engine import SensorType
            
            # Prepare data
            gps_data = torch.tensor(data['gps_data'], dtype=torch.float32)
            accelerometer_data = torch.tensor(data['accelerometer_data'], dtype=torch.float32)
            microphone_data = torch.tensor(data['microphone_data'], dtype=torch.float32)
            camera_data = torch.tensor(data['camera_data'], dtype=torch.float32)
            threat_labels = torch.tensor(data['threat_labels'], dtype=torch.float32).unsqueeze(1)
            
            # Split data
            num_train = int(len(gps_data) * (1 - self.validation_split))
            
            train_gps = gps_data[:num_train]
            train_accel = accelerometer_data[:num_train]
            train_mic = microphone_data[:num_train]
            train_camera = camera_data[:num_train]
            train_threat = threat_labels[:num_train]
            
            val_gps = gps_data[num_train:]
            val_accel = accelerometer_data[num_train:]
            val_mic = microphone_data[num_train:]
            val_camera = camera_data[num_train:]
            val_threat = threat_labels[num_train:]
            
            # Create model
            sensor_dims = {
                SensorType.GPS: 4,
                SensorType.ACCELEROMETER: 3,
                SensorType.MICROPHONE: 13,
                SensorType.CAMERA: 128
            }
            
            model = SensorFusionTransformer(sensor_dims)
            model.to(self.device)
            
            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.BCELoss()
            
            # Training loop
            for epoch in range(self.num_epochs):
                model.train()
                optimizer.zero_grad()
                
                # Prepare sensor data
                sensor_data = {
                    SensorType.GPS: train_gps.to(self.device),
                    SensorType.ACCELEROMETER: train_accel.to(self.device),
                    SensorType.MICROPHONE: train_mic.to(self.device),
                    SensorType.CAMERA: train_camera.to(self.device)
                }
                
                threat_score, confidence = model(sensor_data)
                loss = criterion(threat_score, train_threat.to(self.device))
                loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_sensor_data = {
                            SensorType.GPS: val_gps.to(self.device),
                            SensorType.ACCELEROMETER: val_accel.to(self.device),
                            SensorType.MICROPHONE: val_mic.to(self.device),
                            SensorType.CAMERA: val_camera.to(self.device)
                        }
                        
                        val_threat_score, val_confidence = model(val_sensor_data)
                        val_loss = criterion(val_threat_score, val_threat.to(self.device))
                        
                    logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                    
            # Save model
            model_path = f"{self.config.models_dir}/fusion_model.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Sensor fusion model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error training sensor fusion model: {e}")
            
    def train_all_models(self):
        """Train all Guard AI models"""
        logger.info("Starting training of all Guard AI models...")
        
        try:
            # Generate synthetic data
            data = self.generate_synthetic_data()
            
            # Train each model
            self.train_nlp_model(data['nlp'])
            self.train_behavioral_model(data['behavioral'])
            self.train_facial_model(data['facial'])
            self.train_emergency_model(data['emergency'])
            self.train_sensor_fusion_model(data['sensor_fusion'])
            
            logger.info("All models trained successfully!")
            
            # Save training metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'device': str(self.device),
                'models_trained': [
                    'nlp_model.pth',
                    'behavioral_model.pth',
                    'face_model.pth',
                    'emergency_model.pth',
                    'fusion_model.pth'
                ]
            }
            
            metadata_path = f"{self.config.models_dir}/training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Training metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error in training: {e}")


def main():
    """Main training function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize trainer
    trainer = GuardAITrainer()
    
    # Train all models
    trainer.train_all_models()


if __name__ == "__main__":
    main() 