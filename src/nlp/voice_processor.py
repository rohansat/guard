"""
Voice Processing Module for Guard AI
Handles speech-to-text conversion and voice command preprocessing
"""

import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VoiceProcessor:
    """
    Handles voice processing including speech-to-text and voice command preprocessing
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """
        Initialize the voice processor
        
        Args:
            model_name: Pre-trained model name for speech recognition
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        
        # Voice activity detection parameters
        self.vad_threshold = 0.5
        self.min_audio_length = 0.5  # seconds
        self.max_audio_length = 30.0  # seconds
        
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio data for speech recognition
        
        Args:
            audio_data: Raw audio data
            sample_rate: Audio sample rate
            
        Returns:
            Preprocessed audio data
        """
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        # Apply noise reduction (simple high-pass filter)
        audio_data = librosa.effects.preemphasis(audio_data, coef=0.97)
        
        return audio_data
    
    def detect_voice_activity(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """
        Detect if there's voice activity in the audio
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            True if voice activity detected
        """
        # Calculate energy
        energy = np.mean(audio_data ** 2)
        
        # Calculate zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Simple voice activity detection
        is_voice = energy > self.vad_threshold and zcr < 0.1
        
        return is_voice
    
    def speech_to_text(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Convert speech to text using Wav2Vec2
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Transcribed text
        """
        try:
            # Preprocess audio
            audio_data = self.preprocess_audio(audio_data, sample_rate)
            
            # Check audio length
            duration = len(audio_data) / 16000
            if duration < self.min_audio_length:
                return ""
            if duration > self.max_audio_length:
                audio_data = audio_data[:int(self.max_audio_length * 16000)]
            
            # Convert to tensor
            inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt")
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Get predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error in speech to text: {e}")
            return ""
    
    def extract_voice_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract voice features for emotion and speaker identification
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary of voice features
        """
        features = {}
        
        try:
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            features['mfcc'] = np.mean(mfcc, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            features['spectral_centroid'] = np.mean(spectral_centroids)
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            features['pitch'] = np.mean(pitches[magnitudes > 0.1])
            
            # Energy features
            features['energy'] = np.mean(audio_data ** 2)
            
            # Duration
            features['duration'] = len(audio_data) / sample_rate
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {e}")
            
        return features
    
    def is_emergency_voice(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """
        Detect if the voice indicates an emergency (screaming, distress, etc.)
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            True if emergency voice detected
        """
        try:
            features = self.extract_voice_features(audio_data, sample_rate)
            
            # Simple emergency detection based on energy and pitch
            high_energy = features.get('energy', 0) > 0.1
            high_pitch = features.get('pitch', 0) > 200  # Hz
            
            return high_energy and high_pitch
            
        except Exception as e:
            logger.error(f"Error in emergency voice detection: {e}")
            return False 