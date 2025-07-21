#!/usr/bin/env python3
"""
Voice Recognition Test for Guard AI
Tests wake word detection and voice command processing
"""

import asyncio
import numpy as np
import time
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp.intent_classifier import GuardIntentClassifier
from src.nlp.voice_processor import VoiceProcessor


class VoiceRecognitionTest:
    """
    Test voice recognition capabilities of Guard AI
    """
    
    def __init__(self):
        """Initialize voice recognition test"""
        print("🎤 Initializing Voice Recognition Test...")
        
        # Initialize voice processing components
        self.voice_processor = VoiceProcessor()
        self.intent_classifier = GuardIntentClassifier()
        
        print("✅ Voice recognition components initialized!")
        
    def test_wake_word_detection(self):
        """Test wake word detection capabilities"""
        print("\n🔊 Testing Wake Word Detection...")
        
        # Common wake words and phrases
        wake_words = [
            "Guard AI",
            "Hey Guard",
            "Guard Assistant",
            "Safety Guard",
            "Guard AI, wake up",
            "Hey Guard AI",
            "Guard AI, are you there?",
            "Safety mode activate",
            "Guard AI, I need help",
            "Emergency Guard"
        ]
        
        # Non-wake words (should not trigger)
        non_wake_words = [
            "Hello there",
            "What time is it?",
            "How's the weather?",
            "Play some music",
            "Set a timer",
            "Call mom",
            "Open the door",
            "Turn on the lights",
            "What's for dinner?",
            "Good morning"
        ]
        
        print("🔍 Testing wake word phrases:")
        for phrase in wake_words:
            try:
                # Check if it contains wake word patterns
                is_wake_word = self._check_wake_word_pattern(phrase)
                intent_result = self.intent_classifier.classify_intent(phrase)
                
                print(f"   '{phrase}' -> Wake word: {is_wake_word}, Intent: {intent_result['intent']}")
                
            except Exception as e:
                print(f"   '{phrase}' -> Error: {e}")
        
        print("\n🚫 Testing non-wake word phrases:")
        for phrase in non_wake_words:
            try:
                is_wake_word = self._check_wake_word_pattern(phrase)
                intent_result = self.intent_classifier.classify_intent(phrase)
                
                print(f"   '{phrase}' -> Wake word: {is_wake_word}, Intent: {intent_result['intent']}")
                
            except Exception as e:
                print(f"   '{phrase}' -> Error: {e}")
    
    def _check_wake_word_pattern(self, phrase: str) -> bool:
        """Check if phrase contains wake word patterns"""
        phrase_lower = phrase.lower()
        
        # Common wake word patterns
        wake_patterns = [
            "guard ai",
            "hey guard",
            "guard assistant",
            "safety guard",
            "emergency guard",
            "guard, wake up",
            "guard, activate"
        ]
        
        return any(pattern in phrase_lower for pattern in wake_patterns)
    
    def test_voice_commands(self):
        """Test various voice commands"""
        print("\n🎯 Testing Voice Commands...")
        
        # Test commands by category
        command_categories = {
            "Emergency Commands": [
                "Guard AI, I need help",
                "Emergency mode activate",
                "Call emergency services",
                "Send SOS",
                "I'm in danger",
                "Help me",
                "Activate emergency protocol"
            ],
            "Location Commands": [
                "Where am I?",
                "What's my current location?",
                "Show me on the map",
                "Track my location",
                "Where am I right now?",
                "Get my GPS coordinates"
            ],
            "Safety Commands": [
                "Is it safe here?",
                "Check the area",
                "Scan for threats",
                "Assess safety",
                "Is this area dangerous?",
                "Safety check"
            ],
            "Contact Commands": [
                "Add emergency contact",
                "Call my mom",
                "Contact John",
                "Add Sarah as emergency contact",
                "Call 911",
                "Alert my contacts"
            ],
            "System Commands": [
                "Guard AI status",
                "System health check",
                "What's your status?",
                "Are you working?",
                "System information",
                "Guard AI, report"
            ]
        }
        
        for category, commands in command_categories.items():
            print(f"\n📋 {category}:")
            for command in commands:
                try:
                    intent_result = self.intent_classifier.classify_intent(command)
                    response = self.intent_classifier.generate_response(
                        intent_result['intent'],
                        intent_result.get('entities', {}),
                        self.intent_classifier.get_context()
                    )
                    
                    print(f"   '{command}'")
                    print(f"      Intent: {intent_result['intent']}")
                    print(f"      Confidence: {intent_result['confidence']:.2f}")
                    print(f"      Response: {response[:100]}...")
                    
                except Exception as e:
                    print(f"   '{command}' -> Error: {e}")
    
    def test_audio_processing(self):
        """Test audio processing capabilities"""
        print("\n🎵 Testing Audio Processing...")
        
        # Create dummy audio data (simulating microphone input)
        sample_rate = 16000
        duration = 2  # 2 seconds
        num_samples = sample_rate * duration
        
        # Generate different types of audio
        audio_types = {
            "Silence": np.zeros(num_samples, dtype=np.float32),
            "White Noise": np.random.normal(0, 0.1, num_samples).astype(np.float32),
            "Sine Wave": np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples)).astype(np.float32),
            "Mixed Audio": (np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples)) + 
                          0.5 * np.sin(2 * np.pi * 880 * np.linspace(0, duration, num_samples))).astype(np.float32)
        }
        
        for audio_type, audio_data in audio_types.items():
            print(f"\n🔊 Testing {audio_type}:")
            try:
                # Test speech-to-text (will likely fail with dummy data, but tests the pipeline)
                transcription = self.voice_processor.speech_to_text(audio_data, sample_rate)
                print(f"   Transcription: {transcription if transcription else 'No speech detected'}")
                
                # Test emergency voice detection
                is_emergency = self.voice_processor.is_emergency_voice(audio_data, sample_rate)
                print(f"   Emergency voice detected: {is_emergency}")
                
                # Test voice activity detection
                has_voice = self.voice_processor.detect_voice_activity(audio_data, sample_rate)
                print(f"   Voice activity detected: {has_voice}")
                
            except Exception as e:
                print(f"   Error: {e}")
    
    def test_intent_recognition_accuracy(self):
        """Test intent recognition accuracy with various inputs"""
        print("\n🎯 Testing Intent Recognition Accuracy...")
        
        # Test cases with expected intents
        test_cases = [
            ("Guard AI, I need help", "emergency"),
            ("Where am I?", "location"),
            ("Add John as emergency contact", "contact"),
            ("Is it safe here?", "safety"),
            ("What's your status?", "system"),
            ("Call 911", "emergency"),
            ("Show me the map", "location"),
            ("Check the area", "safety"),
            ("Add Sarah", "contact"),
            ("System health", "system")
        ]
        
        correct = 0
        total = len(test_cases)
        
        for command, expected_intent in test_cases:
            try:
                intent_result = self.intent_classifier.classify_intent(command)
                predicted_intent = intent_result['intent']
                confidence = intent_result['confidence']
                
                is_correct = predicted_intent == expected_intent
                if is_correct:
                    correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"   {status} '{command}' -> {predicted_intent} (expected: {expected_intent}) [{confidence:.2f}]")
                
            except Exception as e:
                print(f"   ❌ '{command}' -> Error: {e}")
        
        accuracy = (correct / total) * 100
        print(f"\n📊 Intent Recognition Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    def run_comprehensive_test(self):
        """Run all voice recognition tests"""
        print("="*60)
        print("🎤 GUARD AI VOICE RECOGNITION COMPREHENSIVE TEST")
        print("="*60)
        
        # Run all tests
        self.test_wake_word_detection()
        self.test_voice_commands()
        self.test_audio_processing()
        self.test_intent_recognition_accuracy()
        
        print("\n" + "="*60)
        print("✅ Voice Recognition Test Completed!")
        print("="*60)
        
        # Summary
        print("\n📋 Test Summary:")
        print("   • Wake word detection patterns checked")
        print("   • Voice command processing tested")
        print("   • Audio processing pipeline verified")
        print("   • Intent recognition accuracy measured")
        print("   • Response generation tested")
        
        print("\n🔧 Recommendations:")
        print("   • For real testing, use actual microphone input")
        print("   • Train models with real voice data for better accuracy")
        print("   • Fine-tune wake word detection thresholds")
        print("   • Add more domain-specific voice commands")
        print("   • Implement noise cancellation for better recognition")


async def main():
    """Main test function"""
    try:
        # Initialize and run tests
        voice_test = VoiceRecognitionTest()
        voice_test.run_comprehensive_test()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 