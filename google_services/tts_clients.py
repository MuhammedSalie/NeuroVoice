import streamlit as st
import base64
import os
from google.cloud import texttospeech

class GoogleTTSClient:
    def __init__(self):
        try:
            # Initialize Google Text-to-Speech client
            self.client = texttospeech.TextToSpeechClient()
            self.voices = {
                'medical_male': 'en-US-Wavenet-D',
                'medical_female': 'en-US-Wavenet-F',
                'calm_male': 'en-US-Standard-B',
                'calm_female': 'en-US-Standard-C'
            }
            st.success("✅ Google Text-to-Speech Connected")
        except Exception as e:
            st.error(f"❌ Google TTS failed: {e}")
            self.client = None
    
    def synthesize_speech(self, text, voice_type='medical_female', emergency_level='medium'):
        """Convert text to speech using Google TTS"""
        if self.client is None:
            return None
            
        try:
            # Adjust speaking rate based on emergency level
            speaking_rates = {
                'high': 1.2,    # Faster for emergencies
                'medium': 1.0,  # Normal speed
                'low': 0.8      # Calmer for reassurance
            }
            
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=self.voices[voice_type],
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rates[emergency_level],
                pitch=0.0,  # Neutral pitch for medical context
                volume_gain_db=6.0  # Slightly louder for alerts
            )
            
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            return base64.b64encode(response.audio_content).decode('utf-8')
            
        except Exception as e:
            st.error(f"❌ TTS synthesis failed: {e}")
            return None
    
    def text_to_speech_base64(self, text):
        """Convert text to base64 audio for Streamlit"""
        audio_content = self.synthesize_speech(text)
        if audio_content:
            return f"data:audio/mp3;base64,{audio_content}"
        return None