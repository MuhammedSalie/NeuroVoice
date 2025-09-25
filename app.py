

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
import os
import random


# Import our custom modules
from utils.data_simulator import EEGDataSimulator
from utils.preprocessor import EEGPreprocessor
from utils.advanced_features import ImpactVisualizer, DemoManager

# Simple fallback for Google TTS
class DemoTextToSpeech:
    def synthesize_speech(self, text):
        # This is a demo version that doesn't actually generate audio
        return None

class NeuroVoiceApp:
    def __init__(self):
        self.simulator = EEGDataSimulator()
        self.preprocessor = EEGPreprocessor()
        self.words = ['YES', 'NO', 'HELP', 'PAIN', 'WATER', 'HELLO', 'THANK YOU']
        self.medical_mode_words = ['PAIN', 'HELP', 'WATER', 'YES', 'NO']
        self.load_model()
        self.setup_services()
        self.demo_manager = DemoManager()
        self.impact_viz = ImpactVisualizer()
        
    def load_model(self):
        """Load or create a simple model"""
        try:
            # Try to load a pre-trained model
            self.model = keras.models.load_model('models/eeg_model.h5')
            st.success("✅ EEG Model Loaded")
        except:
            # Create a simple demo model
            self.model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(6,)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(len(self.medical_mode_words), activation='softmax')
            ])
            # Compile the model
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            st.info("🔬 Using Demo Model")
    
    def setup_services(self):
        """Setup text-to-speech (demo version)"""
        try:
            # In a real implementation, you'd use Google TTS here
            self.tts_client = DemoTextToSpeech()
        except:
            self.tts_client = DemoTextToSpeech()
    
    def predict_with_confidence(self, eeg_data):
        """Predict word from EEG data"""
        try:
            filtered_data = self.preprocessor.apply_filters(eeg_data)
            features = self.preprocessor.extract_advanced_features(filtered_data)
            
            # Ensure we have the right number of features
            if len(features) > 6:
                features = features[:6]
            elif len(features) < 6:
                # Pad with zeros if needed
                features = np.pad(features, (0, 6 - len(features)))
            
            prediction = self.model.predict(features.reshape(1, -1), verbose=0)
            word_idx = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            return self.medical_mode_words[word_idx], confidence, prediction[0]
            
        except Exception as e:
            return "THINKING", 0.0, np.zeros(len(self.medical_mode_words))
    
    def text_to_speech(self, text):
        """Convert text to speech (demo version)"""
        if self.tts_client:
            return self.tts_client.synthesize_speech(text)
        return None

def main():
    # Configure the page
    st.set_page_config(
        page_title="NeuroVoice - Brain-to-Speech Interface",
        page_icon="🧠",
        layout="wide"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    /* Main header */
    .main-header {
        font-size: 3rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .demo-container {
        border: 2px solid #4ECDC4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    /* Main background */
    body, .stApp {
        background-color: #23272b !important; /* grayish-black */
        color: #fff !important;
    }
    /* Sidebar (flyout menu) */
    section[data-testid="stSidebar"], .css-1d391kg, .stSidebar {
        background-color: #1976d2 !important; /* professional medical blue */
        color: #fff !important;
    }
    /* Tertiary color: white for cards, containers, etc. */
    .stMetric, .stButton, .stTabs, .stTabs [data-baseweb="tab"] {
        background-color: #fff !important;
        color: #23272b !important;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<h1 class="main-header">🧠 NeuroVoice</h1>', unsafe_allow_html=True)
    st.markdown("### <center>EEG-to-Speech Interface for Locked-In Syndrome Patients</center>", unsafe_allow_html=True)
    
    # Initialize app state

    if 'app' not in st.session_state:
        st.session_state.app = NeuroVoiceApp()
        # Note: The emotions.csv file contains text data, not EEG signals
        # We'll use simulated EEG data instead for the demo
        data_path = os.path.join('data', 'emotions.csv')
        if os.path.exists(data_path):
            try:
                # Since emotions.csv contains text data (not EEG), we'll generate synthetic EEG data
                # This is more appropriate for the brain-to-speech interface demo
                st.session_state.eeg_data = st.session_state.app.simulator.generate_medical_eeg_data(30, 'HELP')
                st.info("💡 Using simulated EEG data optimized for medical brain-speech interface")
            except Exception as e:
                st.warning(f"Using fallback simulated data: {e}")
                st.session_state.eeg_data = st.session_state.app.simulator.generate_medical_eeg_data(30, 'HELP')
        else:
            st.session_state.eeg_data = st.session_state.app.simulator.generate_medical_eeg_data(30, 'HELP')
        st.session_state.predictions = []
        st.session_state.impact_metrics = {
            'patients_helped': 3,
            'communication_time_saved': 27,
            'successful_communications': 42
        }
        st.session_state.current_pos = 0
    
    app = st.session_state.app
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Demo Controls")
        
        # Impact metrics
        st.subheader("📊 Live Impact")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Patients Helped", st.session_state.impact_metrics['patients_helped'])
        with col2:
            st.metric("Time Saved (hrs)", st.session_state.impact_metrics['communication_time_saved'])
        
        st.metric("Communications", st.session_state.impact_metrics['successful_communications'])
        
        st.markdown("---")
        st.subheader("⚙️ Settings")
        simulation_speed = st.slider("Simulation Speed", 1, 10, 3)
        sensitivity = st.slider("AI Sensitivity", 0.1, 1.0, 0.7)
        
        if st.button("🚀 Start Live Demo", type="primary"):
            st.session_state.simulating = True
            st.rerun()
        
        if st.button("⏸️ Pause Demo"):
            st.session_state.simulating = False
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🧠 Live Demo", "📈 Impact Analysis", "🔧 Technology"])
    
    with tab1:
        st.markdown("## 🎥 Live Brain-to-Speech Demo")
        
        if not st.session_state.get('simulating', False):
            st.info("👆 Click 'Start Live Demo' in the sidebar to begin!")
        else:
            # Live demo content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 🧠 Real-time Brain Activity")
                
                # Get current EEG window with improved bounds checking
                window_size = 256
                current_pos = st.session_state.current_pos
                eeg_data_len = len(st.session_state.eeg_data)
                
                # Ensure we don't go beyond data bounds
                if current_pos + window_size >= eeg_data_len:
                    st.session_state.current_pos = 0  # Reset to beginning
                    current_pos = 0
                
                # Use a smaller step for smoother transition
                step_size = window_size // 16  # 1/16th window for high overlap
                eeg_window = st.session_state.eeg_data[current_pos:current_pos + window_size]

                # Ensure we have enough data points
                if len(eeg_window) < window_size:
                    # Pad with the last available data point if needed
                    padding = window_size - len(eeg_window)
                    if len(eeg_window) > 0:
                        eeg_window = np.concatenate([eeg_window, np.full(padding, eeg_window[-1])])
                    else:
                        eeg_window = np.random.randn(window_size) * 10

                # Ensure data is numeric and handle any issues
                try:
                    eeg_window = np.array(eeg_window, dtype=np.float64)
                    # Remove any NaN or inf values
                    if np.any(~np.isfinite(eeg_window)):
                        eeg_window = eeg_window[np.isfinite(eeg_window)]
                        if len(eeg_window) == 0:
                            eeg_window = np.random.randn(window_size) * 10
                        elif len(eeg_window) < window_size:
                            # Pad with mean value if needed
                            mean_val = np.mean(eeg_window)
                            padding = window_size - len(eeg_window)
                            eeg_window = np.concatenate([eeg_window, np.full(padding, mean_val)])
                except (ValueError, TypeError) as e:
                    # Fallback to random data if conversion fails
                    eeg_window = np.random.randn(window_size) * 10
                    st.error(f"Data processing error: {e}")

                # Interpolate for extra smoothness
                interp_factor = 4
                if len(eeg_window) > 1:
                    x = np.arange(len(eeg_window))
                    x_new = np.linspace(0, len(eeg_window)-1, len(eeg_window)*interp_factor)
                    eeg_window_interp = np.interp(x_new, x, eeg_window)
                else:
                    eeg_window_interp = eeg_window

                # Update position for next iteration
                st.session_state.current_pos = (current_pos + step_size) % (len(st.session_state.eeg_data) - window_size)

                # Create EEG plot (hospital style)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=eeg_window_interp,
                    mode='lines',
                    line=dict(color='#00ff88', width=2),
                    name='EEG',
                ))
                fig.update_layout(
                    title="Live EEG Signal (Hospital Style)",
                    height=300,
                    plot_bgcolor='#111',
                    paper_bgcolor='#111',
                    font=dict(color='#fff'),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='#333',
                        zeroline=False,
                        showticklabels=False,
                        range=[0, len(eeg_window_interp)-1],
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='#333',
                        zeroline=False,
                        range=[-50, 50],  # Adjust based on your data range
                    ),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Make prediction
                predicted_word, confidence, probabilities = app.predict_with_confidence(eeg_window)
                
                # Update metrics for high-confidence predictions
                if confidence > 0.6:
                    st.session_state.impact_metrics['successful_communications'] += 1
                    if st.session_state.impact_metrics['successful_communications'] % 10 == 0:
                        st.session_state.impact_metrics['patients_helped'] += 1
                        st.session_state.impact_metrics['communication_time_saved'] += 5
            
            with col2:
                st.markdown("#### 🎯 AI Prediction")
                
                # Display prediction with color coding
                if confidence > 0.7:
                    st.success(f"## 🔥 {predicted_word}")
                elif confidence > 0.4:
                    st.warning(f"## ⚠️ {predicted_word}")
                else:
                    st.info(f"## 💭 {predicted_word}")
                
                st.progress(float(confidence))
                st.write(f"Confidence: {confidence:.1%}")
                
                # Emergency alerts
                if predicted_word in ['HELP', 'PAIN'] and confidence > 0.6:
                    st.error("🚨 EMERGENCY: Patient requires assistance!")
                
                # Probability chart
                prob_fig = go.Figure(data=[
                    go.Bar(x=app.medical_mode_words, y=probabilities,
                          marker_color=['#FF6B6B' if x == predicted_word else '#4ECDC4' for x in app.medical_mode_words])
                ])
                prob_fig.update_layout(title="Word Probabilities", height=250)
                st.plotly_chart(prob_fig, use_container_width=True)
            
            # Auto-continue the simulation only if still simulating
            if st.session_state.get('simulating', False):
                time.sleep(0.5)  # Reduced sleep time for better performance
                st.rerun()
    
    with tab2:
        app.impact_viz.show_impact_dashboard()
    
    with tab3:
        st.markdown("## 🔬 Technology Stack")
        st.markdown("""
        ### 🏗️ System Architecture
        - **Real-time EEG Processing**: Bandpass filtering and feature extraction
        - **AI Classification**: Deep learning model for intention recognition
        - **Speech Synthesis**: Text-to-speech conversion
        
        ### 🤖 AI/ML Components
        - **TensorFlow/Keras**: Neural network implementation
        - **Scipy/Signal Processing**: EEG filtering and analysis
        - **Plotly**: Real-time data visualization
        
        ### 🎯 Medical Applications
        - Locked-in syndrome communication
        - ALS patient assistance
        - Stroke rehabilitation
        - Emergency alert systems
        """)

if __name__ == "__main__":
    main()