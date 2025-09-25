import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import random
import time

class ImpactVisualizer:
    def __init__(self):
        self.demo_data = self._generate_demo_data()
    
    def _generate_demo_data(self):
        """Generate demo impact data"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
        patients_helped = [1, 3, 8, 15, 28, 45, 67, 89, 112]
        communication_time = [2, 8, 23, 45, 78, 125, 189, 267, 356]
        accuracy_scores = [0.72, 0.75, 0.78, 0.82, 0.85, 0.87, 0.89, 0.91, 0.93]
        
        return {
            'months': months,
            'patients_helped': patients_helped,
            'communication_time': communication_time,
            'accuracy_scores': accuracy_scores
        }
    
    def show_impact_dashboard(self):
        """Display the impact analysis dashboard"""
        st.markdown("## 📊 NeuroVoice Impact Analysis")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "👥 Total Patients", 
                "112",
                delta="23 this month"
            )
        
        with col2:
            st.metric(
                "⏰ Time Saved", 
                "356 hrs",
                delta="89 hrs this month"
            )
        
        with col3:
            st.metric(
                "🎯 Accuracy", 
                "93%",
                delta="2% improvement"
            )
        
        with col4:
            st.metric(
                "🏥 Partner Hospitals", 
                "8",
                delta="2 new partnerships"
            )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Patients helped over time
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=self.demo_data['months'],
                y=self.demo_data['patients_helped'],
                mode='lines+markers',
                name='Patients Helped',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8)
            ))
            fig1.update_layout(
                title="📈 Patients Helped Over Time",
                xaxis_title="Month",
                yaxis_title="Cumulative Patients",
                height=300
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Accuracy improvement
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=self.demo_data['months'],
                y=self.demo_data['accuracy_scores'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=8)
            ))
            fig2.update_layout(
                title="🎯 Model Accuracy Improvement",
                xaxis_title="Month",
                yaxis_title="Accuracy Score",
                height=300,
                yaxis=dict(range=[0.7, 1.0])
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Time saved chart
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=self.demo_data['months'],
            y=self.demo_data['communication_time'],
            name='Time Saved (Hours)',
            marker_color='#45B7D1'
        ))
        fig3.update_layout(
            title="⏰ Communication Time Saved",
            xaxis_title="Month",
            yaxis_title="Hours Saved",
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Patient testimonials
        st.markdown("---")
        st.markdown("### 💬 Patient Impact Stories")
        
        testimonials = [
            {
                "name": "Sarah M.",
                "condition": "ALS Patient",
                "story": "NeuroVoice gave me my voice back. I can now communicate my needs clearly to my family and medical team.",
                "rating": 5
            },
            {
                "name": "Dr. James R.",
                "role": "Neurologist",
                "story": "This technology has revolutionized how we communicate with locked-in syndrome patients in our ICU.",
                "rating": 5
            },
            {
                "name": "Maria L.",
                "condition": "Stroke Survivor",
                "story": "The system understood my thoughts when I couldn't speak. It's truly life-changing technology.",
                "rating": 5
            }
        ]
        
        for testimonial in testimonials:
            with st.expander(f"⭐ {testimonial['name']} - {testimonial.get('condition', testimonial.get('role', ''))}"):
                st.write(f"*\"{testimonial['story']}\"*")
                st.write("⭐" * testimonial['rating'])

class DemoManager:
    def __init__(self):
        self.demo_scenarios = [
            {
                "name": "Emergency Alert",
                "description": "Patient needs immediate help",
                "word": "HELP",
                "urgency": "high"
            },
            {
                "name": "Pain Assessment",
                "description": "Patient experiencing discomfort",
                "word": "PAIN",
                "urgency": "high"
            },
            {
                "name": "Basic Needs",
                "description": "Patient requesting water",
                "word": "WATER",
                "urgency": "medium"
            },
            {
                "name": "Positive Response",
                "description": "Patient agreeing or confirming",
                "word": "YES",
                "urgency": "low"
            },
            {
                "name": "Negative Response",
                "description": "Patient disagreeing or declining",
                "word": "NO",
                "urgency": "low"
            }
        ]
    
    def get_random_scenario(self):
        """Get a random demo scenario"""
        return random.choice(self.demo_scenarios)
    
    def simulate_patient_interaction(self, scenario):
        """Simulate a patient interaction"""
        # This could be expanded to include more complex simulation logic
        confidence_ranges = {
            "high": (0.7, 0.95),
            "medium": (0.5, 0.8),
            "low": (0.3, 0.7)
        }
        
        urgency = scenario["urgency"]
        confidence_range = confidence_ranges[urgency]
        simulated_confidence = random.uniform(*confidence_range)
        
        return {
            "predicted_word": scenario["word"],
            "confidence": simulated_confidence,
            "scenario": scenario
        }

class RealtimeEEGVisualizer:
    def __init__(self):
        self.buffer_size = 1000
        self.eeg_buffer = np.zeros(self.buffer_size)
        
    def update_buffer(self, new_data):
        """Update the EEG data buffer"""
        if len(new_data) >= self.buffer_size:
            self.eeg_buffer = new_data[-self.buffer_size:]
        else:
            # Shift existing data and add new data
            shift_amount = len(new_data)
            self.eeg_buffer[:-shift_amount] = self.eeg_buffer[shift_amount:]
            self.eeg_buffer[-shift_amount:] = new_data
    
    def create_realtime_plot(self, title="Real-time EEG"):
        """Create a real-time EEG plot"""
        fig = go.Figure()
        
        # Time axis
        time_axis = np.arange(len(self.eeg_buffer))
        
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=self.eeg_buffer,
            mode='lines',
            name='EEG Signal',
            line=dict(color='#00ff88', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (samples)",
            yaxis_title="Amplitude (μV)",
            plot_bgcolor='#111',
            paper_bgcolor='#111',
            font=dict(color='#fff'),
            height=300
        )
        
        return fig
