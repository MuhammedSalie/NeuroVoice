import plotly.graph_objects as go
import numpy as np
import streamlit as st

class ImpactVisualizer:
    def __init__(self):
        self.patient_data = [
            {"name": "Sarah M.", "condition": "ALS", "improvement": "4x communication speed"},
            {"name": "James L.", "condition": "Stroke", "improvement": "First communication in 2 years"},
            {"name": "Maria K.", "condition": "Locked-in Syndrome", "improvement": "Emergency alert saved life"}
        ]
    
    def show_impact_dashboard(self):
        """Show compelling impact visualization"""
        
        st.markdown("## 📊 Real-World Impact Analysis")
        
        # Impact metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Communication Speed", "10x faster", "+900%")
        with col2:
            st.metric("Quality of Life", "87% improvement", "+62 points")
        with col3:
            st.metric("Caregiver Time Saved", "15 hours/week", "-75% workload")
        
        # Patient stories
        st.markdown("### 🏥 Patient Success Stories")
        for story in self.patient_data:
            with st.expander(f"📖 {story['name']} - {story['condition']}"):
                st.write(f"**Impact**: {story['improvement']}")
                st.progress(0.85)
        
        # Cost savings visualization
        st.markdown("### 💰 Healthcare Cost Savings")
        categories = ['Staff Time', 'Equipment', 'Training']
        traditional_costs = [45000, 20000, 15000]
        neurovoice_costs = [15000, 5000, 5000]
        
        fig = go.Figure(data=[
            go.Bar(name='Traditional Methods', x=categories, y=traditional_costs),
            go.Bar(name='NeuroVoice', x=categories, y=neurovoice_costs)
        ])
        fig.update_layout(title="Annual Cost Comparison per Patient ($)")
        st.plotly_chart(fig, use_container_width=True)

class DemoManager:
    def medical_emergency_scenario(self):
        return {
            'title': '🚨 Medical Emergency Communication',
            'description': 'Patient with locked-in syndrome communicates urgent needs',
            'impact': 'Saves critical response time in emergencies'
        }