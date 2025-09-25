import streamlit as st
import os

def setup_google_services():
    st.title("🔧 Google Services Setup for NeuroVoice")
    
    st.markdown("""
    ## 🚀 Setting Up Google Cloud
    
    1. **Create Google Cloud Project**
       - Go to [console.cloud.google.com](https://console.cloud.google.com)
       - Create new project: `neurovoice-hackathon`
    
    2. **Enable Required APIs**
    ```bash
    gcloud services enable texttospeech.googleapis.com
    gcloud services enable aiplatform.googleapis.com
    ```
    
    3. **Set Up Authentication**
    ```bash
    # Create service account
    gcloud iam service-accounts create neurovoice-sa
    
    # Download credentials JSON
    gcloud iam service-accounts keys create key.json \
        --iam-account neurovoice-sa@your-project.iam.gserviceaccount.com
    
    # Set environment variable
    export GOOGLE_APPLICATION_CREDENTIALS="key.json"
    ```
    
    4. **Run the Enhanced App**
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)
    
    if st.button("Check Google Services"):
        try:
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechClient()
            st.success("✅ Google TTS API is accessible!")
        except Exception as e:
            st.error(f"❌ Google setup needed: {e}")

if __name__ == "__main__":
    setup_google_services()