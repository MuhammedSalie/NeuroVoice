import streamlit as st
import os

def main():
    st.title("🧠 NeuroVoice Setup Guide")
    
    st.markdown("""
    ## 🚀 Quick Start Guide
    
    1. **Install Python** (if not already installed)
    2. **Open Terminal in VS Code** (Ctrl+` or Terminal → New Terminal)
    3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    4. **Run the app**:
    ```bash
    streamlit run app.py
    ```
    
    The app will open in your web browser automatically!
    """)
    
    # Check if requirements are installed
    if st.button("Check Setup"):
        try:
            import streamlit
            import tensorflow
            import numpy
            st.success("✅ All dependencies are installed correctly!")
        except ImportError as e:
            st.error(f"❌ Missing dependency: {e}")

if __name__ == "__main__":
    main()