import streamlit as st
import os

def main():
    st.title("🧠NeuroVoice")
    st.write("Welcome to NeuroVoice, the AI-powered voice assistant.")

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
