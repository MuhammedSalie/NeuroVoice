import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

class VertexAIManager:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def load_emotions_dataset(self, data_path):
        """Load and preprocess the EEG emotions dataset"""
        try:
            # Load the dataset (adjust based on your actual file structure)
            df = pd.read_csv(data_path)
            
            st.info(f"📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Map emotions to medical commands
            emotion_map = {
                'NEGATIVE': 'HELP',
                'NEUTRAL': 'WATER', 
                'POSITIVE': 'YES'
            }
            
            if 'label' in df.columns:
                df['medical_label'] = df['label'].map(emotion_map)
                df = df.dropna()  # Remove unmapped labels
                
                # Separate features and labels
                X = df.drop(['label', 'medical_label'], axis=1)
                y = pd.get_dummies(df['medical_label'])
                
                st.success(f"✅ Processed {len(X)} samples for medical communication")
                return X, y
            else:
                st.warning("❓ No 'label' column found, using synthetic data")
                return None, None
                
        except Exception as e:
            st.error(f"❌ Dataset loading failed: {e}")
            return None, None
    
    def create_enhanced_model(self, input_shape, num_classes):
        """Create an enhanced model for better accuracy"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.BatchNormalization(),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model_with_dataset(self, data_path):
        """Train model using real EEG emotions data"""
        X, y = self.load_emotions_dataset(data_path)
        
        if X is not None and y is not None:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train model
            model = self.create_enhanced_model(X_train_scaled.shape[1], y.shape[1])
            
            # Train with callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            history = model.fit(
                X_train_scaled, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test_scaled, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
            st.success(f"🎯 Model trained with {test_accuracy:.1%} accuracy on real EEG data!")
            
            return model, history, self.scaler
            
        else:
            st.warning("🔄 Using synthetic data for demo purposes")
            return None, None, None