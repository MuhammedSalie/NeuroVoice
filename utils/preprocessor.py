import numpy as np
from scipy import signal

class EEGPreprocessor:
    def __init__(self, sample_rate=256):
        self.sample_rate = sample_rate
        
    def apply_filters(self, eeg_data):
        """Apply bandpass and notch filters"""
        # Bandpass filter 1-40Hz
        nyquist = self.sample_rate / 2
        low = 1.0 / nyquist
        high = 40.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, eeg_data)
        
        # Notch filter 50Hz (power line noise)
        notch_freq = 50.0
        quality = 30.0
        b, a = signal.iirnotch(notch_freq, quality, self.sample_rate)
        filtered_data = signal.filtfilt(b, a, filtered_data)
        
        return filtered_data
    
    def extract_features(self, eeg_window):
        """Extract frequency band features"""
        frequencies, power_spectrum = signal.welch(eeg_window, self.sample_rate, nperseg=64)
        
        # Define frequency bands
        alpha_band = (8, 13)
        beta_band = (13, 30)
        theta_band = (4, 8)
        
        alpha_power = np.mean(power_spectrum[(frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1])])
        beta_power = np.mean(power_spectrum[(frequencies >= beta_band[0]) & (frequencies <= beta_band[1])])
        theta_power = np.mean(power_spectrum[(frequencies >= theta_band[0]) & (frequencies <= theta_band[1])])
        total_power = np.mean(power_spectrum)
        
        return np.array([alpha_power, beta_power, theta_power, total_power])
    
    def extract_advanced_features(self, eeg_window):
        """Enhanced feature extraction for medical applications"""
        basic_features = self.extract_features(eeg_window)
        
        # Additional features
        complexity = np.std(eeg_window)  # Signal complexity
        coherence = np.mean(np.diff(eeg_window)**2)  # Signal coherence
        
        # Combine all features
        advanced_features = np.concatenate([basic_features, [complexity, coherence]])
        return advanced_features