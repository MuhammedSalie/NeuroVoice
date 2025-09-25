import numpy as np
from scipy import signal

class EEGDataSimulator:
    def __init__(self, sample_rate=256):
        self.sample_rate = sample_rate
        
    def generate_sample_data(self, num_samples=1000):
        """Generate synthetic EEG data for demo purposes"""
        time = np.linspace(0, 10, num_samples)
        
        # Simulate different brain waves
        alpha_wave = 0.5 * np.sin(2 * np.pi * 10 * time)
        beta_wave = 0.3 * np.sin(2 * np.pi * 20 * time)
        theta_wave = 0.7 * np.sin(2 * np.pi * 5 * time)
        noise = 0.1 * np.random.normal(size=num_samples)
        
        eeg_data = alpha_wave + beta_wave + theta_wave + noise
        return eeg_data
    
    def generate_medical_eeg_data(self, duration_seconds=10, intention='HELP'):
        """Generate medically realistic EEG patterns"""
        sample_rate = 256
        t = np.linspace(0, duration_seconds, duration_seconds * sample_rate)
        
        # Base brain rhythms
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)
        beta = 0.3 * np.sin(2 * np.pi * 20 * t)
        theta = 0.7 * np.sin(2 * np.pi * 5 * t)
        
        # Intention-specific patterns
        if intention == 'PAIN':
            intention_pattern = 0.8 * np.sin(2 * np.pi * 25 * t) * np.exp(-0.1 * t)
        elif intention == 'HELP':
            intention_pattern = 1.0 * (np.sin(2 * np.pi * 18 * t) + np.sin(2 * np.pi * 35 * t))
        elif intention == 'WATER':
            intention_pattern = 0.6 * np.sin(2 * np.pi * 12 * t)
        else:
            intention_pattern = 0.5 * np.sin(2 * np.pi * 15 * t)
        
        noise = 0.1 * np.random.normal(size=len(t))
        eeg_signal = alpha + beta + theta + intention_pattern + noise
        
        return eeg_signal
    
    def eeg_stream(self, data, window_size=256, step_size=128):
        """Stream EEG data in real-time chunks"""
        for i in range(0, len(data) - window_size, step_size):
            yield data[i:i + window_size]