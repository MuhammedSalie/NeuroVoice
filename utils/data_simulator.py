import numpy as np
import random
from scipy import signal

class EEGDataSimulator:
    def __init__(self):
        self.sampling_rate = 250  # Hz
        self.channels = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        
    def generate_medical_eeg_data(self, duration_samples, word_type='HELP'):
        """Generate simulated EEG data for medical scenarios"""
        # Base EEG signal with alpha waves (8-12 Hz)
        t = np.linspace(0, duration_samples/self.sampling_rate, duration_samples)
        
        # Different patterns for different words
        if word_type == 'HELP':
            # Emergency signal - higher amplitude, more irregular
            signal_data = (
                2 * np.sin(2 * np.pi * 10 * t) +  # Alpha
                1.5 * np.sin(2 * np.pi * 20 * t) +  # Beta
                0.8 * np.random.randn(duration_samples) +  # Noise
                3 * signal.square(2 * np.pi * 0.5 * t)  # P300-like response
            )
        elif word_type == 'PAIN':
            # Pain signal - high frequency components
            signal_data = (
                1.5 * np.sin(2 * np.pi * 12 * t) +  # Alpha
                2 * np.sin(2 * np.pi * 25 * t) +  # Beta
                1.2 * np.random.randn(duration_samples) +
                2 * signal.sawtooth(2 * np.pi * 0.3 * t)
            )
        elif word_type == 'WATER':
            # Calmer signal
            signal_data = (
                2.5 * np.sin(2 * np.pi * 8 * t) +  # Alpha
                0.8 * np.sin(2 * np.pi * 15 * t) +  # Beta
                0.5 * np.random.randn(duration_samples)
            )
        elif word_type == 'YES':
            # Positive response pattern
            signal_data = (
                2 * np.sin(2 * np.pi * 9 * t) +
                1 * np.sin(2 * np.pi * 18 * t) +
                0.6 * np.random.randn(duration_samples) +
                1.5 * np.sin(2 * np.pi * 0.8 * t)  # Slow positive wave
            )
        elif word_type == 'NO':
            # Negative response pattern
            signal_data = (
                1.8 * np.sin(2 * np.pi * 11 * t) +
                1.2 * np.sin(2 * np.pi * 22 * t) +
                0.7 * np.random.randn(duration_samples) -
                1 * np.sin(2 * np.pi * 0.6 * t)  # Slow negative wave
            )
        else:
            # Default thinking pattern
            signal_data = (
                1.5 * np.sin(2 * np.pi * 10 * t) +
                0.8 * np.sin(2 * np.pi * 20 * t) +
                0.4 * np.random.randn(duration_samples)
            )
        
        # Add realistic EEG artifacts
        # Eye blink artifacts (low frequency, high amplitude)
        if random.random() > 0.7:
            blink_times = np.random.choice(len(t), size=random.randint(1, 3))
            for blink_time in blink_times:
                if blink_time < len(signal_data) - 50:
                    signal_data[blink_time:blink_time+50] += 15 * np.exp(-np.arange(50)/10)
        
        # Muscle artifacts (high frequency)
        if random.random() > 0.8:
            muscle_start = random.randint(0, len(signal_data)//2)
            muscle_duration = random.randint(50, 200)
            if muscle_start + muscle_duration < len(signal_data):
                signal_data[muscle_start:muscle_start+muscle_duration] += \
                    3 * np.random.randn(muscle_duration)
        
        return signal_data
        
    def generate_random_eeg_burst(self, word_class='THINKING'):
        """Generate a short burst of EEG data"""
        duration = random.randint(200, 500)  # samples
        return self.generate_medical_eeg_data(duration, word_class)
