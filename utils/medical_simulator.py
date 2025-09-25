import numpy as np

class MedicalEEGSimulator:
    def __init__(self):
        self.sample_rate = 256
    
    def generate_pain_pattern(self, t):
        return 0.8 * np.sin(2 * np.pi * 25 * t) * np.exp(-0.1 * t)
    
    def generate_emergency_pattern(self, t):
        return 1.0 * (np.sin(2 * np.pi * 18 * t) + np.sin(2 * np.pi * 35 * t))