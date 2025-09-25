import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
import pywt

class EEGPreprocessor:
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        
    def apply_filters(self, eeg_data):
        """Apply bandpass filtering to EEG data"""
        # Bandpass filter for EEG frequencies (1-50 Hz)
        nyquist = self.sampling_rate / 2
        low_cutoff = 1.0 / nyquist
        high_cutoff = 50.0 / nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        
        # Apply filter
        filtered_data = signal.filtfilt(b, a, eeg_data)
        
        # Notch filter for power line interference (50 Hz)
        notch_freq = 50.0 / nyquist
        b_notch, a_notch = signal.iirnotch(notch_freq, 30)
        filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data)
        
        return filtered_data
    
    def extract_advanced_features(self, eeg_data):
        """Extract advanced features from EEG data"""
        features = []
        
        # Statistical features
        features.append(np.mean(eeg_data))
        features.append(np.std(eeg_data))
        features.append(skew(eeg_data))
        features.append(kurtosis(eeg_data))
        features.append(np.max(eeg_data))
        features.append(np.min(eeg_data))
        
        # Frequency domain features
        freqs, psd = signal.welch(eeg_data, fs=self.sampling_rate, nperseg=min(256, len(eeg_data)))
        
        # Power in different frequency bands
        delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
        theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
        beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
        gamma_power = np.sum(psd[(freqs >= 30) & (freqs <= 50)])
        
        total_power = np.sum(psd)
        if total_power > 0:
            features.extend([
                delta_power / total_power,
                theta_power / total_power,
                alpha_power / total_power,
                beta_power / total_power,
                gamma_power / total_power
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Wavelet features
        try:
            coeffs = pywt.wavedec(eeg_data, 'db4', level=4)
            for coeff in coeffs:
                if len(coeff) > 0:
                    features.append(np.mean(coeff))
                    features.append(np.std(coeff))
                else:
                    features.extend([0, 0])
        except:
            features.extend([0] * 10)  # Fallback
        
        # Time-frequency features
        try:
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(eeg_data)) != 0) / len(eeg_data)
            features.append(zero_crossings)
            
            # Hjorth parameters
            features.extend(self._hjorth_parameters(eeg_data))
        except:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def _hjorth_parameters(self, signal_data):
        """Calculate Hjorth parameters (Activity, Mobility, Complexity)"""
        # First derivative
        first_deriv = np.diff(signal_data)
        # Second derivative  
        second_deriv = np.diff(first_deriv)
        
        # Variance
        var_signal = np.var(signal_data)
        var_first_deriv = np.var(first_deriv)
        var_second_deriv = np.var(second_deriv)
        
        # Activity
        activity = var_signal
        
        # Mobility
        if var_signal > 0:
            mobility = np.sqrt(var_first_deriv / var_signal)
        else:
            mobility = 0
            
        # Complexity
        if var_first_deriv > 0 and mobility > 0:
            complexity = np.sqrt(var_second_deriv / var_first_deriv) / mobility
        else:
            complexity = 0
            
        return [activity, mobility, complexity]
    
    def remove_artifacts(self, eeg_data, threshold=3):
        """Remove artifacts using simple thresholding"""
        # Remove samples that are beyond threshold standard deviations
        mean_val = np.mean(eeg_data)
        std_val = np.std(eeg_data)
        
        # Create mask for good samples
        good_samples = np.abs(eeg_data - mean_val) < threshold * std_val
        
        # Interpolate bad samples
        if np.sum(~good_samples) > 0:
            bad_indices = np.where(~good_samples)[0]
            good_indices = np.where(good_samples)[0]
            
            if len(good_indices) > 0:
                eeg_data[bad_indices] = np.interp(bad_indices, good_indices, eeg_data[good_indices])
        
        return eeg_data
