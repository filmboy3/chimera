#!/usr/bin/env python3
"""
Sesame v2 Cognitive State Modulator
Real-time fusion of motion and audio cues for fatigue/focus estimation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from collections import deque
import time


class CognitiveStateModulator(nn.Module):
    """
    Lightweight neural network for real-time cognitive state estimation
    Fuses motion metrics from QLv3 and audio cues to estimate fatigue/focus
    """
    
    def __init__(
        self,
        motion_features: int = 8,  # jitter, jerk_entropy, velocity_variance, etc.
        audio_features: int = 6,   # breathing_rate, vocal_strain, speech_clarity, etc.
        hidden_dim: int = 64,
        sequence_length: int = 10,  # 10 seconds of history at 1Hz
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.motion_features = motion_features
        self.audio_features = audio_features
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # Input feature processing
        self.motion_encoder = nn.Sequential(
            nn.Linear(motion_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal modeling with LSTM
        self.temporal_model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Cognitive state prediction heads
        self.fatigue_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Fatigue score 0-1
        )
        
        self.focus_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Focus score 0-1
        )
        
        # Interruption cost model
        self.interruption_cost_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 4),  # +2 for fatigue/focus
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Interruption cost 0-1 (higher = worse time to interrupt)
        )
    
    def forward(
        self, 
        motion_features: torch.Tensor, 
        audio_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for cognitive state estimation
        
        Args:
            motion_features: (batch_size, sequence_length, motion_features)
            audio_features: (batch_size, sequence_length, audio_features)
            
        Returns:
            Dictionary with fatigue, focus, and interruption cost predictions
        """
        batch_size, seq_len = motion_features.shape[:2]
        
        # Encode features
        motion_encoded = self.motion_encoder(motion_features)  # (B, L, H/2)
        audio_encoded = self.audio_encoder(audio_features)     # (B, L, H/2)
        
        # Fuse modalities
        fused_features = torch.cat([motion_encoded, audio_encoded], dim=-1)  # (B, L, H)
        
        # Temporal modeling
        lstm_out, (hidden, cell) = self.temporal_model(fused_features)
        
        # Use last timestep for prediction
        last_hidden = lstm_out[:, -1, :]  # (B, H)
        
        # Predict cognitive states
        fatigue = self.fatigue_head(last_hidden)  # (B, 1)
        focus = self.focus_head(last_hidden)      # (B, 1)
        
        # Predict interruption cost (includes current fatigue/focus as input)
        interruption_input = torch.cat([last_hidden, fatigue, focus], dim=-1)
        interruption_cost = self.interruption_cost_head(interruption_input)  # (B, 1)
        
        return {
            'fatigue': fatigue.squeeze(-1),           # (B,)
            'focus': focus.squeeze(-1),               # (B,)
            'interruption_cost': interruption_cost.squeeze(-1),  # (B,)
            'temporal_features': lstm_out             # (B, L, H) for analysis
        }
    
    def get_model_size(self) -> Dict[str, int]:
        """Calculate model size for deployment"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'total_parameters': total_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }


class MotionFeatureExtractor:
    """
    Extract cognitive-relevant features from IMU motion data
    """
    
    def __init__(self, window_size: int = 100):  # 1 second at 100Hz
        self.window_size = window_size
        self.motion_history = deque(maxlen=window_size * 10)  # 10 seconds of history
    
    def extract_features(self, imu_data: np.ndarray) -> Dict[str, float]:
        """
        Extract motion features indicative of fatigue and focus
        
        Args:
            imu_data: (n_samples, 6) - [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            
        Returns:
            Dictionary of motion features
        """
        if len(imu_data) == 0:
            return self._get_default_features()
        
        # Add to history
        for sample in imu_data:
            self.motion_history.append(sample)
        
        if len(self.motion_history) < self.window_size:
            return self._get_default_features()
        
        # Convert to numpy array
        motion_array = np.array(list(self.motion_history))
        accel = motion_array[:, :3]
        gyro = motion_array[:, 3:]
        
        # Feature extraction
        features = {}
        
        # 1. Jitter (high-frequency noise indicating fatigue/tremor)
        accel_jitter = np.mean(np.std(np.diff(accel, axis=0), axis=0))
        gyro_jitter = np.mean(np.std(np.diff(gyro, axis=0), axis=0))
        features['jitter'] = float(accel_jitter + gyro_jitter)
        
        # 2. Movement smoothness (jerk analysis)
        accel_jerk = np.diff(accel, n=2, axis=0)  # Second derivative
        jerk_magnitude = np.sqrt(np.sum(accel_jerk**2, axis=1))
        features['jerk_entropy'] = float(self._calculate_entropy(jerk_magnitude))
        
        # 3. Movement consistency (velocity variance)
        velocity = np.diff(accel, axis=0)
        velocity_variance = np.mean(np.var(velocity, axis=0))
        features['velocity_variance'] = float(velocity_variance)
        
        # 4. Postural stability (low-frequency drift)
        accel_mean = np.mean(accel, axis=0)
        postural_drift = np.sqrt(np.sum((accel_mean - np.array([0, 0, 9.81]))**2))
        features['postural_drift'] = float(postural_drift)
        
        # 5. Movement amplitude (range of motion)
        accel_range = np.ptp(accel, axis=0)  # Peak-to-peak
        gyro_range = np.ptp(gyro, axis=0)
        features['movement_amplitude'] = float(np.mean(accel_range) + np.mean(gyro_range))
        
        # 6. Frequency domain features
        accel_magnitude = np.sqrt(np.sum(accel**2, axis=1))
        fft_features = self._extract_frequency_features(accel_magnitude)
        features.update(fft_features)
        
        return features
    
    def _calculate_entropy(self, signal: np.ndarray, bins: int = 20) -> float:
        """Calculate signal entropy as measure of irregularity"""
        if len(signal) == 0:
            return 0.0
        
        hist, _ = np.histogram(signal, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def _extract_frequency_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features"""
        if len(signal) < 32:  # Need minimum samples for FFT
            return {'dominant_frequency': 0.0, 'spectral_centroid': 0.0}
        
        # FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/100)  # 100Hz sampling rate
        
        # Power spectral density
        psd = np.abs(fft)**2
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(psd[:len(psd)//2])  # Only positive frequencies
        dominant_frequency = float(np.abs(freqs[dominant_freq_idx]))
        
        # Spectral centroid (measure of "brightness")
        spectral_centroid = float(np.sum(freqs[:len(freqs)//2] * psd[:len(psd)//2]) / 
                                 (np.sum(psd[:len(psd)//2]) + 1e-10))
        
        return {
            'dominant_frequency': dominant_frequency,
            'spectral_centroid': spectral_centroid
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when insufficient data"""
        return {
            'jitter': 0.0,
            'jerk_entropy': 0.0,
            'velocity_variance': 0.0,
            'postural_drift': 0.0,
            'movement_amplitude': 0.0,
            'dominant_frequency': 0.0,
            'spectral_centroid': 0.0
        }


class AudioFeatureExtractor:
    """
    Extract cognitive-relevant features from audio data
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_history = deque(maxlen=sample_rate * 10)  # 10 seconds
    
    def extract_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract audio features indicative of fatigue and focus
        
        Args:
            audio_data: Raw audio samples
            
        Returns:
            Dictionary of audio features
        """
        if len(audio_data) == 0:
            return self._get_default_features()
        
        # Add to history
        for sample in audio_data:
            self.audio_history.append(sample)
        
        if len(self.audio_history) < self.sample_rate:  # Need at least 1 second
            return self._get_default_features()
        
        audio_array = np.array(list(self.audio_history))
        
        features = {}
        
        # 1. Breathing rate estimation (from low-frequency components)
        breathing_rate = self._estimate_breathing_rate(audio_array)
        features['breathing_rate'] = breathing_rate
        
        # 2. Vocal strain (from spectral characteristics)
        vocal_strain = self._estimate_vocal_strain(audio_array)
        features['vocal_strain'] = vocal_strain
        
        # 3. Speech clarity (from high-frequency content)
        speech_clarity = self._estimate_speech_clarity(audio_array)
        features['speech_clarity'] = speech_clarity
        
        # 4. Volume consistency
        volume_consistency = self._estimate_volume_consistency(audio_array)
        features['volume_consistency'] = volume_consistency
        
        # 5. Pause patterns (silence detection)
        pause_frequency = self._estimate_pause_frequency(audio_array)
        features['pause_frequency'] = pause_frequency
        
        # 6. Overall audio energy
        audio_energy = float(np.mean(audio_array**2))
        features['audio_energy'] = audio_energy
        
        return features
    
    def _estimate_breathing_rate(self, audio: np.ndarray) -> float:
        """Estimate breathing rate from audio (simplified)"""
        # Low-pass filter for breathing sounds (0.1-1 Hz)
        # This is a simplified version - real implementation would use proper filtering
        window_size = self.sample_rate // 10  # 0.1 second windows
        energy_windows = []
        
        for i in range(0, len(audio) - window_size, window_size):
            window = audio[i:i + window_size]
            energy = np.mean(window**2)
            energy_windows.append(energy)
        
        if len(energy_windows) < 10:  # Need enough windows
            return 0.5  # Default breathing rate
        
        # Find peaks in energy (breathing cycles)
        energy_array = np.array(energy_windows)
        # Simplified peak detection
        peaks = 0
        for i in range(1, len(energy_array) - 1):
            if energy_array[i] > energy_array[i-1] and energy_array[i] > energy_array[i+1]:
                peaks += 1
        
        # Convert to breaths per minute
        duration_minutes = len(energy_windows) * 0.1 / 60
        breathing_rate = peaks / duration_minutes if duration_minutes > 0 else 15.0
        
        return float(np.clip(breathing_rate, 5.0, 40.0))  # Reasonable range
    
    def _estimate_vocal_strain(self, audio: np.ndarray) -> float:
        """Estimate vocal strain from spectral characteristics"""
        # FFT analysis
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        psd = np.abs(fft)**2
        
        # Focus on speech frequencies (300-3000 Hz)
        speech_mask = (freqs >= 300) & (freqs <= 3000)
        if not np.any(speech_mask):
            return 0.0
        
        speech_psd = psd[speech_mask]
        speech_freqs = freqs[speech_mask]
        
        # Vocal strain indicators: high-frequency emphasis, spectral tilt
        high_freq_mask = speech_freqs > 1500
        low_freq_mask = speech_freqs < 1500
        
        if np.any(high_freq_mask) and np.any(low_freq_mask):
            high_energy = np.mean(speech_psd[high_freq_mask])
            low_energy = np.mean(speech_psd[low_freq_mask])
            strain_ratio = high_energy / (low_energy + 1e-10)
            return float(np.clip(strain_ratio / 10.0, 0.0, 1.0))  # Normalize
        
        return 0.0
    
    def _estimate_speech_clarity(self, audio: np.ndarray) -> float:
        """Estimate speech clarity from high-frequency content"""
        # Simple measure: ratio of high-frequency to total energy
        fft = np.fft.fft(audio)
        psd = np.abs(fft)**2
        
        total_energy = np.sum(psd)
        if total_energy == 0:
            return 1.0
        
        # High-frequency energy (2-8 kHz for consonants)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        hf_mask = (freqs >= 2000) & (freqs <= 8000)
        hf_energy = np.sum(psd[hf_mask])
        
        clarity = hf_energy / total_energy
        return float(np.clip(clarity * 5.0, 0.0, 1.0))  # Normalize
    
    def _estimate_volume_consistency(self, audio: np.ndarray) -> float:
        """Estimate volume consistency (lower variance = higher consistency)"""
        # RMS energy in overlapping windows
        window_size = self.sample_rate // 10  # 0.1 second windows
        hop_size = window_size // 2
        
        rms_values = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
        
        if len(rms_values) < 2:
            return 1.0
        
        rms_array = np.array(rms_values)
        consistency = 1.0 / (1.0 + np.std(rms_array))  # Higher consistency = lower std
        return float(np.clip(consistency, 0.0, 1.0))
    
    def _estimate_pause_frequency(self, audio: np.ndarray) -> float:
        """Estimate frequency of pauses in speech"""
        # Simple silence detection
        threshold = np.std(audio) * 0.1  # 10% of std as silence threshold
        
        # Find silent regions
        silent_samples = np.abs(audio) < threshold
        
        # Count transitions from non-silent to silent (pause starts)
        transitions = np.diff(silent_samples.astype(int))
        pause_starts = np.sum(transitions == 1)
        
        # Convert to pauses per minute
        duration_minutes = len(audio) / self.sample_rate / 60
        pause_frequency = pause_starts / duration_minutes if duration_minutes > 0 else 0.0
        
        return float(np.clip(pause_frequency / 60.0, 0.0, 1.0))  # Normalize
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when insufficient data"""
        return {
            'breathing_rate': 15.0,  # Normal resting rate
            'vocal_strain': 0.0,
            'speech_clarity': 1.0,
            'volume_consistency': 1.0,
            'pause_frequency': 0.0,
            'audio_energy': 0.0
        }


class CognitiveStateProcessor:
    """
    Real-time cognitive state processing pipeline
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        update_rate: float = 1.0,  # 1 Hz updates
        history_length: int = 10   # 10 seconds of history
    ):
        self.update_rate = update_rate
        self.history_length = history_length
        
        # Load model
        if model_path and Path(model_path).exists():
            self.model = self._load_model(model_path)
        else:
            self.model = CognitiveStateModulator()
        
        self.model.eval()
        
        # Feature extractors
        self.motion_extractor = MotionFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        
        # Feature history for temporal modeling
        self.feature_history = deque(maxlen=history_length)
        
        # Current state
        self.current_state = {
            'fatigue': 0.0,
            'focus': 1.0,
            'interruption_cost': 0.0,
            'last_update': time.time()
        }
        
        print(f"CognitiveStateProcessor initialized with {self.model.get_model_size()['model_size_mb']:.1f}MB model")
    
    def _load_model(self, model_path: str) -> CognitiveStateModulator:
        """Load trained model"""
        model = CognitiveStateModulator()
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def update_state(
        self, 
        imu_data: Optional[np.ndarray] = None, 
        audio_data: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Update cognitive state with new sensor data
        
        Args:
            imu_data: Recent IMU samples (n_samples, 6)
            audio_data: Recent audio samples
            
        Returns:
            Updated cognitive state
        """
        current_time = time.time()
        
        # Extract features
        motion_features = self.motion_extractor.extract_features(imu_data or np.array([]))
        audio_features = self.audio_extractor.extract_features(audio_data or np.array([]))
        
        # Combine features
        combined_features = {
            'timestamp': current_time,
            'motion': motion_features,
            'audio': audio_features
        }
        
        # Add to history
        self.feature_history.append(combined_features)
        
        # Need sufficient history for temporal modeling
        if len(self.feature_history) < 3:
            return self.current_state
        
        # Prepare input tensors
        motion_tensor, audio_tensor = self._prepare_model_input()
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(motion_tensor, audio_tensor)
        
        # Update state
        self.current_state.update({
            'fatigue': float(predictions['fatigue'].item()),
            'focus': float(predictions['focus'].item()),
            'interruption_cost': float(predictions['interruption_cost'].item()),
            'last_update': current_time
        })
        
        return self.current_state.copy()
    
    def _prepare_model_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare model input from feature history"""
        
        # Extract feature vectors
        motion_features = []
        audio_features = []
        
        for frame in list(self.feature_history)[-self.history_length:]:
            # Motion features (8 features)
            motion_vec = [
                frame['motion']['jitter'],
                frame['motion']['jerk_entropy'],
                frame['motion']['velocity_variance'],
                frame['motion']['postural_drift'],
                frame['motion']['movement_amplitude'],
                frame['motion']['dominant_frequency'],
                frame['motion']['spectral_centroid'],
                0.0  # Placeholder for 8th feature
            ]
            
            # Audio features (6 features)
            audio_vec = [
                frame['audio']['breathing_rate'] / 30.0,  # Normalize to 0-1
                frame['audio']['vocal_strain'],
                frame['audio']['speech_clarity'],
                frame['audio']['volume_consistency'],
                frame['audio']['pause_frequency'],
                frame['audio']['audio_energy']
            ]
            
            motion_features.append(motion_vec)
            audio_features.append(audio_vec)
        
        # Pad if necessary
        while len(motion_features) < self.history_length:
            motion_features.insert(0, [0.0] * 8)
            audio_features.insert(0, [0.0] * 6)
        
        # Convert to tensors
        motion_tensor = torch.FloatTensor(motion_features).unsqueeze(0)  # (1, seq_len, 8)
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0)    # (1, seq_len, 6)
        
        return motion_tensor, audio_tensor
    
    def should_interrupt(self, urgency: float = 0.5) -> Dict[str, any]:
        """
        Determine if it's appropriate to interrupt the user
        
        Args:
            urgency: Urgency of the message (0-1)
            
        Returns:
            Interruption decision with reasoning
        """
        current_cost = self.current_state['interruption_cost']
        fatigue = self.current_state['fatigue']
        focus = self.current_state['focus']
        
        # Decision logic
        should_interrupt = (
            urgency > current_cost or  # High urgency overrides cost
            (urgency > 0.3 and current_cost < 0.3) or  # Medium urgency, low cost
            fatigue > 0.8  # Very fatigued - might need encouragement
        )
        
        return {
            'should_interrupt': should_interrupt,
            'interruption_cost': current_cost,
            'urgency_threshold': urgency,
            'reasoning': self._get_interruption_reasoning(urgency, current_cost, fatigue, focus),
            'cognitive_state': self.current_state.copy()
        }
    
    def _get_interruption_reasoning(
        self, 
        urgency: float, 
        cost: float, 
        fatigue: float, 
        focus: float
    ) -> str:
        """Generate human-readable reasoning for interruption decision"""
        
        if urgency > cost:
            return f"High urgency ({urgency:.2f}) overrides interruption cost ({cost:.2f})"
        elif fatigue > 0.8:
            return f"User is very fatigued ({fatigue:.2f}) - may need encouragement"
        elif cost > 0.7:
            return f"High interruption cost ({cost:.2f}) - user appears focused"
        elif focus < 0.3:
            return f"Low focus detected ({focus:.2f}) - good time for guidance"
        else:
            return f"Balanced state - urgency {urgency:.2f}, cost {cost:.2f}"


def main():
    """Test cognitive state modulator"""
    
    # Create processor
    processor = CognitiveStateProcessor()
    
    # Simulate real-time updates
    print("Testing Cognitive State Modulator:")
    print("=" * 40)
    
    for i in range(10):
        # Simulate IMU data (increasing jitter over time to simulate fatigue)
        imu_data = np.random.randn(100, 6) * (1 + i * 0.1)  # Increasing noise
        
        # Simulate audio data
        audio_data = np.random.randn(1600)  # 0.1 second at 16kHz
        
        # Update state
        state = processor.update_state(imu_data, audio_data)
        
        # Test interruption decision
        interruption = processor.should_interrupt(urgency=0.5)
        
        print(f"Update {i+1}:")
        print(f"  Fatigue: {state['fatigue']:.3f}")
        print(f"  Focus: {state['focus']:.3f}")
        print(f"  Interruption Cost: {state['interruption_cost']:.3f}")
        print(f"  Should Interrupt: {interruption['should_interrupt']}")
        print(f"  Reasoning: {interruption['reasoning']}")
        print("-" * 30)
        
        time.sleep(0.1)  # Simulate real-time processing


if __name__ == "__main__":
    main()
