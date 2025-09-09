#!/usr/bin/env python3
"""
Sesame v2 Audio Pipeline - PANNs Sound Classification
On-device audio event detection for [speech_started], [heavy_breathing], [loud_impact_noise]
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional
import librosa
from pathlib import Path
import json


class PANNsAudioClassifier(nn.Module):
    """
    Lightweight PANNs-based audio classifier for fitness coaching events
    Optimized for on-device inference with Core ML conversion
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 320,  # 20ms at 16kHz
        window_size: float = 1.0,  # 1 second analysis windows
        num_classes: int = 6  # speech_start, speech_end, heavy_breathing, impact_noise, ambient, silence
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size
        self.num_classes = num_classes
        
        # Calculate input dimensions
        self.window_samples = int(window_size * sample_rate)
        self.n_frames = self.window_samples // hop_length
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        
        # Log transform
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # CNN backbone (MobileNet-inspired for efficiency)
        self.backbone = self._build_efficient_backbone()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        # Class labels
        self.class_labels = [
            'speech_started',
            'speech_ended', 
            'heavy_breathing',
            'loud_impact_noise',
            'ambient_noise',
            'silence'
        ]
    
    def _build_efficient_backbone(self) -> nn.Module:
        """Build efficient CNN backbone for audio classification"""
        
        def depthwise_separable_conv(in_channels, out_channels, stride=1):
            return nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                
                # Pointwise convolution
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        
        return nn.Sequential(
            # Initial conv
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable blocks
            depthwise_separable_conv(32, 64),
            depthwise_separable_conv(64, 64, stride=2),
            
            depthwise_separable_conv(64, 128),
            depthwise_separable_conv(128, 128, stride=2),
            
            depthwise_separable_conv(128, 256),
            depthwise_separable_conv(256, 256, stride=2),
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for audio classification
        
        Args:
            audio: (batch_size, samples) - Raw audio waveform
            
        Returns:
            logits: (batch_size, num_classes) - Classification logits
        """
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(audio)  # (B, n_mels, time)
        mel_spec = self.amplitude_to_db(mel_spec)
        
        # Add channel dimension for CNN
        mel_spec = mel_spec.unsqueeze(1)  # (B, 1, n_mels, time)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # CNN backbone
        features = self.backbone(mel_spec)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def predict_events(self, audio: torch.Tensor, threshold: float = 0.5) -> List[Dict[str, float]]:
        """
        Predict audio events with confidence scores
        
        Args:
            audio: Raw audio tensor
            threshold: Confidence threshold for event detection
            
        Returns:
            List of detected events with timestamps and confidence
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(audio)
            probabilities = torch.softmax(logits, dim=-1)
            
            events = []
            for i, prob in enumerate(probabilities):
                for class_idx, confidence in enumerate(prob):
                    if confidence > threshold:
                        events.append({
                            'event_type': self.class_labels[class_idx],
                            'confidence': float(confidence),
                            'timestamp': i * self.window_size  # Approximate timestamp
                        })
            
            return events
    
    def get_model_size(self) -> Dict[str, int]:
        """Calculate model size for deployment planning"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Float32
        }


class AudioEventDetector:
    """
    Real-time audio event detection pipeline
    Processes streaming audio and detects coaching-relevant events
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        buffer_duration: float = 1.0,
        overlap: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.overlap = overlap
        
        # Buffer management
        self.buffer_size = int(buffer_duration * sample_rate)
        self.hop_size = int(self.buffer_size * (1 - overlap))
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        # Load model
        if model_path and Path(model_path).exists():
            self.model = self._load_model(model_path)
        else:
            self.model = PANNsAudioClassifier()
        
        self.model.eval()
        
        # Event history for temporal smoothing
        self.event_history = []
        self.history_length = 5  # Keep last 5 predictions
        
        print(f"AudioEventDetector initialized with {self.model.get_model_size()['model_size_mb']:.1f}MB model")
    
    def _load_model(self, model_path: str) -> PANNsAudioClassifier:
        """Load pre-trained model"""
        model = PANNsAudioClassifier()
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> List[Dict[str, float]]:
        """
        Process incoming audio chunk and detect events
        
        Args:
            audio_chunk: New audio samples
            
        Returns:
            List of detected events
        """
        # Update buffer with new audio
        self._update_buffer(audio_chunk)
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(self.audio_buffer).unsqueeze(0)
        
        # Detect events
        events = self.model.predict_events(audio_tensor, threshold=0.6)
        
        # Apply temporal smoothing
        smoothed_events = self._apply_temporal_smoothing(events)
        
        return smoothed_events
    
    def _update_buffer(self, new_audio: np.ndarray):
        """Update circular audio buffer"""
        if len(new_audio) >= self.buffer_size:
            # New audio is longer than buffer, take the end
            self.audio_buffer = new_audio[-self.buffer_size:]
        else:
            # Shift buffer and append new audio
            shift_amount = len(new_audio)
            self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
            self.audio_buffer[-shift_amount:] = new_audio
    
    def _apply_temporal_smoothing(self, events: List[Dict]) -> List[Dict]:
        """
        Apply temporal smoothing to reduce false positives
        Only report events that are consistent across multiple frames
        """
        # Add current events to history
        self.event_history.append(events)
        
        # Keep only recent history
        if len(self.event_history) > self.history_length:
            self.event_history.pop(0)
        
        # Count event occurrences
        event_counts = {}
        for frame_events in self.event_history:
            for event in frame_events:
                event_type = event['event_type']
                if event_type not in event_counts:
                    event_counts[event_type] = []
                event_counts[event_type].append(event['confidence'])
        
        # Filter events that appear consistently
        smoothed_events = []
        min_occurrences = max(1, len(self.event_history) // 2)  # Must appear in at least half of recent frames
        
        for event_type, confidences in event_counts.items():
            if len(confidences) >= min_occurrences:
                avg_confidence = np.mean(confidences)
                if avg_confidence > 0.5:  # High confidence threshold after smoothing
                    smoothed_events.append({
                        'event_type': event_type,
                        'confidence': avg_confidence,
                        'timestamp': 0.0  # Current time
                    })
        
        return smoothed_events


class AudioEventTrainer:
    """
    Training pipeline for PANNs audio classifier
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = PANNsAudioClassifier(**config['model'])
        
    def create_synthetic_training_data(self, output_dir: str, num_samples: int = 10000):
        """
        Create synthetic training data for audio events
        Uses audio synthesis and augmentation techniques
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # This would be implemented with audio synthesis libraries
        # For now, create placeholder structure
        
        training_data = {
            'speech_started': [],
            'speech_ended': [],
            'heavy_breathing': [],
            'loud_impact_noise': [],
            'ambient_noise': [],
            'silence': []
        }
        
        # Generate synthetic samples for each class
        for class_name in training_data.keys():
            for i in range(num_samples // 6):  # Equal samples per class
                # Placeholder for synthetic audio generation
                sample_data = {
                    'audio_path': f"{output_path}/{class_name}_{i:05d}.wav",
                    'label': class_name,
                    'duration': 1.0,
                    'sample_rate': 16000
                }
                training_data[class_name].append(sample_data)
        
        # Save training manifest
        with open(output_path / 'training_manifest.json', 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Created training data structure at {output_path}")
        return training_data


def main():
    """Test audio event detection"""
    
    # Create model
    detector = AudioEventDetector()
    
    # Test with random audio (placeholder)
    sample_rate = 16000
    duration = 2.0
    test_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    
    # Process in chunks
    chunk_size = int(0.1 * sample_rate)  # 100ms chunks
    
    print("Testing audio event detection...")
    for i in range(0, len(test_audio), chunk_size):
        chunk = test_audio[i:i+chunk_size]
        events = detector.process_audio_chunk(chunk)
        
        if events:
            print(f"Time {i/sample_rate:.1f}s: {events}")
    
    print(f"Model size: {detector.model.get_model_size()}")


if __name__ == "__main__":
    main()
