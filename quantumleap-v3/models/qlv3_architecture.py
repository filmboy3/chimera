#!/usr/bin/env python3
"""
QuantumLeap v3 Model Architecture
CNN → VQ-VAE → Transformer pipeline for multi-modal pose estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
from vector_quantize_pytorch import VectorQuantize


class MultiModalCNN(nn.Module):
    """
    1D CNN feature encoder for multi-modal sensor data
    Processes IMU (phone + watch) + barometer streams
    """
    
    def __init__(
        self,
        input_channels: int = 13,  # 6 (phone IMU) + 6 (watch IMU) + 1 (barometer)
        channels: list = [64, 128, 256],
        kernel_size: int = 7,
        stride: int = 2
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.channels = channels
        
        # Build CNN layers
        layers = []
        in_channels = input_channels
        
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        
        # Calculate output dimension after convolutions
        self.output_dim = channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_channels)
        Returns:
            features: (batch_size, output_dim, reduced_sequence_length)
        """
        # Transpose for 1D conv: (B, C, L)
        x = x.transpose(1, 2)
        features = self.cnn(x)
        return features


class MotionVectorQuantizer(nn.Module):
    """
    VQ-VAE module for motion primitive quantization
    Converts continuous features to discrete motion tokens
    """
    
    def __init__(
        self,
        dim: int = 256,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5
    ):
        super().__init__()
        
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        
        # Vector quantization layer
        self.vq = VectorQuantize(
            dim=dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_cost,
            eps=eps,
            use_cosine_sim=True,  # Better for motion data
            threshold_ema_dead_code=2  # Reset unused codes
        )
        
        # Projection layers
        self.pre_vq = nn.Linear(dim, dim)
        self.post_vq = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (batch_size, dim, sequence_length)
        Returns:
            quantized: (batch_size, dim, sequence_length)
            vq_info: Dictionary with loss components and perplexity
        """
        # Transpose and reshape for VQ: (B*L, D)
        B, D, L = x.shape
        x = x.transpose(1, 2).contiguous()  # (B, L, D)
        x = x.view(-1, D)  # (B*L, D)
        
        # Pre-quantization projection
        x = self.pre_vq(x)
        
        # Vector quantization
        quantized, indices, commit_loss = self.vq(x)
        
        # Post-quantization projection
        quantized = self.post_vq(quantized)
        
        # Reshape back
        quantized = quantized.view(B, L, D).transpose(1, 2)  # (B, D, L)
        
        # Calculate perplexity for monitoring codebook usage
        avg_probs = torch.bincount(indices, minlength=self.codebook_size).float()
        avg_probs = avg_probs / avg_probs.sum()
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        vq_info = {
            'commit_loss': commit_loss,
            'perplexity': perplexity,
            'indices': indices.view(B, L)
        }
        
        return quantized, vq_info


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MotionTransformer(nn.Module):
    """
    Transformer encoder for learning motion grammar
    Processes quantized motion tokens to understand exercise patterns
    """
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_sequence_length: int = 1000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, max_sequence_length)
        
        # Input projection to match transformer dimension
        self.input_projection = nn.Linear(256, d_model)  # From VQ-VAE output
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, dim)
            mask: Optional attention mask
        Returns:
            encoded: (batch_size, sequence_length, d_model)
        """
        # Project to transformer dimension
        x = self.input_projection(x)
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        x = self.dropout(x)
        
        # Apply transformer
        encoded = self.transformer(x, src_key_padding_mask=mask)
        
        return encoded


class MultiTaskDecoder(nn.Module):
    """
    Multi-task decoder for pose estimation, classification, and cognitive state
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_joints: int = 4,  # hip_right, hip_left, knee_right, knee_left
        num_exercises: int = 1,  # Just squats for PoC
        num_form_errors: int = 3,  # knee_valgus, forward_lean, insufficient_depth
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Pose regression head (probabilistic)
        self.pose_mean = nn.Linear(hidden_dim, num_joints)
        self.pose_logvar = nn.Linear(hidden_dim, num_joints)
        
        # Exercise classification head
        self.exercise_classifier = nn.Linear(hidden_dim, num_exercises)
        
        # Form error detection heads
        self.form_error_detectors = nn.ModuleDict({
            'knee_valgus': nn.Linear(hidden_dim, 1),
            'forward_lean': nn.Linear(hidden_dim, 1),
            'insufficient_depth': nn.Linear(hidden_dim, 1)
        })
        
        # Cognitive state estimation (fatigue, focus)
        self.cognitive_state = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # fatigue, focus scores
        )
        
        # Rep detection head
        self.rep_detector = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch_size, sequence_length, d_model)
        Returns:
            outputs: Dictionary of task-specific predictions
        """
        # Shared feature extraction
        features = self.shared_layers(x)
        
        # Pose estimation (probabilistic)
        pose_mean = self.pose_mean(features)
        pose_logvar = self.pose_logvar(features)
        pose_std = torch.exp(0.5 * pose_logvar)
        
        # Exercise classification
        exercise_logits = self.exercise_classifier(features)
        
        # Form error detection
        form_errors = {}
        for error_type, detector in self.form_error_detectors.items():
            form_errors[error_type] = torch.sigmoid(detector(features))
        
        # Cognitive state
        cognitive_state = torch.sigmoid(self.cognitive_state(features))
        
        # Rep detection
        rep_probability = torch.sigmoid(self.rep_detector(features))
        
        return {
            'pose_mean': pose_mean,
            'pose_std': pose_std,
            'pose_logvar': pose_logvar,
            'exercise_logits': exercise_logits,
            'form_errors': form_errors,
            'cognitive_state': cognitive_state,  # [fatigue, focus]
            'rep_probability': rep_probability
        }


class QuantumLeapV3(nn.Module):
    """
    Complete QuantumLeap v3 architecture
    Multi-modal perception engine for embodied AI coaching
    """
    
    def __init__(
        self,
        input_channels: int = 13,
        cnn_channels: list = [64, 128, 256],
        vq_codebook_size: int = 512,
        vq_commitment_cost: float = 0.25,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        transformer_dim: int = 512,
        num_joints: int = 4,
        num_exercises: int = 1,
        num_form_errors: int = 3
    ):
        super().__init__()
        
        # Feature extraction
        self.cnn = MultiModalCNN(
            input_channels=input_channels,
            channels=cnn_channels
        )
        
        # Motion quantization
        self.vq = MotionVectorQuantizer(
            dim=cnn_channels[-1],
            codebook_size=vq_codebook_size,
            commitment_cost=vq_commitment_cost
        )
        
        # Contextual encoding
        self.transformer = MotionTransformer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers
        )
        
        # Multi-task decoding
        self.decoder = MultiTaskDecoder(
            d_model=transformer_dim,
            num_joints=num_joints,
            num_exercises=num_exercises,
            num_form_errors=num_form_errors
        )
        
        # Store config for logging
        self.config = {
            'input_channels': input_channels,
            'cnn_channels': cnn_channels,
            'vq_codebook_size': vq_codebook_size,
            'vq_commitment_cost': vq_commitment_cost,
            'transformer_layers': transformer_layers,
            'transformer_heads': transformer_heads,
            'transformer_dim': transformer_dim
        }
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through complete QLv3 pipeline
        
        Args:
            x: (batch_size, sequence_length, input_channels)
            mask: Optional padding mask
            
        Returns:
            predictions: Dictionary of task outputs
            aux_info: Dictionary with VQ info and intermediate features
        """
        # CNN feature extraction
        cnn_features = self.cnn(x)  # (B, C, L)
        
        # Vector quantization
        vq_features, vq_info = self.vq(cnn_features)  # (B, C, L)
        
        # Prepare for transformer (B, L, C)
        vq_features = vq_features.transpose(1, 2)
        
        # Transformer encoding
        encoded = self.transformer(vq_features, mask=mask)  # (B, L, D)
        
        # Multi-task decoding
        predictions = self.decoder(encoded)
        
        # Auxiliary information for training
        aux_info = {
            'vq_commit_loss': vq_info['commit_loss'],
            'vq_perplexity': vq_info['perplexity'],
            'vq_indices': vq_info['indices'],
            'cnn_features': cnn_features,
            'encoded_features': encoded
        }
        
        return predictions, aux_info
    
    def get_model_size(self) -> Dict[str, int]:
        """Calculate model size statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def create_qlv3_model(config: Dict) -> QuantumLeapV3:
    """Factory function to create QLv3 model from config"""
    
    model = QuantumLeapV3(
        input_channels=config.get('input_channels', 13),
        cnn_channels=config['model']['architecture']['cnn_channels'],
        vq_codebook_size=config['model']['architecture']['vq_codebook_size'],
        vq_commitment_cost=config['model']['architecture']['vq_commitment_cost'],
        transformer_layers=config['model']['architecture']['transformer_layers'],
        transformer_heads=config['model']['architecture']['transformer_heads'],
        transformer_dim=config['model']['architecture']['transformer_dim']
    )
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    import yaml
    
    # Load config
    config = {
        'model': {
            'architecture': {
                'cnn_channels': [64, 128, 256],
                'vq_codebook_size': 512,
                'vq_commitment_cost': 0.25,
                'transformer_layers': 6,
                'transformer_heads': 8,
                'transformer_dim': 512
            }
        }
    }
    
    # Create model
    model = create_qlv3_model(config)
    
    # Test forward pass
    batch_size, seq_len, input_dim = 4, 100, 13
    x = torch.randn(batch_size, seq_len, input_dim)
    
    predictions, aux_info = model(x)
    
    print("QuantumLeap v3 Model Test:")
    print(f"Model size: {model.get_model_size()}")
    print(f"Input shape: {x.shape}")
    print(f"Pose prediction shape: {predictions['pose_mean'].shape}")
    print(f"VQ perplexity: {aux_info['vq_perplexity'].item():.2f}")
    print("✅ Model architecture test passed!")
