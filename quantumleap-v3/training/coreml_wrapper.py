#!/usr/bin/env python3
"""
Core ML Wrapper for QuantumLeap v3 Model
Provides dict-based outputs for training and tuple outputs for Core ML conversion
"""

import torch
import torch.nn as nn
from simple_trainer import SimpleQuantumLeapV3

class QuantumLeapV3ForTraining(SimpleQuantumLeapV3):
    """Training version that returns dict outputs"""
    
    def forward(self, x):
        rep_detection, exercise_class, form_quality, cognitive_state = super().forward(x)
        
        return {
            'rep_detection': rep_detection,
            'exercise_class': exercise_class,
            'form_quality': form_quality,
            'cognitive_state': cognitive_state
        }

class QuantumLeapV3ForCoreML(SimpleQuantumLeapV3):
    """Core ML version that returns tuple outputs (for tracing)"""
    
    def forward(self, x):
        # Returns tuple directly from parent class
        return super().forward(x)

def convert_model_for_coreml(training_model_path):
    """Convert training model to Core ML compatible format"""
    
    # Load training model weights
    training_model = QuantumLeapV3ForTraining(input_channels=13, hidden_dim=128, num_layers=4)
    training_model.load_state_dict(torch.load(training_model_path, map_location='cpu'))
    
    # Create Core ML compatible model
    coreml_model = QuantumLeapV3ForCoreML(input_channels=13, hidden_dim=128, num_layers=4)
    
    # Copy weights from training model
    coreml_model.load_state_dict(training_model.state_dict())
    coreml_model.eval()
    
    return coreml_model
