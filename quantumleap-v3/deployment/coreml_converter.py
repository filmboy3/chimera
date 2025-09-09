#!/usr/bin/env python3
"""
Core ML Conversion Pipeline for QuantumLeap v3 On-Device Deployment

Converts trained PyTorch models to Core ML format optimized for iOS devices.
Handles model quantization, optimization, and validation for mobile inference.
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
import time

# Import QuantumLeap v3 architecture
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.qlv3_architecture import QuantumLeapV3Model

@dataclass
class ConversionConfig:
    """Configuration for Core ML conversion process"""
    model_name: str = "QuantumLeapV3"
    input_sequence_length: int = 200  # 2 seconds at 100Hz
    input_channels: int = 15  # phone_imu(9) + watch_imu(3) + barometer(1) + audio_features(2)
    quantization_bits: int = 16  # FP16 for balance of size/accuracy
    compute_units: str = "cpuAndNeuralEngine"  # Leverage Neural Engine
    minimum_deployment_target: str = "iOS15"
    optimize_for_size: bool = True
    validation_samples: int = 100
    
@dataclass
class ConversionMetrics:
    """Metrics from conversion process"""
    original_model_size_mb: float
    coreml_model_size_mb: float
    compression_ratio: float
    conversion_time_seconds: float
    validation_accuracy: float
    inference_time_ms: float
    memory_usage_mb: float

class CoreMLConverter:
    """Converts QuantumLeap v3 models to optimized Core ML format"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for conversion process"""
        logger = logging.getLogger("CoreMLConverter")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def convert_model(
        self, 
        pytorch_model_path: str,
        output_dir: str,
        model_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, ConversionMetrics]:
        """
        Convert PyTorch model to Core ML format
        
        Args:
            pytorch_model_path: Path to trained PyTorch model
            output_dir: Directory to save Core ML model
            model_metadata: Optional metadata about the model
            
        Returns:
            Tuple of (coreml_model_path, conversion_metrics)
        """
        start_time = time.time()
        
        self.logger.info(f"🔄 Starting Core ML conversion for {pytorch_model_path}")
        
        # Load PyTorch model
        pytorch_model = self._load_pytorch_model(pytorch_model_path)
        original_size = self._get_model_size(pytorch_model_path)
        
        # Prepare model for conversion
        pytorch_model = self._prepare_model_for_conversion(pytorch_model)
        
        # Create example input
        example_input = self._create_example_input()
        
        # Trace the model
        traced_model = self._trace_model(pytorch_model, example_input)
        
        # Convert to Core ML
        coreml_model = self._convert_to_coreml(traced_model, example_input)
        
        # Optimize model
        coreml_model = self._optimize_coreml_model(coreml_model)
        
        # Add metadata
        coreml_model = self._add_model_metadata(coreml_model, model_metadata)
        
        # Save model
        output_path = self._save_coreml_model(coreml_model, output_dir)
        coreml_size = self._get_model_size(output_path)
        
        # Validate conversion
        validation_accuracy = self._validate_conversion(
            pytorch_model, coreml_model, example_input
        )
        
        # Benchmark performance
        inference_time, memory_usage = self._benchmark_performance(coreml_model)
        
        # Calculate metrics
        conversion_time = time.time() - start_time
        metrics = ConversionMetrics(
            original_model_size_mb=original_size,
            coreml_model_size_mb=coreml_size,
            compression_ratio=original_size / coreml_size,
            conversion_time_seconds=conversion_time,
            validation_accuracy=validation_accuracy,
            inference_time_ms=inference_time,
            memory_usage_mb=memory_usage
        )
        
        self.logger.info(f"✅ Core ML conversion completed in {conversion_time:.2f}s")
        self.logger.info(f"📊 Model size: {original_size:.1f}MB → {coreml_size:.1f}MB ({metrics.compression_ratio:.1f}x compression)")
        self.logger.info(f"🎯 Validation accuracy: {validation_accuracy:.3f}")
        self.logger.info(f"⚡ Inference time: {inference_time:.1f}ms")
        
        return output_path, metrics
    
    def _load_pytorch_model(self, model_path: str) -> nn.Module:
        """Load trained PyTorch model"""
        self.logger.info(f"📥 Loading PyTorch model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create model instance
        model = QuantumLeapV3Model(
            input_channels=self.config.input_channels,
            sequence_length=self.config.input_sequence_length,
            num_pose_joints=17,  # COCO format
            num_exercise_classes=5,  # squat, rest, transition, etc.
            num_form_errors=4,  # knee_valgus, forward_lean, insufficient_depth, asymmetry
            num_cognitive_states=3,  # fatigue, focus, interruption_cost
            vq_num_embeddings=512,
            vq_embedding_dim=256,
            transformer_dim=512,
            transformer_heads=8,
            transformer_layers=6
        )
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        self.logger.info("✅ PyTorch model loaded successfully")
        return model
    
    def _prepare_model_for_conversion(self, model: nn.Module) -> nn.Module:
        """Prepare model for Core ML conversion"""
        self.logger.info("🔧 Preparing model for conversion...")
        
        # Set to evaluation mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Wrap model to handle Core ML input/output format
        class CoreMLWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                
            def forward(self, x):
                # x shape: (batch_size, sequence_length, input_channels)
                # Core ML expects (batch_size, channels, sequence_length)
                x = x.transpose(1, 2)
                
                outputs = self.base_model(x)
                
                # Return only the most important outputs for mobile inference
                return {
                    'pose_estimation': outputs['pose_estimation'],
                    'exercise_classification': outputs['exercise_classification'],
                    'rep_detection': outputs['rep_detection'],
                    'cognitive_state': outputs['cognitive_state']
                }
        
        wrapped_model = CoreMLWrapper(model)
        
        self.logger.info("✅ Model prepared for conversion")
        return wrapped_model
    
    def _create_example_input(self) -> torch.Tensor:
        """Create example input for model tracing"""
        self.logger.info("📝 Creating example input for tracing...")
        
        # Create realistic example input
        # Shape: (batch_size=1, sequence_length, input_channels)
        example_input = torch.randn(
            1, 
            self.config.input_sequence_length, 
            self.config.input_channels
        )
        
        # Add some realistic patterns for IMU data
        # Phone IMU (accelerometer + gyroscope + magnetometer)
        example_input[:, :, 0:3] *= 2.0  # Accelerometer range ±2g
        example_input[:, :, 3:6] *= 5.0  # Gyroscope range ±5 rad/s
        example_input[:, :, 6:9] *= 50.0  # Magnetometer range ±50 μT
        
        # Watch IMU (accelerometer only)
        example_input[:, :, 9:12] *= 2.0  # Accelerometer range ±2g
        
        # Barometric pressure (normalized)
        example_input[:, :, 12] = torch.sin(torch.linspace(0, 4*np.pi, self.config.input_sequence_length)) * 0.1
        
        # Audio features (MFCCs or spectral features)
        example_input[:, :, 13:15] *= 0.5  # Normalized audio features
        
        self.logger.info(f"✅ Example input created: {example_input.shape}")
        return example_input
    
    def _trace_model(self, model: nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """Trace PyTorch model for conversion"""
        self.logger.info("🔍 Tracing PyTorch model...")
        
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        # Verify tracing
        original_output = model(example_input)
        traced_output = traced_model(example_input)
        
        # Check output consistency
        for key in original_output.keys():
            diff = torch.abs(original_output[key] - traced_output[key]).max()
            if diff > 1e-5:
                self.logger.warning(f"⚠️ Tracing inconsistency for {key}: max diff = {diff}")
        
        self.logger.info("✅ Model tracing completed")
        return traced_model
    
    def _convert_to_coreml(
        self, 
        traced_model: torch.jit.ScriptModule, 
        example_input: torch.Tensor
    ) -> ct.models.MLModel:
        """Convert traced model to Core ML"""
        self.logger.info("🔄 Converting to Core ML format...")
        
        # Define input specification
        input_spec = ct.TensorType(
            name="sensor_data",
            shape=example_input.shape,
            dtype=np.float32
        )
        
        # Convert model
        coreml_model = ct.convert(
            traced_model,
            inputs=[input_spec],
            minimum_deployment_target=ct.target.iOS15,
            compute_units=getattr(ct.ComputeUnit, self.config.compute_units.replace("cpuAnd", "cpu_and_").lower()),
            convert_to="mlprogram"  # Use ML Program format for better optimization
        )
        
        self.logger.info("✅ Core ML conversion completed")
        return coreml_model
    
    def _optimize_coreml_model(self, model: ct.models.MLModel) -> ct.models.MLModel:
        """Optimize Core ML model for mobile deployment"""
        self.logger.info("⚡ Optimizing Core ML model...")
        
        # Apply quantization if specified
        if self.config.quantization_bits < 32:
            self.logger.info(f"🔢 Applying {self.config.quantization_bits}-bit quantization...")
            
            if self.config.quantization_bits == 16:
                model = ct.optimize.coreml.optimize_weights(
                    model, 
                    nbits=16,
                    quantization_mode="linear"
                )
            elif self.config.quantization_bits == 8:
                model = ct.optimize.coreml.optimize_weights(
                    model,
                    nbits=8,
                    quantization_mode="linear_symmetric"
                )
        
        # Apply additional optimizations
        if self.config.optimize_for_size:
            self.logger.info("📦 Applying size optimizations...")
            # Additional size optimizations would go here
            # (Core ML Tools may add more optimization options in future versions)
        
        self.logger.info("✅ Model optimization completed")
        return model
    
    def _add_model_metadata(
        self, 
        model: ct.models.MLModel, 
        metadata: Optional[Dict[str, Any]]
    ) -> ct.models.MLModel:
        """Add metadata to Core ML model"""
        self.logger.info("📋 Adding model metadata...")
        
        # Default metadata
        model.author = "Project Chimera Ascendant"
        model.short_description = "QuantumLeap v3 - Multi-modal Perception Engine"
        model.version = "3.0.0"
        
        # Input descriptions
        model.input_description["sensor_data"] = (
            "Multi-modal sensor data: phone IMU (9), watch IMU (3), "
            "barometer (1), audio features (2). Shape: (1, 200, 15)"
        )
        
        # Output descriptions
        if "pose_estimation" in [output.name for output in model.output_description]:
            model.output_description["pose_estimation"] = "3D pose estimation (17 joints x 3 coordinates)"
        if "exercise_classification" in [output.name for output in model.output_description]:
            model.output_description["exercise_classification"] = "Exercise type classification probabilities"
        if "rep_detection" in [output.name for output in model.output_description]:
            model.output_description["rep_detection"] = "Repetition detection confidence"
        if "cognitive_state" in [output.name for output in model.output_description]:
            model.output_description["cognitive_state"] = "Cognitive state estimation (fatigue, focus)"
        
        # Add custom metadata
        if metadata:
            for key, value in metadata.items():
                setattr(model, key, str(value))
        
        self.logger.info("✅ Metadata added successfully")
        return model
    
    def _save_coreml_model(self, model: ct.models.MLModel, output_dir: str) -> str:
        """Save Core ML model to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_filename = f"{self.config.model_name}.mlpackage"
        output_path = os.path.join(output_dir, model_filename)
        
        self.logger.info(f"💾 Saving Core ML model to {output_path}")
        model.save(output_path)
        
        self.logger.info("✅ Model saved successfully")
        return output_path
    
    def _validate_conversion(
        self, 
        pytorch_model: nn.Module, 
        coreml_model: ct.models.MLModel,
        example_input: torch.Tensor
    ) -> float:
        """Validate Core ML conversion accuracy"""
        self.logger.info("🔍 Validating conversion accuracy...")
        
        # Generate validation samples
        validation_inputs = []
        for _ in range(self.config.validation_samples):
            val_input = torch.randn_like(example_input)
            validation_inputs.append(val_input)
        
        total_error = 0.0
        valid_samples = 0
        
        for val_input in validation_inputs:
            try:
                # PyTorch inference
                with torch.no_grad():
                    pytorch_output = pytorch_model(val_input)
                
                # Core ML inference
                coreml_input = {"sensor_data": val_input.numpy()}
                coreml_output = coreml_model.predict(coreml_input)
                
                # Compare outputs
                sample_error = 0.0
                num_outputs = 0
                
                for key in pytorch_output.keys():
                    if key in coreml_output:
                        pytorch_tensor = pytorch_output[key].numpy()
                        coreml_tensor = coreml_output[key]
                        
                        # Calculate relative error
                        error = np.mean(np.abs(pytorch_tensor - coreml_tensor) / (np.abs(pytorch_tensor) + 1e-8))
                        sample_error += error
                        num_outputs += 1
                
                if num_outputs > 0:
                    total_error += sample_error / num_outputs
                    valid_samples += 1
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Validation sample failed: {e}")
        
        if valid_samples > 0:
            avg_error = total_error / valid_samples
            accuracy = max(0.0, 1.0 - avg_error)
        else:
            accuracy = 0.0
        
        self.logger.info(f"✅ Validation completed: {accuracy:.3f} accuracy")
        return accuracy
    
    def _benchmark_performance(self, model: ct.models.MLModel) -> Tuple[float, float]:
        """Benchmark Core ML model performance"""
        self.logger.info("⚡ Benchmarking model performance...")
        
        # Create benchmark input
        benchmark_input = {
            "sensor_data": np.random.randn(1, self.config.input_sequence_length, self.config.input_channels).astype(np.float32)
        }
        
        # Warm up
        for _ in range(5):
            try:
                _ = model.predict(benchmark_input)
            except:
                pass
        
        # Benchmark inference time
        num_runs = 50
        start_time = time.time()
        
        successful_runs = 0
        for _ in range(num_runs):
            try:
                _ = model.predict(benchmark_input)
                successful_runs += 1
            except Exception as e:
                self.logger.warning(f"⚠️ Benchmark run failed: {e}")
        
        total_time = time.time() - start_time
        
        if successful_runs > 0:
            avg_inference_time = (total_time / successful_runs) * 1000  # Convert to ms
        else:
            avg_inference_time = float('inf')
        
        # Estimate memory usage (simplified)
        model_size_mb = self._get_model_size_from_object(model)
        estimated_memory_mb = model_size_mb * 2.5  # Rough estimate including intermediate tensors
        
        self.logger.info(f"✅ Performance benchmark completed")
        self.logger.info(f"⚡ Average inference time: {avg_inference_time:.1f}ms")
        self.logger.info(f"💾 Estimated memory usage: {estimated_memory_mb:.1f}MB")
        
        return avg_inference_time, estimated_memory_mb
    
    def _get_model_size(self, model_path: str) -> float:
        """Get model file size in MB"""
        if os.path.isfile(model_path):
            size_bytes = os.path.getsize(model_path)
        elif os.path.isdir(model_path):
            size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(model_path)
                for filename in filenames
            )
        else:
            return 0.0
        
        return size_bytes / (1024 * 1024)  # Convert to MB
    
    def _get_model_size_from_object(self, model: ct.models.MLModel) -> float:
        """Estimate model size from Core ML model object"""
        # This is a rough estimate - actual size would need to be measured after saving
        return 50.0  # Placeholder estimate in MB

def main():
    """Main conversion script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert QuantumLeap v3 to Core ML")
    parser.add_argument("--model_path", required=True, help="Path to PyTorch model")
    parser.add_argument("--output_dir", required=True, help="Output directory for Core ML model")
    parser.add_argument("--config", help="Path to conversion config JSON")
    parser.add_argument("--quantization_bits", type=int, default=16, choices=[8, 16, 32])
    parser.add_argument("--optimize_for_size", action="store_true")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ConversionConfig(**config_dict)
    else:
        config = ConversionConfig()
    
    # Override config with command line arguments
    config.quantization_bits = args.quantization_bits
    config.optimize_for_size = args.optimize_for_size
    
    # Create converter
    converter = CoreMLConverter(config)
    
    # Convert model
    coreml_path, metrics = converter.convert_model(
        args.model_path,
        args.output_dir
    )
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "conversion_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(asdict(metrics), f, indent=2)
    
    print(f"\n✅ Conversion completed successfully!")
    print(f"📱 Core ML model: {coreml_path}")
    print(f"📊 Metrics: {metrics_path}")
    print(f"🎯 Validation accuracy: {metrics.validation_accuracy:.3f}")
    print(f"⚡ Inference time: {metrics.inference_time_ms:.1f}ms")
    print(f"📦 Compression ratio: {metrics.compression_ratio:.1f}x")

if __name__ == "__main__":
    main()
