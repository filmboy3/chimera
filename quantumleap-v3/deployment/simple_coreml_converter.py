#!/usr/bin/env python3
"""
Simple Core ML Converter for QuantumLeap v3
Converts trained PyTorch model to Core ML format for iOS deployment
"""

import torch
import coremltools as ct
import numpy as np
import argparse
from pathlib import Path
import logging
import sys

# Add path for model import
sys.path.append('../training')
from coreml_wrapper import convert_model_for_coreml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoreMLConverter:
    """Convert QuantumLeap v3 PyTorch model to Core ML"""
    
    def __init__(self, model_path, output_dir="./"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load PyTorch model (Core ML compatible version)
        self.model = convert_model_for_coreml(model_path)
        
        logger.info(f"Loaded PyTorch model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def convert_to_coreml(self):
        """Convert PyTorch model to Core ML format"""
        
        # Create example input for tracing
        # Input shape: (batch_size=1, channels=13, sequence_length=200)
        example_input = torch.randn(1, 13, 200)
        
        logger.info("Tracing PyTorch model...")
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, example_input)
        
        logger.info("Converting to Core ML...")
        
        # Convert to Core ML
        coreml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="sensor_data",
                    shape=example_input.shape,
                    dtype=np.float32
                )
            ],
            outputs=[
                ct.TensorType(name="rep_detection", dtype=np.float32),
                ct.TensorType(name="exercise_class", dtype=np.float32),
                ct.TensorType(name="form_quality", dtype=np.float32),
                ct.TensorType(name="cognitive_state", dtype=np.float32)
            ],
            minimum_deployment_target=ct.target.iOS15,
            compute_precision=ct.precision.FLOAT16  # Optimize for mobile
        )
        
        # Add metadata
        coreml_model.author = "Project Chimera Ascendant"
        coreml_model.short_description = "QuantumLeap v3 - Multi-modal Exercise Recognition"
        coreml_model.version = "1.0.0"
        
        # Add input description
        coreml_model.input_description["sensor_data"] = (
            "Multi-sensor input: phone IMU (6 channels) + watch IMU (6 channels) + barometer (1 channel). "
            "Shape: (1, 13, 200) representing 2 seconds at 100Hz sampling rate."
        )
        
        # Add output descriptions
        coreml_model.output_description["rep_detection"] = (
            "Rep detection probabilities over time. Shape: (1, 200). "
            "Values > 0.5 indicate active rep motion."
        )
        coreml_model.output_description["exercise_class"] = (
            "Exercise classification logits. Shape: (1, 2). "
            "Index 0: other exercises, Index 1: squat exercises."
        )
        coreml_model.output_description["form_quality"] = (
            "Form quality score. Shape: (1,). "
            "Range: 0.0 (poor form) to 1.0 (excellent form)."
        )
        coreml_model.output_description["cognitive_state"] = (
            "Cognitive state estimation. Shape: (1, 2). "
            "Index 0: fatigue level, Index 1: focus level."
        )
        
        # Save Core ML model
        output_path = self.output_dir / "QuantumLeapV3.mlmodel"
        coreml_model.save(str(output_path))
        
        logger.info(f"Core ML model saved to: {output_path}")
        
        # Get model size
        model_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model size: {model_size_mb:.1f} MB")
        
        return str(output_path)
    
    def validate_coreml_model(self, coreml_path):
        """Validate Core ML model by comparing outputs with PyTorch"""
        
        logger.info("Validating Core ML model...")
        
        # Load Core ML model
        coreml_model = ct.models.MLModel(coreml_path)
        
        # Create test input
        test_input = torch.randn(1, 13, 200)
        
        # Get PyTorch outputs
        with torch.no_grad():
            rep_detection, exercise_class, form_quality, cognitive_state = self.model(test_input)
            pytorch_outputs = {
                'rep_detection': rep_detection,
                'exercise_class': exercise_class,
                'form_quality': form_quality,
                'cognitive_state': cognitive_state
            }
        
        # Get Core ML outputs
        coreml_input = {"sensor_data": test_input.numpy()}
        coreml_outputs = coreml_model.predict(coreml_input)
        
        # Compare outputs
        logger.info("Comparing PyTorch vs Core ML outputs:")
        
        # Rep detection comparison
        pytorch_rep = pytorch_outputs['rep_detection'].numpy()
        coreml_rep = coreml_outputs['rep_detection']
        rep_diff = np.mean(np.abs(pytorch_rep - coreml_rep))
        logger.info(f"  Rep Detection MAE: {rep_diff:.6f}")
        
        # Exercise classification comparison
        pytorch_exercise = pytorch_outputs['exercise_class'].numpy()
        coreml_exercise = coreml_outputs['exercise_class']
        exercise_diff = np.mean(np.abs(pytorch_exercise - coreml_exercise))
        logger.info(f"  Exercise Classification MAE: {exercise_diff:.6f}")
        
        # Form quality comparison
        pytorch_quality = pytorch_outputs['form_quality'].numpy()
        coreml_quality = coreml_outputs['form_quality']
        quality_diff = np.mean(np.abs(pytorch_quality - coreml_quality))
        logger.info(f"  Form Quality MAE: {quality_diff:.6f}")
        
        # Cognitive state comparison
        pytorch_cognitive = pytorch_outputs['cognitive_state'].numpy()
        coreml_cognitive = coreml_outputs['cognitive_state']
        cognitive_diff = np.mean(np.abs(pytorch_cognitive - coreml_cognitive))
        logger.info(f"  Cognitive State MAE: {cognitive_diff:.6f}")
        
        # Overall validation
        max_diff = max(rep_diff, exercise_diff, quality_diff, cognitive_diff)
        if max_diff < 0.001:
            logger.info("✅ Core ML model validation PASSED (excellent accuracy)")
        elif max_diff < 0.01:
            logger.info("⚠️  Core ML model validation PASSED (good accuracy)")
        else:
            logger.warning("❌ Core ML model validation FAILED (poor accuracy)")
        
        return max_diff
    
    def benchmark_performance(self, coreml_path, num_runs=100):
        """Benchmark Core ML model inference performance"""
        
        logger.info(f"Benchmarking Core ML model performance ({num_runs} runs)...")
        
        # Load Core ML model
        coreml_model = ct.models.MLModel(coreml_path)
        
        # Create test input
        test_input = {"sensor_data": np.random.randn(1, 13, 200).astype(np.float32)}
        
        # Warmup
        for _ in range(10):
            _ = coreml_model.predict(test_input)
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(num_runs):
            _ = coreml_model.predict(test_input)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_runs) * 1000
        
        logger.info(f"Average inference time: {avg_time_ms:.2f} ms")
        logger.info(f"Throughput: {1000/avg_time_ms:.1f} inferences/second")
        
        # Check if suitable for real-time use (target: <100ms for 2-second windows)
        if avg_time_ms < 50:
            logger.info("✅ EXCELLENT performance for real-time use")
        elif avg_time_ms < 100:
            logger.info("⚠️  GOOD performance for real-time use")
        else:
            logger.warning("❌ TOO SLOW for real-time use")
        
        return avg_time_ms
    
    def convert_and_validate(self):
        """Complete conversion and validation pipeline"""
        
        logger.info("Starting Core ML Conversion Pipeline")
        logger.info("=" * 50)
        
        # Convert to Core ML
        coreml_path = self.convert_to_coreml()
        
        # Validate conversion
        max_diff = self.validate_coreml_model(coreml_path)
        
        # Benchmark performance
        avg_time = self.benchmark_performance(coreml_path)
        
        # Generate summary
        logger.info("\n" + "=" * 50)
        logger.info("CORE ML CONVERSION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"✅ Model converted successfully: {coreml_path}")
        logger.info(f"📊 Validation accuracy: {max_diff:.6f} MAE")
        logger.info(f"⚡ Inference performance: {avg_time:.2f} ms")
        logger.info(f"📱 iOS deployment ready: QuantumLeapV3.mlmodel")
        
        return {
            'model_path': coreml_path,
            'validation_error': max_diff,
            'inference_time_ms': avg_time
        }

def main():
    parser = argparse.ArgumentParser(description="Convert QuantumLeap v3 to Core ML")
    parser.add_argument("--model", required=True, help="Path to PyTorch model (.pth)")
    parser.add_argument("--output", default="./", help="Output directory")
    
    args = parser.parse_args()
    
    # Convert model
    converter = CoreMLConverter(args.model, args.output)
    results = converter.convert_and_validate()
    
    logger.info("Core ML conversion completed successfully!")
    return results

if __name__ == "__main__":
    main()
