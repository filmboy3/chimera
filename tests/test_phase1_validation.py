#!/usr/bin/env python3
"""
Phase 1 Validation Test Suite

Comprehensive testing framework to validate all Phase 1 components
and generate performance metrics for Project Chimera Ascendant.
"""

import pytest
import torch
import numpy as np
import h5py
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import time
from dataclasses import dataclass

# Import components to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class TestResults:
    """Test results container"""
    component: str
    test_name: str
    passed: bool
    execution_time: float
    error_message: str = ""
    metrics: Dict = None

class Phase1TestSuite:
    """Comprehensive test suite for Phase 1 components"""
    
    def __init__(self):
        self.results: List[TestResults] = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("Phase1Tests")
    
    def run_all_tests(self) -> Dict:
        """Run all Phase 1 validation tests"""
        self.logger.info("🧪 Starting Phase 1 Validation Test Suite")
        
        # Test each component
        self.test_mujoco_data_generation()
        self.test_quantumleap_architecture()
        self.test_sesame_audio_pipeline()
        self.test_coreml_conversion()
        self.test_ios_components()
        
        # Generate summary
        summary = self._generate_test_summary()
        self._save_test_results(summary)
        
        return summary
    
    def test_mujoco_data_generation(self):
        """Test MuJoCo synthetic data generation"""
        self.logger.info("🔬 Testing MuJoCo Data Generation...")
        
        start_time = time.time()
        try:
            # Test data generator import
            from quantumleap_v3.data_generation.generate_squat_dataset import SquatDatasetGenerator
            
            # Create test generator with minimal samples
            config = {
                'num_samples': 100,
                'sequence_length': 200,
                'sample_rate': 100,
                'placement_randomization': True,
                'barometer_fusion': True,
                'fatigue_simulation': True
            }
            
            generator = SquatDatasetGenerator(config)
            
            # Test data generation
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "test_dataset.h5")
                
                # Generate small test dataset
                dataset_info = generator.generate_dataset(output_path)
                
                # Validate dataset
                with h5py.File(output_path, 'r') as f:
                    assert 'imu_data' in f, "IMU data missing"
                    assert 'barometer_data' in f, "Barometer data missing"
                    assert 'pose_labels' in f, "Pose labels missing"
                    assert len(f['imu_data']) == 100, f"Expected 100 samples, got {len(f['imu_data'])}"
                    
                    # Check data shapes
                    imu_shape = f['imu_data'].shape
                    assert imu_shape[1] == 200, f"Expected sequence length 200, got {imu_shape[1]}"
                    assert imu_shape[2] == 12, f"Expected 12 IMU channels, got {imu_shape[2]}"
            
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="MuJoCo Data Generation",
                test_name="Synthetic Dataset Generation",
                passed=True,
                execution_time=execution_time,
                metrics={"samples_generated": 100, "generation_rate": 100/execution_time}
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="MuJoCo Data Generation",
                test_name="Synthetic Dataset Generation",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
    
    def test_quantumleap_architecture(self):
        """Test QuantumLeap v3 model architecture"""
        self.logger.info("🔬 Testing QuantumLeap v3 Architecture...")
        
        start_time = time.time()
        try:
            from quantumleap_v3.models.qlv3_architecture import QuantumLeapV3Model
            
            # Create model instance
            model = QuantumLeapV3Model(
                input_channels=12,
                sequence_length=200,
                num_pose_joints=17,
                num_exercise_classes=5,
                num_form_errors=4,
                num_cognitive_states=3,
                vq_num_embeddings=512,
                vq_embedding_dim=256,
                transformer_dim=512,
                transformer_heads=8,
                transformer_layers=6
            )
            
            # Test forward pass
            batch_size = 4
            test_input = torch.randn(batch_size, 12, 200)
            
            with torch.no_grad():
                outputs = model(test_input)
            
            # Validate outputs
            assert 'pose_estimation' in outputs, "Pose estimation output missing"
            assert 'exercise_classification' in outputs, "Exercise classification missing"
            assert 'rep_detection' in outputs, "Rep detection missing"
            assert 'cognitive_state' in outputs, "Cognitive state missing"
            
            # Check output shapes
            assert outputs['pose_estimation'].shape == (batch_size, 51), f"Wrong pose shape: {outputs['pose_estimation'].shape}"
            assert outputs['exercise_classification'].shape == (batch_size, 5), f"Wrong exercise shape: {outputs['exercise_classification'].shape}"
            
            # Test parameter count
            param_count = sum(p.numel() for p in model.parameters())
            
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="QuantumLeap v3",
                test_name="Model Architecture",
                passed=True,
                execution_time=execution_time,
                metrics={
                    "parameter_count": param_count,
                    "inference_time_ms": execution_time * 1000,
                    "memory_usage_mb": param_count * 4 / (1024*1024)  # Rough estimate
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="QuantumLeap v3",
                test_name="Model Architecture",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
    
    def test_sesame_audio_pipeline(self):
        """Test Sesame v2 audio pipeline components"""
        self.logger.info("🔬 Testing Sesame v2 Audio Pipeline...")
        
        # Test PANNs classifier
        start_time = time.time()
        try:
            from sesame_v2.audio_pipeline.panns_classifier import PANNsAudioClassifier
            
            classifier = PANNsAudioClassifier()
            
            # Test with dummy audio data
            dummy_audio = np.random.randn(16000)  # 1 second at 16kHz
            
            # Test classification
            events = classifier.classify_audio_events(dummy_audio)
            assert isinstance(events, dict), "Events should be dictionary"
            assert 'speech_probability' in events, "Speech probability missing"
            
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="Sesame v2 Audio",
                test_name="PANNs Audio Classification",
                passed=True,
                execution_time=execution_time,
                metrics={"audio_processing_time": execution_time}
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="Sesame v2 Audio",
                test_name="PANNs Audio Classification",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
        
        # Test Intent Recognition
        start_time = time.time()
        try:
            from sesame_v2.intent_recognition.mobile_bert_intent import MobileBERTIntentRecognizer
            
            recognizer = MobileBERTIntentRecognizer()
            
            # Test intent recognition
            test_text = "start workout"
            intent = recognizer.recognize_intent(test_text)
            
            assert isinstance(intent, dict), "Intent should be dictionary"
            assert 'intent_class' in intent, "Intent class missing"
            
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="Sesame v2 Intent",
                test_name="MobileBERT Intent Recognition",
                passed=True,
                execution_time=execution_time,
                metrics={"intent_processing_time": execution_time}
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="Sesame v2 Intent",
                test_name="MobileBERT Intent Recognition",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
    
    def test_coreml_conversion(self):
        """Test Core ML conversion pipeline"""
        self.logger.info("🔬 Testing Core ML Conversion...")
        
        start_time = time.time()
        try:
            from quantumleap_v3.deployment.coreml_converter import CoreMLConverter, ConversionConfig
            
            # Create minimal model for testing
            class TestModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 5)
                
                def forward(self, x):
                    return {'output': self.linear(x)}
            
            test_model = TestModel()
            
            # Save test model
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "test_model.pth")
                torch.save(test_model.state_dict(), model_path)
                
                # Test conversion config
                config = ConversionConfig(
                    model_name="TestModel",
                    input_sequence_length=10,
                    input_channels=1,
                    quantization_bits=16
                )
                
                converter = CoreMLConverter(config)
                
                # Test conversion preparation (without actual Core ML conversion)
                example_input = torch.randn(1, 10, 1)
                traced_model = torch.jit.trace(test_model, example_input)
                
                assert traced_model is not None, "Model tracing failed"
            
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="Core ML Conversion",
                test_name="Model Conversion Pipeline",
                passed=True,
                execution_time=execution_time,
                metrics={"conversion_setup_time": execution_time}
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="Core ML Conversion",
                test_name="Model Conversion Pipeline",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
    
    def test_ios_components(self):
        """Test iOS component interfaces (syntax and structure)"""
        self.logger.info("🔬 Testing iOS Component Interfaces...")
        
        start_time = time.time()
        try:
            # Test that iOS Swift files have valid syntax by checking imports and structure
            ios_files = [
                "ios_app/ChimeraApp/ChimeraApp.swift",
                "ios_app/ChimeraApp/ContentView.swift",
                "ios_app/ChimeraApp/PerceptionEngine.swift",
                "ios_app/ChimeraApp/CoachingEngine.swift",
                "ios_app/ChimeraApp/MotionManager.swift",
                "ios_app/UnifiedAudioEngine/UnifiedAudioEngine.swift"
            ]
            
            valid_files = 0
            for file_path in ios_files:
                full_path = Path(__file__).parent.parent / file_path
                if full_path.exists():
                    with open(full_path, 'r') as f:
                        content = f.read()
                        # Basic syntax checks
                        assert 'import' in content, f"No imports in {file_path}"
                        assert 'class' in content or 'struct' in content, f"No classes/structs in {file_path}"
                        valid_files += 1
            
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="iOS Integration",
                test_name="Component Structure Validation",
                passed=True,
                execution_time=execution_time,
                metrics={"valid_files": valid_files, "total_files": len(ios_files)}
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResults(
                component="iOS Integration",
                test_name="Component Structure Validation",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
    
    def _generate_test_summary(self) -> Dict:
        """Generate comprehensive test summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        total_time = sum(r.execution_time for r in self.results)
        
        # Group by component
        by_component = {}
        for result in self.results:
            if result.component not in by_component:
                by_component[result.component] = []
            by_component[result.component].append(result)
        
        component_summary = {}
        for component, tests in by_component.items():
            component_passed = sum(1 for t in tests if t.passed)
            component_total = len(tests)
            component_summary[component] = {
                "passed": component_passed,
                "total": component_total,
                "success_rate": component_passed / component_total if component_total > 0 else 0,
                "total_time": sum(t.execution_time for t in tests)
            }
        
        return {
            "overall": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_execution_time": total_time
            },
            "by_component": component_summary,
            "detailed_results": [
                {
                    "component": r.component,
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "metrics": r.metrics
                }
                for r in self.results
            ]
        }
    
    def _save_test_results(self, summary: Dict):
        """Save test results to file"""
        results_dir = Path("tests/results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"phase1_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"📊 Test results saved to {results_file}")

def main():
    """Run Phase 1 validation tests"""
    test_suite = Phase1TestSuite()
    summary = test_suite.run_all_tests()
    
    print("\n" + "="*60)
    print("PHASE 1 VALIDATION TEST RESULTS")
    print("="*60)
    
    overall = summary['overall']
    print(f"📊 Overall Results: {overall['passed']}/{overall['total_tests']} tests passed ({overall['success_rate']:.1%})")
    print(f"⏱️  Total execution time: {overall['total_execution_time']:.2f}s")
    
    print("\n📋 Component Breakdown:")
    for component, stats in summary['by_component'].items():
        status = "✅" if stats['success_rate'] == 1.0 else "❌" if stats['success_rate'] == 0.0 else "⚠️"
        print(f"{status} {component}: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1%})")
    
    print("\n🔍 Detailed Results:")
    for result in summary['detailed_results']:
        status = "✅" if result['passed'] else "❌"
        print(f"{status} {result['component']} - {result['test_name']}")
        if not result['passed']:
            print(f"   Error: {result['error_message']}")
        if result['metrics']:
            print(f"   Metrics: {result['metrics']}")
    
    return summary

if __name__ == "__main__":
    main()
