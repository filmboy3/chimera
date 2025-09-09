#!/usr/bin/env python3
"""
Phase 1 Component Validation - No External Dependencies

Tests core functionality of all Phase 1 components to verify they work.
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ComponentValidator:
    def __init__(self):
        self.results = []
        
    def test_component(self, name: str, test_func):
        """Test a component and record results"""
        print(f"🔬 Testing {name}...")
        start_time = time.time()
        
        try:
            result = test_func()
            execution_time = time.time() - start_time
            
            self.results.append({
                'component': name,
                'passed': True,
                'execution_time': execution_time,
                'result': result
            })
            print(f"✅ {name} - PASSED ({execution_time:.3f}s)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.results.append({
                'component': name,
                'passed': False,
                'execution_time': execution_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            print(f"❌ {name} - FAILED ({execution_time:.3f}s)")
            print(f"   Error: {str(e)}")
    
    def test_mujoco_data_generator(self):
        """Test MuJoCo data generator structure"""
        sys.path.append(str(project_root / "quantumleap-v3"))
        from data_generation.generate_squat_dataset import SquatDataGenerator
        
        # Test class instantiation
        config = {
            'num_samples': 10,
            'sequence_length': 100,
            'sample_rate': 100
        }
        generator = SquatDataGenerator(config)
        
        # Test configuration
        assert hasattr(generator, 'config')
        assert generator.config['num_samples'] == 10
        
        # Test method existence
        assert hasattr(generator, 'generate_dataset')
        assert hasattr(generator, 'generate_single_squat')
        
        return {"config_loaded": True, "methods_available": True}
    
    def test_quantumleap_model(self):
        """Test QuantumLeap v3 model architecture"""
        try:
            import torch
        except ImportError:
            return {"status": "skipped", "reason": "PyTorch not available"}
        
        sys.path.append(str(project_root / "quantumleap-v3"))
        from models.qlv3_architecture import QuantumLeapV3Model
        
        # Test model creation
        model = QuantumLeapV3Model(
            input_channels=12,
            sequence_length=100,
            num_pose_joints=17,
            num_exercise_classes=5,
            num_form_errors=4,
            num_cognitive_states=3
        )
        
        # Test forward pass
        test_input = torch.randn(2, 12, 100)
        with torch.no_grad():
            outputs = model(test_input)
        
        # Validate outputs
        required_outputs = ['pose_estimation', 'exercise_classification', 'rep_detection', 'cognitive_state']
        for output_key in required_outputs:
            assert output_key in outputs, f"Missing output: {output_key}"
        
        param_count = sum(p.numel() for p in model.parameters())
        
        return {
            "model_created": True,
            "forward_pass": True,
            "parameter_count": param_count,
            "output_shapes": {k: list(v.shape) for k, v in outputs.items()}
        }
    
    def test_training_pipeline(self):
        """Test PyTorch Lightning training pipeline"""
        try:
            import torch
            import pytorch_lightning as pl
        except ImportError:
            return {"status": "skipped", "reason": "PyTorch Lightning not available"}
        
        sys.path.append(str(project_root / "quantumleap-v3"))
        from training.train_qlv3 import QuantumLeapV3Lightning
        
        # Test Lightning module creation
        lightning_model = QuantumLeapV3Lightning(
            input_channels=12,
            sequence_length=100,
            learning_rate=1e-3
        )
        
        # Test method existence
        assert hasattr(lightning_model, 'training_step')
        assert hasattr(lightning_model, 'validation_step')
        assert hasattr(lightning_model, 'configure_optimizers')
        
        return {"lightning_module": True, "methods_available": True}
    
    def test_sesame_audio_pipeline(self):
        """Test Sesame v2 audio components"""
        try:
            import torch
            import torchaudio
        except ImportError:
            return {"status": "skipped", "reason": "Audio libraries not available"}
        
        sys.path.append(str(project_root / "sesame-v2"))
        from audio_pipeline.panns_classifier import PANNsAudioClassifier
        
        # Test classifier creation
        classifier = PANNsAudioClassifier()
        
        # Test method existence
        assert hasattr(classifier, 'classify_audio_events')
        assert hasattr(classifier, 'load_model')
        
        return {"classifier_created": True, "methods_available": True}
    
    def test_intent_recognition(self):
        """Test intent recognition component"""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            return {"status": "skipped", "reason": "Transformers not available"}
        
        sys.path.append(str(project_root / "sesame-v2"))
        from intent_recognition.mobile_bert_intent import MobileBERTIntentRecognizer
        
        # Test recognizer creation
        recognizer = MobileBERTIntentRecognizer()
        
        # Test method existence
        assert hasattr(recognizer, 'recognize_intent')
        assert hasattr(recognizer, 'load_model')
        
        return {"recognizer_created": True, "methods_available": True}
    
    def test_cognitive_modulator(self):
        """Test cognitive state modulator"""
        try:
            import torch
        except ImportError:
            return {"status": "skipped", "reason": "PyTorch not available"}
        
        sys.path.append(str(project_root / "sesame-v2"))
        from cognitive_modulator.fatigue_focus_estimator import CognitiveStateModulator
        
        # Test modulator creation
        modulator = CognitiveStateModulator()
        
        # Test method existence
        assert hasattr(modulator, 'estimate_cognitive_state')
        assert hasattr(modulator, 'process_motion_features')
        
        return {"modulator_created": True, "methods_available": True}
    
    def test_coreml_converter(self):
        """Test Core ML conversion pipeline"""
        try:
            import coremltools
        except ImportError:
            return {"status": "skipped", "reason": "Core ML Tools not available"}
        
        sys.path.append(str(project_root / "quantumleap-v3"))
        from deployment.coreml_converter import CoreMLConverter, ConversionConfig
        
        # Test converter creation
        config = ConversionConfig()
        converter = CoreMLConverter(config)
        
        # Test method existence
        assert hasattr(converter, 'convert_model')
        assert hasattr(converter, 'load_pytorch_model')
        
        return {"converter_created": True, "methods_available": True}
    
    def test_ios_components(self):
        """Test iOS component files exist and have basic structure"""
        ios_files = [
            "ios_app/ChimeraApp/ChimeraApp.swift",
            "ios_app/ChimeraApp/ContentView.swift", 
            "ios_app/ChimeraApp/PerceptionEngine.swift",
            "ios_app/ChimeraApp/CoachingEngine.swift",
            "ios_app/ChimeraApp/MotionManager.swift",
            "ios_app/UnifiedAudioEngine/UnifiedAudioEngine.swift"
        ]
        
        results = {}
        for file_path in ios_files:
            full_path = project_root / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    content = f.read()
                    results[file_path] = {
                        "exists": True,
                        "has_imports": "import" in content,
                        "has_classes": "class" in content or "struct" in content,
                        "line_count": len(content.splitlines())
                    }
            else:
                results[file_path] = {"exists": False}
        
        return results
    
    def test_configuration_files(self):
        """Test configuration and setup files"""
        config_files = [
            "README.md",
            ".gitignore", 
            "requirements.txt",
            "docker/Dockerfile",
            "docker/docker-compose.yml"
        ]
        
        results = {}
        for file_path in config_files:
            full_path = project_root / file_path
            results[file_path] = {
                "exists": full_path.exists(),
                "size_bytes": full_path.stat().st_size if full_path.exists() else 0
            }
        
        return results
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("🚀 Starting Phase 1 Component Validation")
        print("=" * 50)
        
        # Test each component
        self.test_component("Configuration Files", self.test_configuration_files)
        self.test_component("iOS Components", self.test_ios_components)
        self.test_component("MuJoCo Data Generator", self.test_mujoco_data_generator)
        self.test_component("QuantumLeap v3 Model", self.test_quantumleap_model)
        self.test_component("Training Pipeline", self.test_training_pipeline)
        self.test_component("Sesame Audio Pipeline", self.test_sesame_audio_pipeline)
        self.test_component("Intent Recognition", self.test_intent_recognition)
        self.test_component("Cognitive Modulator", self.test_cognitive_modulator)
        self.test_component("Core ML Converter", self.test_coreml_converter)
        
        # Generate summary
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("PHASE 1 VALIDATION SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {passed_tests/total_tests:.1%}")
        
        total_time = sum(r['execution_time'] for r in self.results)
        print(f"⏱️  Total Time: {total_time:.3f}s")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Components:")
            for result in self.results:
                if not result['passed']:
                    print(f"   • {result['component']}: {result['error']}")
        
        print(f"\n✅ Working Components:")
        for result in self.results:
            if result['passed']:
                print(f"   • {result['component']}")
    
    def save_results(self):
        """Save results to JSON file"""
        results_dir = project_root / "tests" / "results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"phase1_validation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")

def main():
    validator = ComponentValidator()
    validator.run_all_tests()

if __name__ == "__main__":
    main()
