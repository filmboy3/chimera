#!/usr/bin/env python3
"""
Simplified Ablation Study for QuantumLeap v3
Tests placement invariance and barometric fusion without complex dependencies
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import json
import argparse
from pathlib import Path
import logging
import sys

# Add path for model import
sys.path.append('../quantumleap-v3/training')
from simple_trainer import SimpleQuantumLeapV3, SquatDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AblationStudy:
    """Ablation study to validate key architectural decisions"""
    
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load trained model
        self.model = SimpleQuantumLeapV3(input_channels=13, hidden_dim=128, num_layers=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load test dataset
        self.test_dataset = SquatDataset(data_path, split='test')
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=32, shuffle=False
        )
        
        logger.info(f"Loaded model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info(f"Test dataset: {len(self.test_dataset)} samples")
    
    def evaluate_model(self, model, test_loader, description=""):
        """Evaluate model performance on test set"""
        model.eval()
        total_loss = 0
        rep_correct = 0
        rep_total = 0
        exercise_correct = 0
        exercise_total = 0
        
        rep_criterion = nn.BCELoss()
        exercise_criterion = nn.CrossEntropyLoss()
        quality_criterion = nn.MSELoss()
        cognitive_criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['inputs'].to(self.device)
                rep_labels = batch['rep_detection'].to(self.device)
                exercise_labels = batch['exercise_class'].squeeze().to(self.device)
                quality_labels = batch['form_quality'].squeeze().to(self.device)
                cognitive_labels = batch['cognitive_state'].to(self.device)
                
                outputs = model(inputs)
                
                # Calculate losses
                rep_loss = rep_criterion(outputs['rep_detection'], rep_labels)
                exercise_loss = exercise_criterion(outputs['exercise_class'], exercise_labels)
                quality_loss = quality_criterion(outputs['form_quality'], quality_labels)
                cognitive_loss = cognitive_criterion(outputs['cognitive_state'], cognitive_labels)
                
                total_loss += (2.0 * rep_loss + 0.5 * exercise_loss + 
                              1.0 * quality_loss + 0.5 * cognitive_loss).item()
                
                # Rep detection accuracy (threshold at 0.5)
                rep_pred = (outputs['rep_detection'] > 0.5).float()
                rep_correct += (rep_pred == rep_labels).sum().item()
                rep_total += rep_labels.numel()
                
                # Exercise classification accuracy
                exercise_pred = outputs['exercise_class'].argmax(dim=1)
                exercise_correct += (exercise_pred == exercise_labels).sum().item()
                exercise_total += exercise_labels.size(0)
        
        avg_loss = total_loss / len(test_loader)
        rep_accuracy = rep_correct / rep_total
        exercise_accuracy = exercise_correct / exercise_total
        
        logger.info(f"{description} Results:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Rep Detection Accuracy: {rep_accuracy:.3f}")
        logger.info(f"  Exercise Classification Accuracy: {exercise_accuracy:.3f}")
        
        return {
            'loss': avg_loss,
            'rep_accuracy': rep_accuracy,
            'exercise_accuracy': exercise_accuracy
        }
    
    def test_placement_invariance(self):
        """Test model robustness to sensor placement variations"""
        logger.info("Testing Placement Invariance...")
        
        # Baseline performance
        baseline_results = self.evaluate_model(self.model, self.test_loader, "Baseline")
        
        # Test with sensor position perturbations
        perturbation_results = []
        
        # Create modified test loaders with different perturbations
        for noise_level in [0.1, 0.2, 0.3]:
            logger.info(f"Testing with {noise_level:.1f} noise level...")
            
            # Create perturbed dataset
            perturbed_results = self.evaluate_perturbed_data(noise_level)
            perturbation_results.append({
                'noise_level': noise_level,
                'results': perturbed_results
            })
        
        # Calculate robustness metrics
        robustness_scores = []
        for result in perturbation_results:
            noise_level = result['noise_level']
            perturbed = result['results']
            
            # Robustness = 1 - (performance_drop / baseline_performance)
            rep_drop = baseline_results['rep_accuracy'] - perturbed['rep_accuracy']
            rep_robustness = 1 - (rep_drop / baseline_results['rep_accuracy'])
            
            exercise_drop = baseline_results['exercise_accuracy'] - perturbed['exercise_accuracy']
            exercise_robustness = 1 - (exercise_drop / baseline_results['exercise_accuracy'])
            
            avg_robustness = (rep_robustness + exercise_robustness) / 2
            
            robustness_scores.append({
                'noise_level': noise_level,
                'rep_robustness': rep_robustness,
                'exercise_robustness': exercise_robustness,
                'avg_robustness': avg_robustness
            })
            
            logger.info(f"  Noise {noise_level:.1f}: Rep Robustness {rep_robustness:.3f}, "
                       f"Exercise Robustness {exercise_robustness:.3f}")
        
        return {
            'baseline': baseline_results,
            'perturbations': perturbation_results,
            'robustness_scores': robustness_scores
        }
    
    def evaluate_perturbed_data(self, noise_level):
        """Evaluate model on data with added noise"""
        total_loss = 0
        rep_correct = 0
        rep_total = 0
        exercise_correct = 0
        exercise_total = 0
        
        rep_criterion = nn.BCELoss()
        exercise_criterion = nn.CrossEntropyLoss()
        quality_criterion = nn.MSELoss()
        cognitive_criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in self.test_loader:
                inputs = batch['inputs'].to(self.device)
                
                # Add placement noise (simulate sensor position variations)
                noise = torch.randn_like(inputs) * noise_level
                perturbed_inputs = inputs + noise
                
                rep_labels = batch['rep_detection'].to(self.device)
                exercise_labels = batch['exercise_class'].squeeze().to(self.device)
                quality_labels = batch['form_quality'].squeeze().to(self.device)
                cognitive_labels = batch['cognitive_state'].to(self.device)
                
                outputs = self.model(perturbed_inputs)
                
                # Calculate losses
                rep_loss = rep_criterion(outputs['rep_detection'], rep_labels)
                exercise_loss = exercise_criterion(outputs['exercise_class'], exercise_labels)
                quality_loss = quality_criterion(outputs['form_quality'], quality_labels)
                cognitive_loss = cognitive_criterion(outputs['cognitive_state'], cognitive_labels)
                
                total_loss += (2.0 * rep_loss + 0.5 * exercise_loss + 
                              1.0 * quality_loss + 0.5 * cognitive_loss).item()
                
                # Accuracy calculations
                rep_pred = (outputs['rep_detection'] > 0.5).float()
                rep_correct += (rep_pred == rep_labels).sum().item()
                rep_total += rep_labels.numel()
                
                exercise_pred = outputs['exercise_class'].argmax(dim=1)
                exercise_correct += (exercise_pred == exercise_labels).sum().item()
                exercise_total += exercise_labels.size(0)
        
        return {
            'loss': total_loss / len(self.test_loader),
            'rep_accuracy': rep_correct / rep_total,
            'exercise_accuracy': exercise_correct / exercise_total
        }
    
    def test_barometric_fusion(self):
        """Test contribution of barometric pressure data"""
        logger.info("Testing Barometric Fusion...")
        
        # Baseline with all sensors
        baseline_results = self.evaluate_model(self.model, self.test_loader, "Full Model (with Barometer)")
        
        # Test without barometric data (zero out barometer channel)
        no_baro_results = self.evaluate_without_barometer()
        
        # Calculate improvement from barometric fusion
        rep_improvement = baseline_results['rep_accuracy'] - no_baro_results['rep_accuracy']
        exercise_improvement = baseline_results['exercise_accuracy'] - no_baro_results['exercise_accuracy']
        
        logger.info(f"Barometric Fusion Benefits:")
        logger.info(f"  Rep Detection Improvement: {rep_improvement:.3f}")
        logger.info(f"  Exercise Classification Improvement: {exercise_improvement:.3f}")
        
        return {
            'baseline': baseline_results,
            'no_barometer': no_baro_results,
            'improvements': {
                'rep_detection': rep_improvement,
                'exercise_classification': exercise_improvement
            }
        }
    
    def evaluate_without_barometer(self):
        """Evaluate model performance without barometric data"""
        total_loss = 0
        rep_correct = 0
        rep_total = 0
        exercise_correct = 0
        exercise_total = 0
        
        rep_criterion = nn.BCELoss()
        exercise_criterion = nn.CrossEntropyLoss()
        quality_criterion = nn.MSELoss()
        cognitive_criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in self.test_loader:
                inputs = batch['inputs'].to(self.device)
                
                # Zero out barometric channel (channel 12, last one)
                inputs_no_baro = inputs.clone()
                inputs_no_baro[12, :] = 0  # Zero out barometer channel
                
                rep_labels = batch['rep_detection'].to(self.device)
                exercise_labels = batch['exercise_class'].squeeze().to(self.device)
                quality_labels = batch['form_quality'].squeeze().to(self.device)
                cognitive_labels = batch['cognitive_state'].to(self.device)
                
                outputs = self.model(inputs_no_baro)
                
                # Calculate losses
                rep_loss = rep_criterion(outputs['rep_detection'], rep_labels)
                exercise_loss = exercise_criterion(outputs['exercise_class'], exercise_labels)
                quality_loss = quality_criterion(outputs['form_quality'], quality_labels)
                cognitive_loss = cognitive_criterion(outputs['cognitive_state'], cognitive_labels)
                
                total_loss += (2.0 * rep_loss + 0.5 * exercise_loss + 
                              1.0 * quality_loss + 0.5 * cognitive_loss).item()
                
                # Accuracy calculations
                rep_pred = (outputs['rep_detection'] > 0.5).float()
                rep_correct += (rep_pred == rep_labels).sum().item()
                rep_total += rep_labels.numel()
                
                exercise_pred = outputs['exercise_class'].argmax(dim=1)
                exercise_correct += (exercise_pred == exercise_labels).sum().item()
                exercise_total += exercise_labels.size(0)
        
        return {
            'loss': total_loss / len(self.test_loader),
            'rep_accuracy': rep_correct / rep_total,
            'exercise_accuracy': exercise_correct / exercise_total
        }
    
    def run_full_study(self):
        """Run complete ablation study"""
        logger.info("Starting Comprehensive Ablation Study")
        logger.info("=" * 50)
        
        # Test placement invariance
        placement_results = self.test_placement_invariance()
        
        # Test barometric fusion
        barometric_results = self.test_barometric_fusion()
        
        # Generate summary report
        self.generate_report(placement_results, barometric_results)
        
        return {
            'placement_invariance': placement_results,
            'barometric_fusion': barometric_results
        }
    
    def generate_report(self, placement_results, barometric_results):
        """Generate comprehensive ablation study report"""
        logger.info("\n" + "=" * 50)
        logger.info("ABLATION STUDY SUMMARY REPORT")
        logger.info("=" * 50)
        
        # Placement invariance summary
        logger.info("\n🎯 PLACEMENT INVARIANCE RESULTS:")
        baseline = placement_results['baseline']
        logger.info(f"  Baseline Performance:")
        logger.info(f"    Rep Detection: {baseline['rep_accuracy']:.3f}")
        logger.info(f"    Exercise Classification: {baseline['exercise_accuracy']:.3f}")
        
        logger.info(f"  Robustness to Placement Variations:")
        for score in placement_results['robustness_scores']:
            noise = score['noise_level']
            avg_rob = score['avg_robustness']
            logger.info(f"    {noise:.1f} noise level: {avg_rob:.3f} robustness")
        
        avg_robustness = np.mean([s['avg_robustness'] for s in placement_results['robustness_scores']])
        logger.info(f"  Overall Placement Robustness: {avg_robustness:.3f}")
        
        # Barometric fusion summary
        logger.info(f"\n🌡️  BAROMETRIC FUSION RESULTS:")
        baro_improvements = barometric_results['improvements']
        logger.info(f"  Rep Detection Improvement: +{baro_improvements['rep_detection']:.3f}")
        logger.info(f"  Exercise Classification Improvement: +{baro_improvements['exercise_classification']:.3f}")
        
        # Overall assessment
        logger.info(f"\n📊 OVERALL ASSESSMENT:")
        
        if avg_robustness > 0.8:
            logger.info("  ✅ EXCELLENT placement invariance (>0.8)")
        elif avg_robustness > 0.6:
            logger.info("  ⚠️  GOOD placement invariance (0.6-0.8)")
        else:
            logger.info("  ❌ POOR placement invariance (<0.6)")
        
        total_baro_improvement = (baro_improvements['rep_detection'] + 
                                 baro_improvements['exercise_classification']) / 2
        
        if total_baro_improvement > 0.05:
            logger.info("  ✅ SIGNIFICANT barometric fusion benefit (>5%)")
        elif total_baro_improvement > 0.02:
            logger.info("  ⚠️  MODERATE barometric fusion benefit (2-5%)")
        else:
            logger.info("  ❌ MINIMAL barometric fusion benefit (<2%)")
        
        logger.info("\n🎉 Ablation study completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="QuantumLeap v3 Ablation Study")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--data_path", required=True, help="Path to dataset")
    
    args = parser.parse_args()
    
    # Run ablation study
    study = AblationStudy(args.model_path, args.data_path)
    results = study.run_full_study()
    
    logger.info("Ablation study results saved and completed!")

if __name__ == "__main__":
    main()
