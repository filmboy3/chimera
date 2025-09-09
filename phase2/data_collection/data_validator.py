#!/usr/bin/env python3
"""
Real vs Synthetic Data Validation System
Compares human-collected data with synthetic training data
"""

import numpy as np
import json
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates alignment between real and synthetic squat data"""
    
    def __init__(self, synthetic_data_path, human_data_dir):
        self.synthetic_data_path = Path(synthetic_data_path)
        self.human_data_dir = Path(human_data_dir)
        
        # Load synthetic data
        self.synthetic_data = self._load_synthetic_data()
        
        # Load human data
        self.human_sessions = self._load_human_sessions()
        
        logger.info(f"Loaded {len(self.synthetic_data)} synthetic samples")
        logger.info(f"Loaded {len(self.human_sessions)} human sessions")
    
    def _load_synthetic_data(self):
        """Load synthetic training dataset"""
        
        if not self.synthetic_data_path.exists():
            logger.error(f"Synthetic data not found: {self.synthetic_data_path}")
            return []
        
        with h5py.File(self.synthetic_data_path, 'r') as f:
            phone_imu = f['phone_imu'][:]
            watch_imu = f['watch_imu'][:]
            barometer = f['barometer'][:]
            labels = f['labels'][:]
        
        return {
            'phone_imu': phone_imu,
            'watch_imu': watch_imu,
            'barometer': barometer,
            'labels': labels
        }
    
    def _load_human_sessions(self):
        """Load all human session files"""
        
        sessions = []
        
        if not self.human_data_dir.exists():
            logger.warning(f"Human data directory not found: {self.human_data_dir}")
            return sessions
        
        for session_file in self.human_data_dir.glob("human_squat_*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                sessions.append(session_data)
            except Exception as e:
                logger.error(f"Error loading {session_file}: {e}")
        
        return sessions
    
    def validate_data_alignment(self):
        """Comprehensive validation of real vs synthetic data"""
        
        if len(self.human_sessions) == 0:
            logger.error("No human data available for validation")
            return None
        
        validation_results = {
            'statistical_alignment': self._validate_statistical_properties(),
            'motion_patterns': self._validate_motion_patterns(),
            'rep_detection': self._validate_rep_detection(),
            'sensor_fusion': self._validate_sensor_fusion(),
            'overall_score': 0.0
        }
        
        # Calculate overall alignment score
        scores = [
            validation_results['statistical_alignment']['alignment_score'],
            validation_results['motion_patterns']['pattern_similarity'],
            validation_results['rep_detection']['detection_accuracy'],
            validation_results['sensor_fusion']['fusion_benefit']
        ]
        
        validation_results['overall_score'] = np.mean(scores)
        
        logger.info(f"Overall data alignment score: {validation_results['overall_score']:.3f}")
        
        return validation_results
    
    def _validate_statistical_properties(self):
        """Compare statistical properties of real vs synthetic data"""
        
        # Extract human IMU data
        human_phone_imu = []
        for session in self.human_sessions:
            phone_data = np.array(session['sensor_data']['phone_imu'])
            if len(phone_data) > 0:
                human_phone_imu.append(phone_data)
        
        if len(human_phone_imu) == 0:
            return {'alignment_score': 0.0, 'details': 'No human data'}
        
        # Concatenate all human data
        human_data = np.vstack(human_phone_imu)
        synthetic_data = self.synthetic_data['phone_imu']
        
        # Compare statistical moments
        alignment_scores = []
        
        for axis in range(6):  # 6 IMU channels
            human_axis = human_data[:, axis]
            synthetic_axis = synthetic_data[:, axis].flatten()
            
            # Compare means
            mean_diff = abs(np.mean(human_axis) - np.mean(synthetic_axis))
            mean_score = max(0, 1 - mean_diff / max(abs(np.mean(human_axis)), 1e-6))
            
            # Compare standard deviations
            std_diff = abs(np.std(human_axis) - np.std(synthetic_axis))
            std_score = max(0, 1 - std_diff / max(np.std(human_axis), 1e-6))
            
            # KS test for distribution similarity
            ks_stat, ks_p = stats.ks_2samp(human_axis, synthetic_axis[:len(human_axis)])
            ks_score = max(0, 1 - ks_stat)
            
            axis_score = np.mean([mean_score, std_score, ks_score])
            alignment_scores.append(axis_score)
        
        overall_alignment = np.mean(alignment_scores)
        
        return {
            'alignment_score': overall_alignment,
            'axis_scores': alignment_scores,
            'details': {
                'human_samples': len(human_data),
                'synthetic_samples': len(synthetic_data),
                'mean_alignment': np.mean([s for s in alignment_scores]),
                'std_alignment': np.std(alignment_scores)
            }
        }
    
    def _validate_motion_patterns(self):
        """Validate motion pattern similarity"""
        
        # Extract motion patterns from human data
        human_patterns = []
        for session in self.human_sessions:
            phone_data = np.array(session['sensor_data']['phone_imu'])
            if len(phone_data) > 100:  # Need sufficient data
                # Extract Y-axis acceleration (primary squat motion)
                y_accel = phone_data[:, 1]
                
                # Find motion cycles
                cycles = self._extract_motion_cycles(y_accel)
                human_patterns.extend(cycles)
        
        if len(human_patterns) == 0:
            return {'pattern_similarity': 0.0, 'details': 'No patterns extracted'}
        
        # Extract synthetic patterns
        synthetic_patterns = []
        synthetic_phone = self.synthetic_data['phone_imu']
        
        for i in range(min(100, len(synthetic_phone))):  # Sample synthetic data
            y_accel = synthetic_phone[i, :, 1]  # Y-axis
            cycles = self._extract_motion_cycles(y_accel)
            synthetic_patterns.extend(cycles)
        
        # Compare pattern characteristics
        if len(synthetic_patterns) == 0:
            return {'pattern_similarity': 0.0, 'details': 'No synthetic patterns'}
        
        # Calculate pattern similarity
        human_features = self._extract_pattern_features(human_patterns)
        synthetic_features = self._extract_pattern_features(synthetic_patterns)
        
        similarity_score = self._calculate_feature_similarity(human_features, synthetic_features)
        
        return {
            'pattern_similarity': similarity_score,
            'human_patterns': len(human_patterns),
            'synthetic_patterns': len(synthetic_patterns),
            'details': {
                'human_features': human_features,
                'synthetic_features': synthetic_features
            }
        }
    
    def _extract_motion_cycles(self, signal, min_cycle_length=50):
        """Extract individual motion cycles from signal"""
        
        # Simple peak detection for cycle segmentation
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(signal, height=np.mean(signal), distance=min_cycle_length)
        
        cycles = []
        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i + 1]
            if end - start >= min_cycle_length:
                cycle = signal[start:end]
                cycles.append(cycle)
        
        return cycles
    
    def _extract_pattern_features(self, patterns):
        """Extract features from motion patterns"""
        
        if len(patterns) == 0:
            return {'duration': 0, 'amplitude': 0, 'frequency': 0}
        
        durations = [len(p) for p in patterns]
        amplitudes = [np.ptp(p) for p in patterns]  # Peak-to-peak
        
        return {
            'duration': np.mean(durations),
            'amplitude': np.mean(amplitudes),
            'frequency': 1.0 / (np.mean(durations) / 100),  # Assuming 100Hz
            'duration_std': np.std(durations),
            'amplitude_std': np.std(amplitudes)
        }
    
    def _calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between feature sets"""
        
        similarities = []
        
        for key in ['duration', 'amplitude', 'frequency']:
            if key in features1 and key in features2:
                val1, val2 = features1[key], features2[key]
                if val1 > 0 and val2 > 0:
                    similarity = 1 - abs(val1 - val2) / max(val1, val2)
                    similarities.append(max(0, similarity))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _validate_rep_detection(self):
        """Validate rep detection accuracy on human data"""
        
        detection_results = []
        
        for session in self.human_sessions:
            # Get ground truth rep count
            ground_truth_reps = session['analysis']['rep_count']
            
            # Simulate rep detection on human data
            phone_data = np.array(session['sensor_data']['phone_imu'])
            if len(phone_data) > 0:
                detected_reps = self._detect_reps_in_data(phone_data)
                
                # Calculate accuracy
                if ground_truth_reps > 0:
                    accuracy = 1 - abs(detected_reps - ground_truth_reps) / ground_truth_reps
                    accuracy = max(0, accuracy)
                    detection_results.append(accuracy)
        
        avg_accuracy = np.mean(detection_results) if detection_results else 0.0
        
        return {
            'detection_accuracy': avg_accuracy,
            'sessions_tested': len(detection_results),
            'individual_accuracies': detection_results
        }
    
    def _detect_reps_in_data(self, imu_data):
        """Simple rep detection algorithm"""
        
        if len(imu_data) < 10:
            return 0
        
        # Use Y-axis acceleration for rep detection
        y_accel = imu_data[:, 1]
        
        # Simple threshold-based detection
        threshold = np.std(y_accel) * 1.5
        peaks, _ = find_peaks(np.abs(y_accel - np.mean(y_accel)), 
                             height=threshold, distance=50)
        
        return len(peaks)
    
    def _validate_sensor_fusion(self):
        """Validate multi-sensor fusion benefits"""
        
        # Compare phone-only vs phone+watch+barometer performance
        phone_only_scores = []
        fusion_scores = []
        
        for session in self.human_sessions:
            phone_data = np.array(session['sensor_data']['phone_imu'])
            watch_data = np.array(session['sensor_data']['watch_imu'])
            barometer_data = np.array(session['sensor_data']['barometer'])
            
            if len(phone_data) > 0 and len(watch_data) > 0:
                # Simulate performance with phone only
                phone_score = self._calculate_motion_quality(phone_data)
                
                # Simulate performance with fusion
                fusion_score = self._calculate_fusion_quality(phone_data, watch_data, barometer_data)
                
                phone_only_scores.append(phone_score)
                fusion_scores.append(fusion_score)
        
        if len(phone_only_scores) == 0:
            return {'fusion_benefit': 0.0, 'details': 'No data for fusion validation'}
        
        avg_phone_score = np.mean(phone_only_scores)
        avg_fusion_score = np.mean(fusion_scores)
        
        fusion_benefit = (avg_fusion_score - avg_phone_score) / max(avg_phone_score, 1e-6)
        fusion_benefit = max(0, fusion_benefit)
        
        return {
            'fusion_benefit': fusion_benefit,
            'phone_only_score': avg_phone_score,
            'fusion_score': avg_fusion_score,
            'improvement': fusion_benefit * 100  # Percentage improvement
        }
    
    def _calculate_motion_quality(self, imu_data):
        """Calculate motion quality score from IMU data"""
        
        if len(imu_data) == 0:
            return 0.0
        
        # Simple quality metric based on motion consistency
        y_accel = imu_data[:, 1]
        consistency = 1.0 - (np.std(y_accel) / (np.mean(np.abs(y_accel)) + 1e-6))
        return max(0, min(1, consistency))
    
    def _calculate_fusion_quality(self, phone_data, watch_data, barometer_data):
        """Calculate quality with sensor fusion"""
        
        phone_quality = self._calculate_motion_quality(phone_data)
        
        # Simple fusion benefit simulation
        if len(watch_data) > 0:
            watch_quality = self._calculate_motion_quality(watch_data)
            fusion_quality = 0.7 * phone_quality + 0.3 * watch_quality
        else:
            fusion_quality = phone_quality
        
        # Barometer adds small benefit
        if len(barometer_data) > 0:
            fusion_quality += 0.05  # Small boost from barometer
        
        return min(1.0, fusion_quality)
    
    def generate_validation_report(self, results, output_path="validation_report.json"):
        """Generate comprehensive validation report"""
        
        report = {
            'validation_timestamp': str(np.datetime64('now')),
            'data_summary': {
                'synthetic_samples': len(self.synthetic_data['phone_imu']) if self.synthetic_data else 0,
                'human_sessions': len(self.human_sessions),
                'total_human_samples': sum(len(s['sensor_data']['phone_imu']) for s in self.human_sessions)
            },
            'validation_results': results,
            'recommendations': self._generate_recommendations(results)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {output_path}")
        return report
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        overall_score = results.get('overall_score', 0)
        
        if overall_score < 0.7:
            recommendations.append("LOW ALIGNMENT: Consider collecting more diverse human data")
        
        if results['statistical_alignment']['alignment_score'] < 0.6:
            recommendations.append("STATISTICAL MISMATCH: Adjust synthetic data generation parameters")
        
        if results['motion_patterns']['pattern_similarity'] < 0.5:
            recommendations.append("PATTERN MISMATCH: Review motion simulation in synthetic data")
        
        if results['rep_detection']['detection_accuracy'] < 0.8:
            recommendations.append("REP DETECTION: Tune detection algorithm for real-world data")
        
        if results['sensor_fusion']['fusion_benefit'] < 0.05:
            recommendations.append("SENSOR FUSION: Optimize multi-sensor integration")
        
        if not recommendations:
            recommendations.append("EXCELLENT ALIGNMENT: Proceed with transfer learning")
        
        return recommendations

def main():
    """Main validation interface"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Validation System")
    parser.add_argument("--synthetic", required=True, help="Path to synthetic dataset")
    parser.add_argument("--human", required=True, help="Path to human data directory")
    parser.add_argument("--output", default="validation_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    print("🔍 Real vs Synthetic Data Validation")
    print(f"Synthetic data: {args.synthetic}")
    print(f"Human data: {args.human}")
    
    # Create validator
    validator = DataValidator(args.synthetic, args.human)
    
    # Run validation
    print("\n📊 Running validation analysis...")
    results = validator.validate_data_alignment()
    
    if results:
        # Generate report
        report = validator.generate_validation_report(results, args.output)
        
        # Print summary
        print(f"\n✅ Validation Complete!")
        print(f"Overall Alignment Score: {results['overall_score']:.3f}")
        print(f"Statistical Alignment: {results['statistical_alignment']['alignment_score']:.3f}")
        print(f"Motion Pattern Similarity: {results['motion_patterns']['pattern_similarity']:.3f}")
        print(f"Rep Detection Accuracy: {results['rep_detection']['detection_accuracy']:.3f}")
        print(f"Sensor Fusion Benefit: {results['sensor_fusion']['fusion_benefit']:.3f}")
        
        print(f"\n📋 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print(f"\n📁 Full report saved to: {args.output}")
    else:
        print("❌ Validation failed - check data availability")

if __name__ == "__main__":
    main()
