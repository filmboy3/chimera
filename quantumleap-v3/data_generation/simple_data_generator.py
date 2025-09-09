#!/usr/bin/env python3
"""
Simplified synthetic data generator for QuantumLeap v3 training
Generates realistic IMU data for squat exercises without MuJoCo complexity
"""

import numpy as np
import h5py
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSquatGenerator:
    def __init__(self, num_samples=5000, sequence_length=200, sample_rate=100):
        self.num_samples = num_samples
        self.sequence_length = sequence_length  # 2 seconds at 100Hz
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        logger.info(f"Initialized SimpleSquatGenerator: {num_samples} samples, {sequence_length} timesteps")
    
    def generate_squat_motion(self, rep_count=3, form_quality=1.0, fatigue_level=0.0):
        """Generate realistic squat motion patterns"""
        
        # Time array
        t = np.linspace(0, self.sequence_length * self.dt, self.sequence_length)
        
        # Initialize IMU data arrays
        phone_accel = np.zeros((self.sequence_length, 3))  # x, y, z acceleration
        phone_gyro = np.zeros((self.sequence_length, 3))   # x, y, z angular velocity
        watch_accel = np.zeros((self.sequence_length, 3))
        watch_gyro = np.zeros((self.sequence_length, 3))
        barometer = np.zeros(self.sequence_length)
        
        # Generate rep cycles
        rep_duration = 1.5  # seconds per rep
        rest_duration = 0.5  # seconds between reps
        cycle_duration = rep_duration + rest_duration
        
        for rep in range(rep_count):
            start_time = rep * cycle_duration
            end_time = start_time + rep_duration
            
            # Find indices for this rep
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)
            
            if end_idx >= self.sequence_length:
                break
            
            # Generate squat motion for this rep
            rep_length = end_idx - start_idx
            rep_t = np.linspace(0, rep_duration, rep_length)
            
            # Squat motion: down phase (0-50%), up phase (50-100%)
            # Y-axis is primary motion direction (vertical)
            
            # Phone on thigh - primary motion sensor
            squat_cycle = np.sin(2 * np.pi * rep_t / rep_duration)  # Full cycle
            depth_factor = form_quality * (0.8 + 0.4 * np.random.random())  # Depth variation
            
            # Acceleration patterns
            phone_accel[start_idx:end_idx, 0] = 0.5 * np.sin(4 * np.pi * rep_t / rep_duration)  # X: lateral sway
            phone_accel[start_idx:end_idx, 1] = -2.0 * depth_factor * squat_cycle  # Y: main vertical motion
            phone_accel[start_idx:end_idx, 2] = 9.81 + 0.3 * squat_cycle  # Z: gravity + motion
            
            # Gyroscope patterns
            phone_gyro[start_idx:end_idx, 0] = 0.2 * np.cos(2 * np.pi * rep_t / rep_duration)  # Pitch
            phone_gyro[start_idx:end_idx, 1] = 0.1 * np.sin(4 * np.pi * rep_t / rep_duration)  # Roll
            phone_gyro[start_idx:end_idx, 2] = 0.05 * np.random.randn(rep_length)  # Yaw noise
            
            # Watch on wrist - secondary sensor with different motion
            watch_accel[start_idx:end_idx, 0] = 0.3 * squat_cycle  # Arm swing
            watch_accel[start_idx:end_idx, 1] = -0.5 * depth_factor * squat_cycle  # Reduced vertical
            watch_accel[start_idx:end_idx, 2] = 9.81 + 0.1 * squat_cycle
            
            watch_gyro[start_idx:end_idx, 0] = 0.1 * np.sin(2 * np.pi * rep_t / rep_duration)
            watch_gyro[start_idx:end_idx, 1] = 0.15 * np.cos(2 * np.pi * rep_t / rep_duration)
            watch_gyro[start_idx:end_idx, 2] = 0.02 * np.random.randn(rep_length)
            
            # Add fatigue effects (increasing jitter)
            fatigue_noise = fatigue_level * 0.1 * np.random.randn(rep_length, 3)
            phone_accel[start_idx:end_idx] += fatigue_noise
            phone_gyro[start_idx:end_idx] += fatigue_noise * 0.1
        
        # Add realistic noise
        accel_noise = 0.02 * np.random.randn(self.sequence_length, 3)
        gyro_noise = 0.01 * np.random.randn(self.sequence_length, 3)
        
        phone_accel += accel_noise
        phone_gyro += gyro_noise
        watch_accel += accel_noise * 0.5
        watch_gyro += gyro_noise * 0.5
        
        # Barometric pressure simulation
        baseline_pressure = 1013.25 + np.random.normal(0, 10)  # Altitude variation
        pressure_drift = 0.1 * t  # Slow drift
        pressure_noise = 0.5 * np.random.randn(self.sequence_length)
        barometer = baseline_pressure + pressure_drift + pressure_noise
        
        # Combine IMU data
        phone_imu = np.column_stack([phone_accel, phone_gyro])  # 6 channels
        watch_imu = np.column_stack([watch_accel, watch_gyro])  # 6 channels
        
        return phone_imu, watch_imu, barometer, rep_count
    
    def generate_labels(self, rep_count, form_quality, fatigue_level):
        """Generate multi-task learning labels"""
        
        # Rep detection labels (1 during rep, 0 during rest)
        rep_labels = np.zeros(self.sequence_length)
        rep_duration = 1.5 * self.sample_rate  # samples
        rest_duration = 0.5 * self.sample_rate
        
        for rep in range(rep_count):
            start_idx = int(rep * (rep_duration + rest_duration))
            end_idx = int(start_idx + rep_duration)
            if end_idx < self.sequence_length:
                rep_labels[start_idx:end_idx] = 1.0
        
        # Exercise classification (1 = squat, 0 = other)
        exercise_label = 1.0
        
        # Form quality score (0-1)
        quality_score = form_quality
        
        # Cognitive state (fatigue/focus)
        cognitive_state = np.array([fatigue_level, 1.0 - fatigue_level])  # [fatigue, focus]
        
        return {
            'rep_detection': rep_labels,
            'exercise_class': exercise_label,
            'form_quality': quality_score,
            'cognitive_state': cognitive_state
        }
    
    def generate_dataset(self, output_path):
        """Generate complete synthetic dataset"""
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        h5_file = output_path / "synthetic_squat_dataset.h5"
        
        logger.info(f"Generating {self.num_samples} samples...")
        
        with h5py.File(h5_file, 'w') as f:
            # Create datasets
            phone_imu_data = f.create_dataset('phone_imu', (self.num_samples, self.sequence_length, 6), dtype=np.float32)
            watch_imu_data = f.create_dataset('watch_imu', (self.num_samples, self.sequence_length, 6), dtype=np.float32)
            barometer_data = f.create_dataset('barometer', (self.num_samples, self.sequence_length), dtype=np.float32)
            
            # Label datasets
            rep_labels = f.create_dataset('rep_detection', (self.num_samples, self.sequence_length), dtype=np.float32)
            exercise_labels = f.create_dataset('exercise_class', (self.num_samples,), dtype=np.float32)
            quality_labels = f.create_dataset('form_quality', (self.num_samples,), dtype=np.float32)
            cognitive_labels = f.create_dataset('cognitive_state', (self.num_samples, 2), dtype=np.float32)
            
            # Metadata
            metadata = []
            
            # Generate samples
            for i in tqdm(range(self.num_samples)):
                # Sample parameters
                rep_count = np.random.randint(2, 6)  # 2-5 reps per sample
                form_quality = np.random.beta(2, 1)  # Bias toward good form
                fatigue_level = np.random.exponential(0.3)  # Most samples low fatigue
                fatigue_level = min(fatigue_level, 1.0)
                
                # Generate motion data
                phone_imu, watch_imu, barometer, actual_reps = self.generate_squat_motion(
                    rep_count, form_quality, fatigue_level
                )
                
                # Generate labels
                labels = self.generate_labels(actual_reps, form_quality, fatigue_level)
                
                # Store data
                phone_imu_data[i] = phone_imu
                watch_imu_data[i] = watch_imu
                barometer_data[i] = barometer
                
                rep_labels[i] = labels['rep_detection']
                exercise_labels[i] = labels['exercise_class']
                quality_labels[i] = labels['form_quality']
                cognitive_labels[i] = labels['cognitive_state']
                
                # Store metadata
                metadata.append({
                    'sample_id': i,
                    'rep_count': actual_reps,
                    'form_quality': form_quality,
                    'fatigue_level': fatigue_level,
                    'sequence_length': self.sequence_length
                })
            
            # Save metadata as JSON string attribute
            f.attrs['metadata'] = json.dumps(metadata)
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['sequence_length'] = self.sequence_length
            f.attrs['num_samples'] = self.num_samples
        
        logger.info(f"Dataset saved to {h5_file}")
        logger.info(f"Dataset size: {h5_file.stat().st_size / (1024*1024):.1f} MB")
        
        # Generate summary statistics
        self.print_dataset_summary(metadata)
        
        return str(h5_file)
    
    def print_dataset_summary(self, metadata):
        """Print dataset statistics"""
        
        rep_counts = [m['rep_count'] for m in metadata]
        form_qualities = [m['form_quality'] for m in metadata]
        fatigue_levels = [m['fatigue_level'] for m in metadata]
        
        logger.info("Dataset Summary:")
        logger.info(f"  Total samples: {len(metadata)}")
        logger.info(f"  Rep count - Mean: {np.mean(rep_counts):.1f}, Range: {min(rep_counts)}-{max(rep_counts)}")
        logger.info(f"  Form quality - Mean: {np.mean(form_qualities):.2f}, Std: {np.std(form_qualities):.2f}")
        logger.info(f"  Fatigue level - Mean: {np.mean(fatigue_levels):.2f}, Std: {np.std(fatigue_levels):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic squat dataset")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--length", type=int, default=200, help="Sequence length (timesteps)")
    parser.add_argument("--output", default="./data", help="Output directory")
    
    args = parser.parse_args()
    
    generator = SimpleSquatGenerator(
        num_samples=args.samples,
        sequence_length=args.length,
        sample_rate=100
    )
    
    dataset_path = generator.generate_dataset(args.output)
    logger.info(f"Training dataset ready: {dataset_path}")


if __name__ == "__main__":
    main()
