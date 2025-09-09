#!/usr/bin/env python3
"""
Project Chimera Ascendant - Synthetic Data Generation Engine
Generates 500K+ squat samples with placement invariance and barometric fusion
"""

import numpy as np
import mujoco
import yaml
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import h5py
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SquatDataGenerator:
    """
    MuJoCo-based synthetic data generator for squat exercises with:
    - Placement invariance via aggressive sensor randomization
    - Barometric pressure simulation with realistic noise
    - Fatigue modeling through progressive jitter increase
    - Form error injection (knee valgus, forward lean, insufficient depth)
    """
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.sim_config = self.config['simulation']
        self.aug_config = self.config['augmentation']
        self.error_config = self.config['form_errors']
        
        # Initialize MuJoCo model
        self.model_path = self._create_squat_model()
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Sensor attachment points (thigh and wrist for phone/watch)
        self.sensor_sites = {
            'phone': 'thigh_sensor',  # Primary IMU location
            'watch': 'wrist_sensor'   # Secondary IMU location
        }
        
        logger.info(f"Initialized SquatDataGenerator for {self.dataset_config['num_samples']} samples")
    
    def _create_squat_model(self) -> str:
        """Create MuJoCo XML model for squat simulation with sensor attachment points"""
        
        model_xml = """
        <mujoco model="squat_humanoid">
          <compiler angle="degree" inertiafromgeom="true"/>
          
          <default>
            <joint armature="1" damping="1" limited="true"/>
            <geom conaffinity="0" friction="1 0.1 0.005" rgba="0.8 0.6 0.4 1"/>
          </default>
          
          <worldbody>
            <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
            <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
            
            <!-- Humanoid body -->
            <body name="torso" pos="0 0 1.4">
              <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0.342 0.940"/>
              <joint name="root" type="free"/>
              <geom name="torso" size="0.25 0.15" type="capsule"/>
              <geom name="head" pos="0 0 0.19" size="0.09" type="sphere"/>
              
              <!-- Arms -->
              <body name="upper_arm_right" pos="0 -0.17 0.06">
                <joint axis="0 0 1" name="shoulder_right" pos="0 0 0" range="-85 60" type="hinge"/>
                <geom name="uarm_right" size="0.04 0.16" type="capsule"/>
                <body name="lower_arm_right" pos="0 0 -0.18">
                  <joint axis="0 1 0" name="elbow_right" pos="0 0 0.02" range="-90 50" type="hinge"/>
                  <geom name="larm_right" size="0.031 0.14" type="capsule"/>
                  <body name="hand_right" pos="0 0 -0.16">
                    <geom name="hand_right" size="0.04 0.06" type="capsule"/>
                    <!-- Wrist sensor attachment point -->
                    <site name="wrist_sensor" pos="0 0 0" size="0.01" rgba="0 1 0 1"/>
                  </body>
                </body>
              </body>
              
              <body name="upper_arm_left" pos="0 0.17 0.06">
                <joint axis="0 0 1" name="shoulder_left" pos="0 0 0" range="-60 85" type="hinge"/>
                <geom name="uarm_left" size="0.04 0.16" type="capsule"/>
                <body name="lower_arm_left" pos="0 0 -0.18">
                  <joint axis="0 1 0" name="elbow_left" pos="0 0 0.02" range="-90 50" type="hinge"/>
                  <geom name="larm_left" size="0.031 0.14" type="capsule"/>
                  <body name="hand_left" pos="0 0 -0.16">
                    <geom name="hand_left" size="0.04 0.06" type="capsule"/>
                  </body>
                </body>
              </body>
              
              <!-- Legs -->
              <body name="upper_leg_right" pos="0 -0.11 -0.25">
                <joint axis="0 0 1" name="hip_right" pos="0 0 0" range="-30 70" type="hinge"/>
                <geom name="uleg_right" size="0.06 0.18" type="capsule"/>
                <!-- Thigh sensor attachment point (primary phone location) -->
                <site name="thigh_sensor" pos="0.08 0 -0.1" size="0.02" rgba="1 0 0 1"/>
                <body name="lower_leg_right" pos="0 0 -0.2">
                  <joint axis="0 1 0" name="knee_right" pos="0 0 0.02" range="-160 -2" type="hinge"/>
                  <geom name="lleg_right" size="0.049 0.15" type="capsule"/>
                  <body name="foot_right" pos="0 0 -0.17">
                    <geom name="foot_right" pos="0.06 0 0.1" size="0.075 0.05 0.02" type="box"/>
                  </body>
                </body>
              </body>
              
              <body name="upper_leg_left" pos="0 0.11 -0.25">
                <joint axis="0 0 1" name="hip_left" pos="0 0 0" range="-70 30" type="hinge"/>
                <geom name="uleg_left" size="0.06 0.18" type="capsule"/>
                <body name="lower_leg_left" pos="0 0 -0.2">
                  <joint axis="0 1 0" name="knee_left" pos="0 0 0.02" range="-160 -2" type="hinge"/>
                  <geom name="lleg_left" size="0.049 0.15" type="capsule"/>
                  <body name="foot_left" pos="0 0 -0.17">
                    <geom name="foot_left" pos="0.06 0 0.1" size="0.075 0.05 0.02" type="box"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          
          <!-- Actuators for controlled movement -->
          <actuator>
            <motor gear="150" joint="hip_right"/>
            <motor gear="150" joint="hip_left"/>
            <motor gear="100" joint="knee_right"/>
            <motor gear="100" joint="knee_left"/>
          </actuator>
          
          <!-- Sensors -->
          <sensor>
            <!-- IMU sensors at attachment points -->
            <accelerometer name="phone_accel" site="thigh_sensor"/>
            <gyro name="phone_gyro" site="thigh_sensor"/>
            <accelerometer name="watch_accel" site="wrist_sensor"/>
            <gyro name="watch_gyro" site="wrist_sensor"/>
            
            <!-- Joint position sensors -->
            <jointpos name="hip_right_pos" joint="hip_right"/>
            <jointpos name="hip_left_pos" joint="hip_left"/>
            <jointpos name="knee_right_pos" joint="knee_right"/>
            <jointpos name="knee_left_pos" joint="knee_left"/>
          </sensor>
        </mujoco>
        """
        
        # Save model to temporary file
        model_path = "/tmp/squat_model.xml"
        with open(model_path, 'w') as f:
            f.write(model_xml)
        
        return model_path
    
    def _generate_squat_trajectory(self, form_errors: Dict[str, float] = None) -> Tuple[np.ndarray, Dict]:
        """Generate a single squat trajectory with optional form errors"""
        
        if form_errors is None:
            form_errors = {}
        
        duration = self.sim_config['duration']
        timestep = self.sim_config['timestep']
        n_steps = int(duration / timestep)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Generate squat motion pattern
        time_array = np.linspace(0, duration, n_steps)
        
        # Base squat pattern: 3-4 reps over 30 seconds
        n_reps = np.random.randint(3, 5)
        rep_duration = duration / n_reps
        
        trajectory_data = {
            'time': time_array,
            'joint_positions': np.zeros((n_steps, 4)),  # hip_right, hip_left, knee_right, knee_left
            'phone_imu': np.zeros((n_steps, 6)),  # accel_x,y,z + gyro_x,y,z
            'watch_imu': np.zeros((n_steps, 6)),
            'barometer': np.zeros(n_steps),
            'labels': {
                'exercise_type': 'squat',
                'rep_count': n_reps,
                'form_errors': form_errors,
                'fatigue_progression': np.zeros(n_steps)
            }
        }
        
        for step in range(n_steps):
            t = time_array[step]
            
            # Generate squat motion
            rep_phase = (t % rep_duration) / rep_duration
            
            # Squat depth (0 = standing, 1 = bottom of squat)
            if rep_phase < 0.4:  # Descending
                squat_depth = rep_phase / 0.4
            elif rep_phase < 0.6:  # Bottom hold
                squat_depth = 1.0
            else:  # Ascending
                squat_depth = 1.0 - (rep_phase - 0.6) / 0.4
            
            # Apply form errors
            if 'insufficient_depth' in form_errors:
                squat_depth *= (1.0 - form_errors['insufficient_depth'])
            
            # Base joint angles
            hip_angle = -30 * squat_depth  # Hip flexion
            knee_angle = -120 * squat_depth  # Knee flexion
            
            # Apply knee valgus error
            if 'knee_valgus' in form_errors:
                valgus_offset = 15 * form_errors['knee_valgus'] * squat_depth
                trajectory_data['joint_positions'][step] = [
                    hip_angle, hip_angle,
                    knee_angle + valgus_offset, knee_angle - valgus_offset
                ]
            else:
                trajectory_data['joint_positions'][step] = [
                    hip_angle, hip_angle, knee_angle, knee_angle
                ]
            
            # Apply joint positions to model
            self.data.qpos[7:11] = np.radians(trajectory_data['joint_positions'][step])
            
            # Forward lean error affects torso orientation
            if 'forward_lean' in form_errors:
                lean_angle = 20 * form_errors['forward_lean'] * squat_depth
                # Apply lean to root body orientation (simplified)
                self.data.qpos[4] = np.radians(lean_angle)
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Extract sensor data with placement randomization
            phone_data, watch_data = self._extract_sensor_data_with_randomization()
            trajectory_data['phone_imu'][step] = phone_data
            trajectory_data['watch_imu'][step] = watch_data
            
            # Simulate barometric pressure (height-based)
            base_pressure = 101325  # Sea level pressure in Pa
            height = self.data.qpos[2]  # Z position of root body
            pressure = base_pressure - (height * 12.0)  # Approximate pressure gradient
            
            # Add barometric noise
            if self.aug_config['barometer_simulation']['enabled']:
                noise_std = self.aug_config['barometer_simulation']['noise_std']
                drift_rate = self.aug_config['barometer_simulation']['drift_rate']
                pressure += np.random.normal(0, noise_std) + drift_rate * t
            
            trajectory_data['barometer'][step] = pressure
            
            # Fatigue modeling - increase jitter over time
            if self.aug_config['fatigue_modeling']['enabled']:
                rep_number = int(t / rep_duration)
                fatigue_factor = min(1.0, rep_number * self.aug_config['fatigue_modeling']['jitter_increase_rate'])
                trajectory_data['labels']['fatigue_progression'][step] = fatigue_factor
                
                # Apply fatigue to sensor data
                if fatigue_factor > 0:
                    jitter_mult = 1.0 + fatigue_factor * (self.aug_config['fatigue_modeling']['max_jitter_multiplier'] - 1.0)
                    trajectory_data['phone_imu'][step] += np.random.normal(0, 0.1 * jitter_mult, 6)
                    trajectory_data['watch_imu'][step] += np.random.normal(0, 0.1 * jitter_mult, 6)
        
        return trajectory_data
    
    def _extract_sensor_data_with_randomization(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract IMU data with aggressive placement randomization"""
        
        # Get base sensor readings
        phone_accel = self.data.sensordata[0:3]  # phone accelerometer
        phone_gyro = self.data.sensordata[3:6]   # phone gyroscope
        watch_accel = self.data.sensordata[6:9]  # watch accelerometer
        watch_gyro = self.data.sensordata[9:12]  # watch gyroscope
        
        # Apply placement invariance randomization
        if self.aug_config['placement_randomization']['enabled']:
            pos_var = self.aug_config['placement_randomization']['position_variance']
            ori_var = np.radians(self.aug_config['placement_randomization']['orientation_variance'])
            
            # Random position offset (simulates phone shifting in pocket)
            pos_offset = np.random.normal(0, pos_var, 3)
            
            # Random orientation offset (simulates phone orientation variation)
            ori_offset = np.random.normal(0, ori_var, 3)
            
            # Apply rotation matrix to sensor data (simplified)
            rotation_noise = np.random.normal(1.0, 0.05, (3, 3))  # Small rotation perturbation
            phone_accel = rotation_noise @ phone_accel
            phone_gyro = rotation_noise @ phone_gyro
            watch_accel = rotation_noise @ watch_accel
            watch_gyro = rotation_noise @ watch_gyro
        
        return np.concatenate([phone_accel, phone_gyro]), np.concatenate([watch_accel, watch_gyro])
    
    def _sample_form_errors(self) -> Dict[str, float]:
        """Sample form errors based on configured probabilities"""
        
        errors = {}
        
        for error_type, config in self.error_config.items():
            if np.random.random() < config['probability']:
                severity = np.random.uniform(*config['severity_range'])
                errors[error_type] = severity
        
        return errors
    
    def generate_dataset(self, output_dir: str) -> None:
        """Generate the complete synthetic dataset"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        total_samples = self.dataset_config['num_samples']
        dataset_name = self.dataset_config['name']
        
        logger.info(f"Generating {total_samples} samples for {dataset_name}")
        
        # Create HDF5 file for efficient storage
        h5_path = output_path / f"{dataset_name}.h5"
        
        with h5py.File(h5_path, 'w') as h5f:
            # Pre-allocate datasets
            max_timesteps = int(self.sim_config['duration'] / self.sim_config['timestep'])
            
            h5f.create_dataset('phone_imu', (total_samples, max_timesteps, 6), dtype=np.float32)
            h5f.create_dataset('watch_imu', (total_samples, max_timesteps, 6), dtype=np.float32)
            h5f.create_dataset('barometer', (total_samples, max_timesteps), dtype=np.float32)
            h5f.create_dataset('joint_positions', (total_samples, max_timesteps, 4), dtype=np.float32)
            h5f.create_dataset('fatigue_labels', (total_samples, max_timesteps), dtype=np.float32)
            
            # Metadata
            metadata = []
            
            # Generate samples
            for i in tqdm(range(total_samples), desc="Generating samples"):
                # Sample form errors for this trajectory
                form_errors = self._sample_form_errors()
                
                # Generate trajectory
                trajectory = self._generate_squat_trajectory(form_errors)
                
                # Store in HDF5
                h5f['phone_imu'][i] = trajectory['phone_imu']
                h5f['watch_imu'][i] = trajectory['watch_imu']
                h5f['barometer'][i] = trajectory['barometer']
                h5f['joint_positions'][i] = trajectory['joint_positions']
                h5f['fatigue_labels'][i] = trajectory['labels']['fatigue_progression']
                
                # Store metadata
                sample_metadata = {
                    'sample_id': i,
                    'exercise_type': trajectory['labels']['exercise_type'],
                    'rep_count': trajectory['labels']['rep_count'],
                    'form_errors': trajectory['labels']['form_errors']
                }
                metadata.append(sample_metadata)
            
            # Save metadata as JSON
            metadata_path = output_path / f"{dataset_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save generation config
            config_path = output_path / f"{dataset_name}_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Dataset generation complete! Saved to {h5_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Generate summary statistics
        self._generate_dataset_summary(h5_path, metadata_path)
    
    def _generate_dataset_summary(self, h5_path: Path, metadata_path: Path) -> None:
        """Generate and log dataset summary statistics"""
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Count form errors
        error_counts = {}
        for sample in metadata:
            for error_type in sample['form_errors']:
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Rep count distribution
        rep_counts = [sample['rep_count'] for sample in metadata]
        
        logger.info("Dataset Summary:")
        logger.info(f"  Total samples: {len(metadata)}")
        logger.info(f"  Average reps per sample: {np.mean(rep_counts):.1f}")
        logger.info(f"  Form error distribution:")
        for error_type, count in error_counts.items():
            percentage = (count / len(metadata)) * 100
            logger.info(f"    {error_type}: {count} samples ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic squat dataset for QuantumLeap v3")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--samples", type=int, help="Override number of samples")
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.samples:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['dataset']['num_samples'] = args.samples
        
        # Save modified config
        temp_config = "/tmp/modified_config.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        args.config = temp_config
    
    # Generate dataset
    generator = SquatDataGenerator(args.config)
    generator.generate_dataset(args.output)


if __name__ == "__main__":
    main()
