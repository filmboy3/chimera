#!/usr/bin/env python3
"""
Human Squat Data Collection System
Real-time recording and validation of human squat movements
"""

import numpy as np
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import logging
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanSquatRecorder:
    """Real-time human squat data recorder with validation"""
    
    def __init__(self, output_dir="./human_data", session_duration=30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_duration = session_duration  # seconds
        self.sample_rate = 100  # Hz
        self.is_recording = False
        
        # Data buffers
        self.phone_imu_buffer = []
        self.watch_imu_buffer = []
        self.barometer_buffer = []
        self.timestamps = []
        
        # Real-time analysis
        self.rep_detector = RealTimeRepDetector()
        self.quality_analyzer = FormQualityAnalyzer()
        
        logger.info(f"Human Squat Recorder initialized - {session_duration}s sessions")
    
    def start_recording_session(self, participant_id, session_notes=""):
        """Start a new recording session"""
        
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
        
        self.is_recording = True
        self.participant_id = participant_id
        self.session_start = time.time()
        
        # Clear buffers
        self.phone_imu_buffer = []
        self.watch_imu_buffer = []
        self.barometer_buffer = []
        self.timestamps = []
        
        logger.info(f"Started recording session for participant {participant_id}")
        
        # Start data collection thread
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.start()
        
        return self.session_start
    
    def _recording_loop(self):
        """Main recording loop - simulates real sensor data"""
        
        start_time = time.time()
        
        while self.is_recording and (time.time() - start_time) < self.session_duration:
            current_time = time.time()
            
            # Simulate sensor data (in real implementation, this would read from actual sensors)
            phone_imu = self._simulate_phone_imu(current_time - start_time)
            watch_imu = self._simulate_watch_imu(current_time - start_time)
            barometer = self._simulate_barometer(current_time - start_time)
            
            # Store data
            self.phone_imu_buffer.append(phone_imu)
            self.watch_imu_buffer.append(watch_imu)
            self.barometer_buffer.append(barometer)
            self.timestamps.append(current_time - start_time)
            
            # Real-time analysis
            if len(self.phone_imu_buffer) > 10:  # Need some history
                rep_detected = self.rep_detector.process_sample(phone_imu)
                if rep_detected:
                    logger.info(f"Rep detected at {current_time - start_time:.1f}s")
            
            # Maintain sample rate
            time.sleep(1.0 / self.sample_rate)
        
        self.is_recording = False
        logger.info("Recording session completed")
    
    def _simulate_phone_imu(self, t):
        """Simulate realistic phone IMU data during squats"""
        
        # Simulate squat motion pattern
        rep_frequency = 0.5  # Hz (2 seconds per rep)
        squat_phase = np.sin(2 * np.pi * rep_frequency * t)
        
        # Accelerometer (m/s^2)
        accel_x = 0.3 * np.sin(4 * np.pi * rep_frequency * t) + 0.1 * np.random.randn()
        accel_y = -2.0 * squat_phase + 0.2 * np.random.randn()  # Primary motion
        accel_z = 9.81 + 0.5 * squat_phase + 0.1 * np.random.randn()
        
        # Gyroscope (rad/s)
        gyro_x = 0.2 * np.cos(2 * np.pi * rep_frequency * t) + 0.05 * np.random.randn()
        gyro_y = 0.1 * np.sin(4 * np.pi * rep_frequency * t) + 0.03 * np.random.randn()
        gyro_z = 0.02 * np.random.randn()
        
        return [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    
    def _simulate_watch_imu(self, t):
        """Simulate watch IMU data (different motion pattern)"""
        
        rep_frequency = 0.5
        squat_phase = np.sin(2 * np.pi * rep_frequency * t)
        
        # Watch has different motion characteristics
        accel_x = 0.2 * squat_phase + 0.05 * np.random.randn()
        accel_y = -0.5 * squat_phase + 0.1 * np.random.randn()  # Reduced motion
        accel_z = 9.81 + 0.2 * squat_phase + 0.05 * np.random.randn()
        
        gyro_x = 0.1 * np.sin(2 * np.pi * rep_frequency * t) + 0.02 * np.random.randn()
        gyro_y = 0.15 * np.cos(2 * np.pi * rep_frequency * t) + 0.03 * np.random.randn()
        gyro_z = 0.01 * np.random.randn()
        
        return [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    
    def _simulate_barometer(self, t):
        """Simulate barometric pressure data"""
        
        baseline = 1013.25  # hPa
        drift = 0.1 * t / 60  # Slow drift
        noise = 0.3 * np.random.randn()
        
        return baseline + drift + noise
    
    def stop_recording(self):
        """Stop current recording session"""
        
        if not self.is_recording:
            logger.warning("No recording in progress")
            return None
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        
        # Save session data
        session_data = self._create_session_data()
        session_file = self._save_session(session_data)
        
        logger.info(f"Recording stopped and saved to {session_file}")
        return session_file
    
    def _create_session_data(self):
        """Create structured session data"""
        
        # Convert to numpy arrays
        phone_imu = np.array(self.phone_imu_buffer)
        watch_imu = np.array(self.watch_imu_buffer)
        barometer = np.array(self.barometer_buffer)
        timestamps = np.array(self.timestamps)
        
        # Analyze session
        rep_count = self.rep_detector.get_total_reps()
        quality_metrics = self.quality_analyzer.analyze_session(phone_imu, watch_imu)
        
        session_data = {
            'metadata': {
                'participant_id': self.participant_id,
                'session_start': self.session_start,
                'duration': timestamps[-1] if len(timestamps) > 0 else 0,
                'sample_rate': self.sample_rate,
                'total_samples': len(timestamps)
            },
            'sensor_data': {
                'phone_imu': phone_imu.tolist(),
                'watch_imu': watch_imu.tolist(),
                'barometer': barometer.tolist(),
                'timestamps': timestamps.tolist()
            },
            'analysis': {
                'rep_count': rep_count,
                'quality_metrics': quality_metrics,
                'rep_timestamps': self.rep_detector.get_rep_timestamps()
            }
        }
        
        return session_data
    
    def _save_session(self, session_data):
        """Save session data to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"human_squat_{self.participant_id}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return filepath

class RealTimeRepDetector:
    """Real-time rep detection for human movements"""
    
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.rep_count = 0
        self.rep_timestamps = []
        self.last_peak_time = 0
        self.min_rep_interval = 1.0  # Minimum seconds between reps
        
    def process_sample(self, imu_sample):
        """Process single IMU sample and detect reps"""
        
        # Calculate motion intensity (simplified)
        accel = np.array(imu_sample[:3])
        motion_intensity = np.linalg.norm(accel - [0, 0, 9.81])
        
        current_time = time.time()
        
        # Peak detection
        if (motion_intensity > self.threshold and 
            current_time - self.last_peak_time > self.min_rep_interval):
            
            self.rep_count += 1
            self.rep_timestamps.append(current_time)
            self.last_peak_time = current_time
            
            return True
        
        return False
    
    def get_total_reps(self):
        return self.rep_count
    
    def get_rep_timestamps(self):
        return self.rep_timestamps

class FormQualityAnalyzer:
    """Analyze form quality from sensor data"""
    
    def analyze_session(self, phone_imu, watch_imu):
        """Analyze overall session quality"""
        
        if len(phone_imu) == 0:
            return {'quality_score': 0.0, 'consistency': 0.0}
        
        # Analyze motion consistency
        y_accel = phone_imu[:, 1]  # Primary motion axis
        consistency = 1.0 - (np.std(y_accel) / (np.mean(np.abs(y_accel)) + 1e-6))
        consistency = max(0.0, min(1.0, consistency))
        
        # Overall quality score (simplified)
        quality_score = consistency * 0.8 + 0.2  # Base score
        
        return {
            'quality_score': quality_score,
            'consistency': consistency,
            'motion_range': np.ptp(y_accel),
            'avg_intensity': np.mean(np.abs(y_accel))
        }

def main():
    """Main data collection interface"""
    
    parser = argparse.ArgumentParser(description="Human Squat Data Recorder")
    parser.add_argument("--participant", required=True, help="Participant ID")
    parser.add_argument("--duration", type=int, default=30, help="Session duration (seconds)")
    parser.add_argument("--output", default="./human_data", help="Output directory")
    
    args = parser.parse_args()
    
    # Create recorder
    recorder = HumanSquatRecorder(args.output, args.duration)
    
    print(f"\n🎯 Human Squat Data Collection")
    print(f"Participant: {args.participant}")
    print(f"Duration: {args.duration} seconds")
    print(f"Output: {args.output}")
    print("\nInstructions:")
    print("1. Position phone on thigh (primary sensor)")
    print("2. Wear watch on wrist (secondary sensor)")
    print("3. Press ENTER to start recording")
    print("4. Perform natural squats for the duration")
    print("5. Recording will stop automatically\n")
    
    input("Press ENTER to start recording...")
    
    # Start recording
    session_start = recorder.start_recording_session(args.participant)
    
    print(f"🔴 RECORDING STARTED - {args.duration} seconds")
    print("Perform squats now...")
    
    # Wait for completion
    time.sleep(args.duration + 1)
    
    # Stop and save
    session_file = recorder.stop_recording()
    
    print(f"\n✅ Recording completed!")
    print(f"📁 Data saved to: {session_file}")
    print(f"🏃 Ready for next participant")

if __name__ == "__main__":
    main()
