#!/usr/bin/env python3
"""
Real Hardware Sensor Data Recorder
Connects to actual phone/watch sensors via network or USB
"""

import numpy as np
import json
import time
import socket
import threading
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealSensorRecorder:
    """Records from actual phone and watch sensors"""
    
    def __init__(self, output_dir="./human_data", session_duration=30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_duration = session_duration
        self.sample_rate = 100  # Hz
        self.is_recording = False
        
        # Data buffers
        self.phone_imu_buffer = []
        self.watch_imu_buffer = []
        self.barometer_buffer = []
        self.timestamps = []
        
        # Network connections
        self.phone_socket = None
        self.watch_socket = None
        
        # Connection settings
        self.phone_ip = "192.168.1.100"  # Your phone's IP
        self.phone_port = 8080
        self.watch_ip = "192.168.1.101"   # Your watch's IP  
        self.watch_port = 8081
        
        logger.info("Real sensor recorder initialized")
    
    def connect_to_sensors(self):
        """Connect to phone and watch via network"""
        
        try:
            # Connect to phone
            logger.info(f"Connecting to phone at {self.phone_ip}:{self.phone_port}")
            self.phone_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.phone_socket.connect((self.phone_ip, self.phone_port))
            logger.info("✅ Phone connected")
            
            # Connect to watch
            logger.info(f"Connecting to watch at {self.watch_ip}:{self.watch_port}")
            self.watch_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.watch_socket.connect((self.watch_ip, self.watch_port))
            logger.info("✅ Watch connected")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _recording_loop(self):
        """Main recording loop - reads from actual sensors"""
        
        start_time = time.time()
        
        while self.is_recording and (time.time() - start_time) < self.session_duration:
            current_time = time.time()
            
            try:
                # Read from phone sensor
                phone_imu = self._read_phone_sensor()
                
                # Read from watch sensor
                watch_imu = self._read_watch_sensor()
                
                # Read barometer (usually from phone)
                barometer = self._read_barometer()
                
                # Store data
                self.phone_imu_buffer.append(phone_imu)
                self.watch_imu_buffer.append(watch_imu)
                self.barometer_buffer.append(barometer)
                self.timestamps.append(current_time - start_time)
                
                # Maintain sample rate
                time.sleep(1.0 / self.sample_rate)
                
            except Exception as e:
                logger.error(f"Sensor read error: {e}")
                break
        
        self.is_recording = False
        logger.info("Recording session completed")
    
    def _read_phone_sensor(self):
        """Read IMU data from phone"""
        
        if not self.phone_socket:
            return [0, 0, 0, 0, 0, 0]  # Fallback
        
        try:
            # Send request for IMU data
            self.phone_socket.send(b"GET_IMU\n")
            
            # Receive response
            data = self.phone_socket.recv(1024).decode()
            values = json.loads(data)
            
            # Expected format: [ax, ay, az, gx, gy, gz]
            return values['imu']
            
        except Exception as e:
            logger.warning(f"Phone sensor read failed: {e}")
            return [0, 0, 0, 0, 0, 0]
    
    def _read_watch_sensor(self):
        """Read IMU data from watch"""
        
        if not self.watch_socket:
            return [0, 0, 0, 0, 0, 0]  # Fallback
        
        try:
            # Send request for IMU data
            self.watch_socket.send(b"GET_IMU\n")
            
            # Receive response
            data = self.watch_socket.recv(1024).decode()
            values = json.loads(data)
            
            return values['imu']
            
        except Exception as e:
            logger.warning(f"Watch sensor read failed: {e}")
            return [0, 0, 0, 0, 0, 0]
    
    def _read_barometer(self):
        """Read barometric pressure from phone"""
        
        if not self.phone_socket:
            return 1013.25  # Standard pressure
        
        try:
            self.phone_socket.send(b"GET_PRESSURE\n")
            data = self.phone_socket.recv(1024).decode()
            values = json.loads(data)
            
            return values['pressure']
            
        except Exception as e:
            logger.warning(f"Barometer read failed: {e}")
            return 1013.25

# Alternative: USB/ADB Connection Method
class ADBSensorRecorder:
    """Records sensors via Android Debug Bridge (USB)"""
    
    def __init__(self):
        self.adb_available = self._check_adb()
    
    def _check_adb(self):
        """Check if ADB is available"""
        import subprocess
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            return 'device' in result.stdout
        except FileNotFoundError:
            logger.error("ADB not found. Install Android SDK Platform Tools")
            return False
    
    def _read_phone_sensor_adb(self):
        """Read sensor via ADB shell commands"""
        import subprocess
        
        try:
            # Use ADB to read sensor data from Android app
            cmd = ['adb', 'shell', 'am', 'broadcast', '-a', 'com.chimera.GET_SENSORS']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse sensor data from broadcast result
            # This requires a companion Android app
            return self._parse_adb_sensor_data(result.stdout)
            
        except Exception as e:
            logger.error(f"ADB sensor read failed: {e}")
            return [0, 0, 0, 0, 0, 0]

# iOS Sensor Connection (requires companion iOS app)
class iOSSensorRecorder:
    """Records sensors from iOS devices via network"""
    
    def __init__(self):
        self.bonjour_service = None
    
    def discover_ios_devices(self):
        """Discover iOS devices on network via Bonjour"""
        try:
            import zeroconf
            
            class ServiceListener:
                def __init__(self):
                    self.devices = []
                
                def add_service(self, zeroconf, type, name):
                    info = zeroconf.get_service_info(type, name)
                    if info and 'chimera' in name.lower():
                        self.devices.append({
                            'name': name,
                            'address': socket.inet_ntoa(info.addresses[0]),
                            'port': info.port
                        })
            
            listener = ServiceListener()
            zc = zeroconf.Zeroconf()
            browser = zeroconf.ServiceBrowser(zc, "_chimera._tcp.local.", listener)
            
            time.sleep(3)  # Discovery time
            
            zc.close()
            return listener.devices
            
        except ImportError:
            logger.error("Install zeroconf: pip install zeroconf")
            return []

def main():
    """Demo of real sensor connection"""
    
    print("🔗 Real Sensor Connection Options:")
    print("1. Network connection (WiFi)")
    print("2. USB/ADB connection (Android)")
    print("3. Bonjour discovery (iOS)")
    print("4. Fallback to simulation")
    
    choice = input("Select option (1-4): ")
    
    if choice == "1":
        # Network connection
        recorder = RealSensorRecorder()
        
        # Get IP addresses
        phone_ip = input("Enter phone IP address (e.g., 192.168.1.100): ")
        watch_ip = input("Enter watch IP address (e.g., 192.168.1.101): ")
        
        recorder.phone_ip = phone_ip
        recorder.watch_ip = watch_ip
        
        if recorder.connect_to_sensors():
            print("✅ Sensors connected! Ready to record.")
        else:
            print("❌ Connection failed. Check network and companion apps.")
    
    elif choice == "2":
        # ADB connection
        recorder = ADBSensorRecorder()
        if recorder.adb_available:
            print("✅ ADB available. Connect Android device via USB.")
        else:
            print("❌ ADB not available. Install Android SDK Platform Tools.")
    
    elif choice == "3":
        # iOS Bonjour
        recorder = iOSSensorRecorder()
        devices = recorder.discover_ios_devices()
        
        if devices:
            print(f"✅ Found {len(devices)} iOS devices:")
            for device in devices:
                print(f"  - {device['name']} at {device['address']}:{device['port']}")
        else:
            print("❌ No iOS devices found. Ensure Chimera app is running.")
    
    else:
        # Fallback to simulation
        print("📱 Using simulation mode (no real sensors)")
        from human_squat_recorder import HumanSquatRecorder
        recorder = HumanSquatRecorder()

if __name__ == "__main__":
    main()
