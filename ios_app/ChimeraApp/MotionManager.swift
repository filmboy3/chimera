import Foundation
import CoreMotion
import Combine
import os.log

/**
 * MotionManager - Core Motion integration for multi-device IMU data
 * 
 * Handles iPhone and Apple Watch IMU data collection, processing,
 * and rep detection using adaptive thresholds and sustained motion logic.
 * Based on proven algorithms from QuantumLeap Validator project.
 */

@MainActor
class MotionManager: ObservableObject {
    
    // MARK: - Published State
    @Published private(set) var isMotionActive: Bool = false
    @Published private(set) var currentMotionIntensity: Double = 0.0
    @Published private(set) var repCount: Int = 0
    @Published private(set) var lastRepTimestamp: Date?
    
    // MARK: - Core Motion
    private let motionManager = CMMotionManager()
    private let altimeter = CMAltimeter()
    
    // MARK: - Rep Detection
    private var motionBuffer: [Double] = []
    private let bufferSize = 50  // 0.5 seconds at 100Hz
    private var baselineMotion: Double = 0.0
    private var adaptiveThreshold: Double = 0.15
    private var peakThreshold: Double = 0.25
    
    // MARK: - State Tracking
    private var isInRep = false
    private var repStartTime: Date?
    private var sustainedMotionCount = 0
    private let sustainedMotionRequired = 10  // 0.1 seconds of sustained motion
    
    // MARK: - Publishers
    let repDetectedPublisher = PassthroughSubject<RepData, Never>()
    let sensorDataPublisher = PassthroughSubject<SensorData, Never>()
    
    // MARK: - Logging
    private let logger = Logger(subsystem: "com.chimera.motion", category: "MotionManager")
    
    // MARK: - Motion Updates
    
    func startMotionUpdates() async {
        guard motionManager.isDeviceMotionAvailable else {
            logger.error("❌ Device motion not available")
            return
        }
        
        logger.info("🚀 Starting motion updates at 100Hz...")
        
        // Configure motion manager
        motionManager.deviceMotionUpdateInterval = 0.01  // 100Hz
        motionManager.showsDeviceMovementDisplay = true
        
        // Start device motion updates
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            guard let self = self, let motion = motion else {
                if let error = error {
                    self?.logger.error("❌ Motion update error: \(error.localizedDescription)")
                }
                return
            }
            
            Task { @MainActor in
                await self.processMotionData(motion)
            }
        }
        
        // Start altimeter updates for barometric pressure
        if CMAltimeter.isRelativeAltitudeAvailable() {
            altimeter.startRelativeAltitudeUpdates(to: .main) { [weak self] altitudeData, error in
                guard let self = self, let altitudeData = altitudeData else { return }
                
                Task { @MainActor in
                    await self.processAltitudeData(altitudeData)
                }
            }
        }
        
        isMotionActive = true
        logger.info("✅ Motion updates started successfully")
    }
    
    func stopMotionUpdates() async {
        logger.info("🛑 Stopping motion updates...")
        
        motionManager.stopDeviceMotionUpdates()
        altimeter.stopRelativeAltitudeUpdates()
        
        isMotionActive = false
        repCount = 0
        motionBuffer.removeAll()
        
        logger.info("✅ Motion updates stopped")
    }
    
    // MARK: - Motion Processing
    
    private func processMotionData(_ motion: CMDeviceMotion) async {
        let timestamp = Date()
        
        // Extract IMU data
        let acceleration = motion.userAcceleration
        let gyroscope = motion.rotationRate
        let magnetometer = motion.magneticField.field
        
        // Calculate motion intensity (jerk-based)
        let accelerationMagnitude = sqrt(
            acceleration.x * acceleration.x +
            acceleration.y * acceleration.y +
            acceleration.z * acceleration.z
        )
        
        let gyroscopeMagnitude = sqrt(
            gyroscope.x * gyroscope.x +
            gyroscope.y * gyroscope.y +
            gyroscope.z * gyroscope.z
        )
        
        // Combined motion intensity
        let motionIntensity = accelerationMagnitude + gyroscopeMagnitude * 0.1
        currentMotionIntensity = motionIntensity
        
        // Update motion buffer
        updateMotionBuffer(intensity: motionIntensity)
        
        // Perform rep detection
        await performRepDetection(
            intensity: motionIntensity,
            timestamp: timestamp
        )
        
        // Create sensor data
        let sensorData = SensorData(
            phoneAcceleration: [acceleration.x, acceleration.y, acceleration.z],
            phoneGyroscope: [gyroscope.x, gyroscope.y, gyroscope.z],
            phoneMagnetometer: [magnetometer.x, magnetometer.y, magnetometer.z],
            watchAcceleration: [0.0, 0.0, 0.0], // Placeholder - would come from Watch Connectivity
            barometricPressure: 1013.25, // Will be updated by altimeter
            audioFeatures: [0.0, 0.0], // Placeholder - would come from audio processing
            timestamp: timestamp
        )
        
        // Publish sensor data
        sensorDataPublisher.send(sensorData)
    }
    
    private func processAltitudeData(_ altitudeData: CMAltitudeData) async {
        // Convert relative altitude to approximate pressure
        // This is a simplified conversion - real implementation would be more sophisticated
        let pressureKPa = altitudeData.pressure.doubleValue
        let pressureHPa = pressureKPa * 10.0  // Convert kPa to hPa
        
        // Store for next sensor data update
        // In a real implementation, this would be properly synchronized
    }
    
    // MARK: - Rep Detection Algorithm
    
    private func updateMotionBuffer(intensity: Double) {
        motionBuffer.append(intensity)
        
        if motionBuffer.count > bufferSize {
            motionBuffer.removeFirst()
        }
        
        // Update baseline every 50 samples
        if motionBuffer.count == bufferSize {
            updateBaseline()
        }
    }
    
    private func updateBaseline() {
        let sortedBuffer = motionBuffer.sorted()
        let medianIndex = sortedBuffer.count / 2
        baselineMotion = sortedBuffer[medianIndex]
        
        // Adaptive thresholds based on baseline
        adaptiveThreshold = max(0.1, baselineMotion + 0.05)
        peakThreshold = max(0.15, baselineMotion + 0.12)
        
        logger.debug("📊 Baseline updated: \(baselineMotion:.3f), Thresholds: \(adaptiveThreshold:.3f)/\(peakThreshold:.3f)")
    }
    
    private func performRepDetection(intensity: Double, timestamp: Date) async {
        // Three detection patterns based on QuantumLeap Validator learnings
        
        // Pattern 1: Peak-based detection
        if intensity > peakThreshold && !isInRep {
            await startRep(intensity: intensity, timestamp: timestamp, pattern: .peak)
        }
        
        // Pattern 2: Sustained motion detection
        if intensity > adaptiveThreshold {
            sustainedMotionCount += 1
            if sustainedMotionCount >= sustainedMotionRequired && !isInRep {
                await startRep(intensity: intensity, timestamp: timestamp, pattern: .sustained)
            }
        } else {
            sustainedMotionCount = 0
        }
        
        // Pattern 3: Rep completion detection
        if isInRep && intensity < adaptiveThreshold {
            let repDuration = timestamp.timeIntervalSince(repStartTime ?? timestamp)
            if repDuration > 0.5 && repDuration < 5.0 {  // Valid rep duration
                await completeRep(intensity: intensity, timestamp: timestamp)
            }
        }
        
        // Reset detection if rep takes too long
        if let startTime = repStartTime, timestamp.timeIntervalSince(startTime) > 8.0 {
            logger.warning("⚠️ Rep timeout - resetting detection")
            isInRep = false
            repStartTime = nil
        }
    }
    
    private func startRep(intensity: Double, timestamp: Date, pattern: RepDetectionPattern) async {
        isInRep = true
        repStartTime = timestamp
        
        logger.info("🔄 Rep started - pattern: \(pattern), intensity: \(intensity:.3f)")
    }
    
    private func completeRep(intensity: Double, timestamp: Date) async {
        guard let startTime = repStartTime else { return }
        
        let repDuration = timestamp.timeIntervalSince(startTime)
        let maxIntensity = motionBuffer.max() ?? intensity
        
        // Calculate rep quality score
        let qualityScore = calculateRepQuality(
            duration: repDuration,
            maxIntensity: maxIntensity,
            avgIntensity: motionBuffer.reduce(0, +) / Double(motionBuffer.count)
        )
        
        // Create rep data
        let repData = RepData(
            timestamp: timestamp,
            imuData: createIMUSnapshot(),
            qualityScore: qualityScore,
            depth: min(1.0, maxIntensity / peakThreshold),
            speed: max(0.1, min(1.0, 3.0 / repDuration)),  // Optimal rep ~3 seconds
            smoothness: calculateSmoothness()
        )
        
        // Update state
        repCount += 1
        lastRepTimestamp = timestamp
        isInRep = false
        repStartTime = nil
        sustainedMotionCount = 0
        
        logger.info("✅ Rep #\(repCount) completed - quality: \(qualityScore:.2f), duration: \(repDuration:.1f)s")
        
        // Publish rep detection
        repDetectedPublisher.send(repData)
    }
    
    private func calculateRepQuality(duration: Double, maxIntensity: Double, avgIntensity: Double) -> Double {
        // Multi-factor quality scoring
        let durationScore = max(0.0, min(1.0, 1.0 - abs(duration - 3.0) / 3.0))  // Optimal ~3s
        let intensityScore = min(1.0, maxIntensity / (peakThreshold * 2.0))
        let consistencyScore = min(1.0, avgIntensity / maxIntensity)
        
        // Weighted combination
        return (durationScore * 0.3 + intensityScore * 0.5 + consistencyScore * 0.2)
    }
    
    private func calculateSmoothness() -> Double {
        guard motionBuffer.count > 1 else { return 1.0 }
        
        var totalVariation = 0.0
        for i in 1..<motionBuffer.count {
            totalVariation += abs(motionBuffer[i] - motionBuffer[i-1])
        }
        
        let avgVariation = totalVariation / Double(motionBuffer.count - 1)
        return max(0.0, min(1.0, 1.0 - avgVariation))
    }
    
    private func createIMUSnapshot() -> [Double] {
        // Create a snapshot of recent IMU data for perception engine
        // This would include the last few samples of all sensor data
        var snapshot: [Double] = []
        
        // Add recent motion intensities
        snapshot.append(contentsOf: motionBuffer.suffix(15))
        
        // Pad to expected size if needed
        while snapshot.count < 15 {
            snapshot.append(0.0)
        }
        
        return snapshot
    }
    
    // MARK: - Health Check
    
    func performHealthCheck() -> Bool {
        let motionHealthy = motionManager.isDeviceMotionActive
        let bufferHealthy = motionBuffer.count > 0
        let thresholdHealthy = adaptiveThreshold > 0 && peakThreshold > adaptiveThreshold
        
        let isHealthy = motionHealthy && bufferHealthy && thresholdHealthy
        
        if !isHealthy {
            logger.warning("⚠️ Motion manager health check failed")
            logger.info("Motion: \(motionHealthy), Buffer: \(bufferHealthy), Thresholds: \(thresholdHealthy)")
        }
        
        return isHealthy
    }
}

// MARK: - Supporting Types

enum RepDetectionPattern {
    case peak
    case sustained
    case combined
}
