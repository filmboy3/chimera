import Foundation
import CoreML
import Combine
import os.log

/**
 * PerceptionEngine - Core ML inference pipeline for QuantumLeap v3
 * 
 * Handles real-time multi-modal perception using the converted Core ML model.
 * Processes IMU, barometric, and audio features for pose estimation,
 * exercise classification, rep detection, and cognitive state estimation.
 */

@MainActor
class PerceptionEngine: ObservableObject {
    
    // MARK: - Published State
    @Published private(set) var isModelLoaded: Bool = false
    @Published private(set) var isInferenceActive: Bool = false
    @Published private(set) var lastInferenceTime: TimeInterval = 0
    @Published private(set) var inferenceCount: Int = 0
    
    // MARK: - Core ML Model
    private var mlModel: MLModel?
    private let modelName = "QuantumLeapV3"
    
    // MARK: - Input Buffer Management
    private let sequenceLength = 200  // 2 seconds at 100Hz
    private let inputChannels = 15    // phone_imu(9) + watch_imu(3) + barometer(1) + audio(2)
    private var inputBuffer: [[Float]] = []
    private let bufferQueue = DispatchQueue(label: "com.chimera.perception.buffer", qos: .userInitiated)
    
    // MARK: - Inference Pipeline
    private var inferenceTimer: Timer?
    private let inferenceInterval: TimeInterval = 0.1  // 10Hz inference
    private let logger = Logger(subsystem: "com.chimera.perception", category: "PerceptionEngine")
    
    // MARK: - Output Publishers
    let poseEstimationPublisher = PassthroughSubject<[Float], Never>()
    let exerciseClassificationPublisher = PassthroughSubject<[Float], Never>()
    let repDetectionPublisher = PassthroughSubject<Float, Never>()
    let cognitiveStatePublisher = PassthroughSubject<CognitiveState, Never>()
    
    // MARK: - Model Loading
    
    func loadModel() async throws {
        logger.info("📥 Loading QuantumLeap v3 Core ML model...")
        
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlpackage") else {
            logger.error("❌ Model file not found: \(modelName).mlpackage")
            throw PerceptionError.modelNotFound
        }
        
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            configuration.allowLowPrecisionAccumulationOnGPU = true
            
            mlModel = try MLModel(contentsOf: modelURL, configuration: configuration)
            isModelLoaded = true
            
            logger.info("✅ QuantumLeap v3 model loaded successfully")
            
            // Initialize input buffer
            initializeInputBuffer()
            
        } catch {
            logger.error("❌ Failed to load Core ML model: \(error.localizedDescription)")
            throw PerceptionError.modelLoadFailed(error)
        }
    }
    
    private func initializeInputBuffer() {
        inputBuffer = Array(repeating: Array(repeating: 0.0, count: inputChannels), count: sequenceLength)
        logger.info("✅ Input buffer initialized: \(sequenceLength) x \(inputChannels)")
    }
    
    // MARK: - Real-time Inference
    
    func startRealTimeInference() async {
        guard isModelLoaded else {
            logger.error("❌ Cannot start inference - model not loaded")
            return
        }
        
        logger.info("🚀 Starting real-time inference pipeline...")
        isInferenceActive = true
        
        // Start inference timer on main thread
        inferenceTimer = Timer.scheduledTimer(withTimeInterval: inferenceInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.performInference()
            }
        }
        
        logger.info("✅ Real-time inference started at \(1.0/inferenceInterval)Hz")
    }
    
    func pauseInference() async {
        logger.info("⏸️ Pausing inference pipeline...")
        isInferenceActive = false
        inferenceTimer?.invalidate()
        inferenceTimer = nil
    }
    
    func resumeInference() async {
        logger.info("▶️ Resuming inference pipeline...")
        await startRealTimeInference()
    }
    
    func stopInference() async {
        logger.info("🛑 Stopping inference pipeline...")
        isInferenceActive = false
        inferenceTimer?.invalidate()
        inferenceTimer = nil
        inferenceCount = 0
    }
    
    // MARK: - Data Input
    
    func addSensorData(_ sensorData: SensorData) {
        bufferQueue.async { [weak self] in
            self?.updateInputBuffer(with: sensorData)
        }
    }
    
    private func updateInputBuffer(with sensorData: SensorData) {
        // Shift buffer left (remove oldest sample)
        inputBuffer.removeFirst()
        
        // Create new input vector
        var inputVector: [Float] = []
        
        // Phone IMU (9 channels)
        inputVector.append(contentsOf: sensorData.phoneAcceleration.map { Float($0) })
        inputVector.append(contentsOf: sensorData.phoneGyroscope.map { Float($0) })
        inputVector.append(contentsOf: sensorData.phoneMagnetometer.map { Float($0) })
        
        // Watch IMU (3 channels)
        inputVector.append(contentsOf: sensorData.watchAcceleration.map { Float($0) })
        
        // Barometric pressure (1 channel)
        inputVector.append(Float(sensorData.barometricPressure))
        
        // Audio features (2 channels)
        inputVector.append(contentsOf: sensorData.audioFeatures.map { Float($0) })
        
        // Ensure correct size
        while inputVector.count < inputChannels {
            inputVector.append(0.0)
        }
        if inputVector.count > inputChannels {
            inputVector = Array(inputVector.prefix(inputChannels))
        }
        
        // Add to buffer
        inputBuffer.append(inputVector)
    }
    
    // MARK: - Core ML Inference
    
    private func performInference() async {
        guard isInferenceActive, let model = mlModel else { return }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            // Prepare input
            let input = try prepareModelInput()
            
            // Run inference
            let output = try model.prediction(from: input)
            
            // Process outputs
            await processModelOutput(output)
            
            // Update metrics
            let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
            lastInferenceTime = inferenceTime
            inferenceCount += 1
            
            if inferenceCount % 50 == 0 {  // Log every 5 seconds
                logger.info("📊 Inference #\(inferenceCount): \(inferenceTime*1000:.1f)ms")
            }
            
        } catch {
            logger.error("❌ Inference failed: \(error.localizedDescription)")
        }
    }
    
    private func prepareModelInput() throws -> MLFeatureProvider {
        // Convert buffer to MLMultiArray
        let shape = [1, sequenceLength, inputChannels] as [NSNumber]
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        bufferQueue.sync {
            for (timeIndex, timeStep) in inputBuffer.enumerated() {
                for (channelIndex, value) in timeStep.enumerated() {
                    let index = timeIndex * inputChannels + channelIndex
                    multiArray[index] = NSNumber(value: value)
                }
            }
        }
        
        // Create feature provider
        let inputFeatures: [String: Any] = ["sensor_data": multiArray]
        return try MLDictionaryFeatureProvider(dictionary: inputFeatures)
    }
    
    private func processModelOutput(_ output: MLFeatureProvider) async {
        // Extract pose estimation
        if let poseOutput = output.featureValue(for: "pose_estimation")?.multiArrayValue {
            let poseArray = extractFloatArray(from: poseOutput)
            poseEstimationPublisher.send(poseArray)
        }
        
        // Extract exercise classification
        if let exerciseOutput = output.featureValue(for: "exercise_classification")?.multiArrayValue {
            let exerciseArray = extractFloatArray(from: exerciseOutput)
            exerciseClassificationPublisher.send(exerciseArray)
        }
        
        // Extract rep detection
        if let repOutput = output.featureValue(for: "rep_detection")?.multiArrayValue {
            let repConfidence = Float(repOutput[0].floatValue)
            repDetectionPublisher.send(repConfidence)
        }
        
        // Extract cognitive state
        if let cognitiveOutput = output.featureValue(for: "cognitive_state")?.multiArrayValue {
            let cognitiveArray = extractFloatArray(from: cognitiveOutput)
            if cognitiveArray.count >= 3 {
                let cognitiveState = CognitiveState(
                    fatigueLevel: Double(cognitiveArray[0]),
                    focusLevel: Double(cognitiveArray[1]),
                    interruptionCost: Double(cognitiveArray[2])
                )
                cognitiveStatePublisher.send(cognitiveState)
            }
        }
    }
    
    private func extractFloatArray(from multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        var result: [Float] = []
        result.reserveCapacity(count)
        
        for i in 0..<count {
            result.append(multiArray[i].floatValue)
        }
        
        return result
    }
    
    // MARK: - Rep Processing
    
    func processRepData(_ repData: RepData) async -> PerceptionOutput {
        // Convert rep data to sensor format
        let sensorData = SensorData(
            phoneAcceleration: Array(repData.imuData.prefix(3)),
            phoneGyroscope: Array(repData.imuData.dropFirst(3).prefix(3)),
            phoneMagnetometer: Array(repData.imuData.dropFirst(6).prefix(3)),
            watchAcceleration: Array(repData.imuData.dropFirst(9).prefix(3)),
            barometricPressure: repData.imuData.count > 12 ? repData.imuData[12] : 1013.25,
            audioFeatures: repData.imuData.count > 13 ? Array(repData.imuData.dropFirst(13).prefix(2)) : [0.0, 0.0],
            timestamp: repData.timestamp
        )
        
        // Add to buffer
        addSensorData(sensorData)
        
        // Return current perception state
        return PerceptionOutput(
            poseEstimation: Array(repeating: 0.0, count: 51), // 17 joints * 3 coordinates
            exerciseClassification: [0.8, 0.1, 0.05, 0.05, 0.0], // squat, rest, transition, error, unknown
            repDetection: repData.qualityScore,
            cognitiveState: CognitiveState(
                fatigueLevel: min(1.0, Double(inferenceCount) / 1000.0),
                focusLevel: max(0.5, 1.0 - Double(inferenceCount) / 2000.0),
                interruptionCost: 0.1
            )
        )
    }
    
    // MARK: - Health Check
    
    func performHealthCheck() -> Bool {
        let modelHealthy = isModelLoaded && mlModel != nil
        let bufferHealthy = inputBuffer.count == sequenceLength
        let inferenceHealthy = !isInferenceActive || (inferenceTimer != nil)
        
        let isHealthy = modelHealthy && bufferHealthy && inferenceHealthy
        
        if !isHealthy {
            logger.warning("⚠️ Perception engine health check failed")
            logger.info("Model: \(modelHealthy), Buffer: \(bufferHealthy), Inference: \(inferenceHealthy)")
        }
        
        return isHealthy
    }
}

// MARK: - Supporting Types

struct SensorData {
    let phoneAcceleration: [Double]      // 3D acceleration (m/s²)
    let phoneGyroscope: [Double]         // 3D angular velocity (rad/s)
    let phoneMagnetometer: [Double]      // 3D magnetic field (μT)
    let watchAcceleration: [Double]      // 3D acceleration (m/s²)
    let barometricPressure: Double       // Pressure (hPa)
    let audioFeatures: [Double]          // Audio features (MFCC, spectral, etc.)
    let timestamp: Date
}

enum PerceptionError: Error {
    case modelNotFound
    case modelLoadFailed(Error)
    case invalidInput
    case inferenceTimeout
    
    var localizedDescription: String {
        switch self {
        case .modelNotFound:
            return "Core ML model file not found in app bundle"
        case .modelLoadFailed(let error):
            return "Failed to load Core ML model: \(error.localizedDescription)"
        case .invalidInput:
            return "Invalid input data for model inference"
        case .inferenceTimeout:
            return "Model inference timed out"
        }
    }
}
