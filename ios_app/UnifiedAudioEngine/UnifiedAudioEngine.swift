import Foundation
import AVFoundation
import Speech
import os.log

/**
 * UnifiedAudioEngine - Single-owner audio session management
 * 
 * Critical architectural fix based on historical audio session conflicts.
 * Implements strict state machine to eliminate race conditions between
 * AVSpeechSynthesizer (TTS) and SFSpeechRecognizer + AVAudioEngine (ASR).
 */

enum AudioHardwareState {
    case idle
    case listening    // Speech recognition active
    case speaking     // Text-to-speech active
}

enum AudioEngineError: Error {
    case sessionConfigurationFailed
    case microphonePermissionDenied
    case speechRecognitionPermissionDenied
    case audioEngineStartFailed
    case invalidStateTransition(from: AudioHardwareState, to: AudioHardwareState)
    case concurrentAccessAttempt
}

@MainActor
class UnifiedAudioEngine: NSObject, ObservableObject {
    
    // MARK: - Singleton Pattern
    static let shared = UnifiedAudioEngine()
    
    // MARK: - Published State
    @Published private(set) var currentState: AudioHardwareState = .idle
    @Published private(set) var isSessionActive: Bool = false
    @Published private(set) var lastError: AudioEngineError?
    
    // MARK: - Audio Components
    private let audioSession = AVAudioSession.sharedInstance()
    private let speechSynthesizer = AVSpeechSynthesizer()
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let audioEngine = AVAudioEngine()
    
    // MARK: - Speech Recognition
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    
    // MARK: - State Management
    private let stateQueue = DispatchQueue(label: "com.chimera.audio.state", qos: .userInitiated)
    private var isTransitioning = false
    
    // MARK: - Logging
    private let logger = Logger(subsystem: "com.chimera.audio", category: "UnifiedAudioEngine")
    
    // MARK: - Callbacks
    var onSpeechRecognized: ((String) -> Void)?
    var onSpeechRecognitionError: ((Error) -> Void)?
    var onSpeechSynthesisComplete: (() -> Void)?
    
    private override init() {
        super.init()
        setupAudioSession()
        setupSpeechSynthesizer()
        logger.info("UnifiedAudioEngine initialized with strict state machine")
    }
    
    // MARK: - Audio Session Setup
    
    private func setupAudioSession() {
        do {
            // Configure for both playback and recording
            try audioSession.setCategory(
                .playAndRecord,
                mode: .default,
                options: [.defaultToSpeaker, .allowBluetooth, .allowBluetoothA2DP]
            )
            
            // Set preferred sample rate and buffer duration
            try audioSession.setPreferredSampleRate(16000) // Optimized for speech
            try audioSession.setPreferredIOBufferDuration(0.02) // 20ms for low latency
            
            logger.info("✅ Audio session configured successfully")
            
        } catch {
            logger.error("❌ Audio session configuration failed: \(error.localizedDescription)")
            lastError = .sessionConfigurationFailed
        }
    }
    
    private func setupSpeechSynthesizer() {
        speechSynthesizer.delegate = self
    }
    
    // MARK: - Permission Management
    
    func requestPermissions() async -> Bool {
        logger.info("🔐 Requesting audio permissions...")
        
        // Request microphone permission
        let micPermission = await withCheckedContinuation { continuation in
            audioSession.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
        
        guard micPermission else {
            logger.error("❌ Microphone permission denied")
            lastError = .microphonePermissionDenied
            return false
        }
        
        // Request speech recognition permission
        let speechPermission = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
        
        guard speechPermission else {
            logger.error("❌ Speech recognition permission denied")
            lastError = .speechRecognitionPermissionDenied
            return false
        }
        
        logger.info("✅ All audio permissions granted")
        return true
    }
    
    // MARK: - State Transitions
    
    func activateListening() async throws {
        try await performStateTransition(to: .listening) {
            try await self.startSpeechRecognition()
        }
    }
    
    func activateSpeaking(text: String) async throws {
        try await performStateTransition(to: .speaking) {
            try await self.startSpeechSynthesis(text: text)
        }
    }
    
    func deactivateAudio() async throws {
        try await performStateTransition(to: .idle) {
            await self.stopAllAudioOperations()
        }
    }
    
    private func performStateTransition(
        to newState: AudioHardwareState,
        operation: @escaping () async throws -> Void
    ) async throws {
        
        return try await withCheckedThrowingContinuation { continuation in
            stateQueue.async { [weak self] in
                guard let self = self else {
                    continuation.resume(throwing: AudioEngineError.concurrentAccessAttempt)
                    return
                }
                
                // Prevent concurrent transitions
                guard !self.isTransitioning else {
                    self.logger.warning("⚠️ State transition blocked - already transitioning")
                    continuation.resume(throwing: AudioEngineError.concurrentAccessAttempt)
                    return
                }
                
                // Validate transition
                guard self.isValidTransition(from: self.currentState, to: newState) else {
                    self.logger.error("❌ Invalid state transition: \(self.currentState) -> \(newState)")
                    continuation.resume(throwing: AudioEngineError.invalidStateTransition(
                        from: self.currentState, to: newState
                    ))
                    return
                }
                
                self.isTransitioning = true
                self.logger.info("🔄 State transition: \(self.currentState) -> \(newState)")
                
                Task { @MainActor in
                    do {
                        // Stop current operations first
                        await self.stopCurrentOperations()
                        
                        // Activate audio session if needed
                        if newState != .idle && !self.isSessionActive {
                            try self.audioSession.setActive(true)
                            self.isSessionActive = true
                            self.logger.info("✅ Audio session activated")
                        }
                        
                        // Perform the operation
                        try await operation()
                        
                        // Update state
                        self.currentState = newState
                        
                        // Deactivate session if going idle
                        if newState == .idle && self.isSessionActive {
                            try self.audioSession.setActive(false, options: .notifyOthersOnDeactivation)
                            self.isSessionActive = false
                            self.logger.info("✅ Audio session deactivated")
                        }
                        
                        self.stateQueue.async {
                            self.isTransitioning = false
                        }
                        
                        continuation.resume()
                        
                    } catch {
                        self.logger.error("❌ State transition failed: \(error.localizedDescription)")
                        self.lastError = error as? AudioEngineError
                        
                        self.stateQueue.async {
                            self.isTransitioning = false
                        }
                        
                        continuation.resume(throwing: error)
                    }
                }
            }
        }
    }
    
    private func isValidTransition(from: AudioHardwareState, to: AudioHardwareState) -> Bool {
        switch (from, to) {
        case (.idle, .listening), (.idle, .speaking):
            return true
        case (.listening, .idle), (.listening, .speaking):
            return true
        case (.speaking, .idle), (.speaking, .listening):
            return true
        case (let current, let new) where current == new:
            return true // Same state is valid
        default:
            return false
        }
    }
    
    // MARK: - Speech Recognition
    
    private func startSpeechRecognition() async throws {
        logger.info("🎤 Starting speech recognition...")
        
        // Cancel any existing recognition
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw AudioEngineError.audioEngineStartFailed
        }
        
        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = false
        
        // Configure audio engine
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        // Start audio engine
        audioEngine.prepare()
        try audioEngine.start()
        
        // Start recognition task
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            Task { @MainActor in
                if let result = result {
                    let recognizedText = result.bestTranscription.formattedString
                    self?.logger.info("🎤 Recognized: '\(recognizedText)'")
                    self?.onSpeechRecognized?(recognizedText)
                }
                
                if let error = error {
                    self?.logger.error("❌ Speech recognition error: \(error.localizedDescription)")
                    self?.onSpeechRecognitionError?(error)
                }
            }
        }
        
        logger.info("✅ Speech recognition started successfully")
    }
    
    private func stopSpeechRecognition() async {
        logger.info("🛑 Stopping speech recognition...")
        
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        
        recognitionTask?.cancel()
        recognitionTask = nil
        
        logger.info("✅ Speech recognition stopped")
    }
    
    // MARK: - Speech Synthesis
    
    private func startSpeechSynthesis(text: String) async throws {
        logger.info("🔊 Starting speech synthesis: '\(text)'")
        
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.5
        utterance.pitchMultiplier = 1.0
        utterance.volume = 0.8
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        
        speechSynthesizer.speak(utterance)
        
        logger.info("✅ Speech synthesis started")
    }
    
    private func stopSpeechSynthesis() async {
        logger.info("🛑 Stopping speech synthesis...")
        speechSynthesizer.stopSpeaking(at: .immediate)
        logger.info("✅ Speech synthesis stopped")
    }
    
    // MARK: - Operation Management
    
    private func stopCurrentOperations() async {
        switch currentState {
        case .listening:
            await stopSpeechRecognition()
        case .speaking:
            await stopSpeechSynthesis()
        case .idle:
            break
        }
    }
    
    private func stopAllAudioOperations() async {
        await stopSpeechRecognition()
        await stopSpeechSynthesis()
    }
    
    // MARK: - Convenience Methods
    
    func speak(_ text: String) async throws {
        try await activateSpeaking(text: text)
    }
    
    func startListening() async throws {
        try await activateListening()
    }
    
    func stopListening() async throws {
        try await deactivateAudio()
    }
    
    // MARK: - Health Check
    
    func performHealthCheck() -> Bool {
        let sessionHealthy = audioSession.isOtherAudioPlaying == false
        let engineHealthy = !audioEngine.isRunning || currentState == .listening
        let synthesizerHealthy = !speechSynthesizer.isSpeaking || currentState == .speaking
        
        let isHealthy = sessionHealthy && engineHealthy && synthesizerHealthy
        
        if !isHealthy {
            logger.warning("⚠️ Audio engine health check failed")
            logger.info("Session: \(sessionHealthy), Engine: \(engineHealthy), Synthesizer: \(synthesizerHealthy)")
        }
        
        return isHealthy
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension UnifiedAudioEngine: AVSpeechSynthesizerDelegate {
    
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor in
            logger.info("🔊 Speech synthesis started")
        }
    }
    
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            logger.info("✅ Speech synthesis completed")
            onSpeechSynthesisComplete?()
            
            // Auto-transition to idle after speech completes
            do {
                try await deactivateAudio()
            } catch {
                logger.error("❌ Failed to deactivate audio after speech completion: \(error)")
            }
        }
    }
    
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            logger.info("🛑 Speech synthesis cancelled")
            onSpeechSynthesisComplete?()
        }
    }
}
