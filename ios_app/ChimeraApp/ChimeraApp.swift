import SwiftUI
import CoreML
import AVFoundation
import CoreMotion
import Combine

@main
struct ChimeraApp: App {
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .onAppear {
                    Task {
                        await appState.initialize()
                    }
                }
        }
    }
}

@MainActor
class AppState: ObservableObject {
    
    // MARK: - Published State
    @Published var workoutState: WorkoutState = .idle
    @Published var currentRepCount: Int = 0
    @Published var targetReps: Int = 10
    @Published var repQualityScore: Double = 0.0
    @Published var cognitiveState: CognitiveState = CognitiveState()
    @Published var isAudioSessionActive: Bool = false
    @Published var lastCoachingMessage: String = ""
    @Published var connectionStatus: ConnectionStatus = .disconnected
    
    // MARK: - Core Components
    private let audioEngine = UnifiedAudioEngine.shared
    private let motionManager = MotionManager()
    private let perceptionEngine = PerceptionEngine()
    private let coachingEngine = CoachingEngine()
    
    // MARK: - Cancellables
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    
    func initialize() async {
        setupBindings()
        await setupAudioPermissions()
        await setupMotionTracking()
        await loadPerceptionModel()
    }
    
    private func setupBindings() {
        // Bind audio engine state
        audioEngine.$currentState
            .map { $0 != .idle }
            .assign(to: &$isAudioSessionActive)
        
        // Bind motion manager updates
        motionManager.repDetectedPublisher
            .sink { [weak self] repData in
                Task { @MainActor in
                    await self?.handleRepDetected(repData)
                }
            }
            .store(in: &cancellables)
        
        // Bind coaching messages
        coachingEngine.coachingMessagePublisher
            .assign(to: &$lastCoachingMessage)
            .store(in: &cancellables)
    }
    
    private func setupAudioPermissions() async {
        let granted = await audioEngine.requestPermissions()
        if !granted {
            print("❌ Audio permissions denied")
        }
    }
    
    private func setupMotionTracking() async {
        await motionManager.startMotionUpdates()
    }
    
    private func loadPerceptionModel() async {
        do {
            try await perceptionEngine.loadModel()
            connectionStatus = .connected
        } catch {
            print("❌ Failed to load perception model: \(error)")
            connectionStatus = .error
        }
    }
    
    // MARK: - Workout Control
    
    func startWorkout(targetReps: Int = 10) async {
        self.targetReps = targetReps
        self.currentRepCount = 0
        
        workoutState = .countdown
        
        // Start countdown with audio feedback
        try? await audioEngine.speak("Starting workout in 3")
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        
        try? await audioEngine.speak("2")
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        
        try? await audioEngine.speak("1")
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        
        try? await audioEngine.speak("Go!")
        
        workoutState = .exercising
        
        // Start perception pipeline
        await perceptionEngine.startRealTimeInference()
        await coachingEngine.startSession(targetReps: targetReps)
    }
    
    func pauseWorkout() async {
        workoutState = .paused
        await perceptionEngine.pauseInference()
        try? await audioEngine.speak("Workout paused")
    }
    
    func resumeWorkout() async {
        workoutState = .exercising
        await perceptionEngine.resumeInference()
        try? await audioEngine.speak("Resuming workout")
    }
    
    func stopWorkout() async {
        workoutState = .idle
        await perceptionEngine.stopInference()
        await coachingEngine.endSession()
        
        let completionMessage = currentRepCount >= targetReps ? 
            "Great job! Workout completed with \(currentRepCount) reps!" :
            "Workout stopped at \(currentRepCount) reps"
        
        try? await audioEngine.speak(completionMessage)
    }
    
    // MARK: - Rep Detection Handling
    
    private func handleRepDetected(_ repData: RepData) async {
        currentRepCount += 1
        
        // Update rep quality score
        repQualityScore = repData.qualityScore
        
        // Process through perception engine
        let perceptionOutput = await perceptionEngine.processRepData(repData)
        
        // Update cognitive state
        cognitiveState = perceptionOutput.cognitiveState
        
        // Generate coaching feedback
        let coachingContext = CoachingContext(
            currentRep: currentRepCount,
            targetReps: targetReps,
            repQuality: repData.qualityScore,
            cognitiveState: cognitiveState,
            workoutHistory: []
        )
        
        await coachingEngine.processRep(context: coachingContext)
        
        // Check for workout completion
        if currentRepCount >= targetReps {
            await completeWorkout()
        }
    }
    
    private func completeWorkout() async {
        workoutState = .completed
        await perceptionEngine.stopInference()
        await coachingEngine.endSession()
        
        try? await audioEngine.speak("Excellent work! Set completed with \(currentRepCount) reps!")
    }
}

// MARK: - Supporting Types

enum WorkoutState {
    case idle
    case countdown
    case exercising
    case paused
    case completed
}

enum ConnectionStatus {
    case disconnected
    case connecting
    case connected
    case error
}

struct CognitiveState {
    var fatigueLevel: Double = 0.0
    var focusLevel: Double = 1.0
    var interruptionCost: Double = 0.0
}

struct RepData {
    let timestamp: Date
    let imuData: [Double]
    let qualityScore: Double
    let depth: Double
    let speed: Double
    let smoothness: Double
}

struct PerceptionOutput {
    let poseEstimation: [Double]
    let exerciseClassification: [Double]
    let repDetection: Double
    let cognitiveState: CognitiveState
}

struct CoachingContext {
    let currentRep: Int
    let targetReps: Int
    let repQuality: Double
    let cognitiveState: CognitiveState
    let workoutHistory: [RepData]
}
