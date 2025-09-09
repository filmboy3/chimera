import Foundation
import Combine
import os.log

/**
 * CoachingEngine - Intelligent coaching logic for Sesame v2
 * 
 * Implements stateful coaching with context awareness, trend analysis,
 * and strategic feedback timing. Replaces reactive robotic responses
 * with intelligent, milestone-based coaching.
 */

@MainActor
class CoachingEngine: ObservableObject {
    
    // MARK: - Published State
    @Published private(set) var isSessionActive: Bool = false
    @Published private(set) var currentSessionStats: SessionStats = SessionStats()
    
    // MARK: - Coaching Output
    let coachingMessagePublisher = PassthroughSubject<String, Never>()
    
    // MARK: - Session Management
    private var sessionStartTime: Date?
    private var targetReps: Int = 0
    private var repHistory: [RepAnalysis] = []
    private var lastCoachingTime: Date = Date.distantPast
    private var milestonesSaid: Set<CoachingMilestone> = []
    
    // MARK: - Coaching Logic
    private let minCoachingInterval: TimeInterval = 3.0  // Minimum 3s between messages
    private let pauseDetectionThreshold: TimeInterval = 8.0  // 8s pause detection
    private let trendAnalysisWindow = 3  // Analyze last 3 reps for trends
    
    private let logger = Logger(subsystem: "com.chimera.coaching", category: "CoachingEngine")
    
    // MARK: - Session Control
    
    func startSession(targetReps: Int) async {
        self.targetReps = targetReps
        self.sessionStartTime = Date()
        self.isSessionActive = true
        self.repHistory.removeAll()
        self.milestonesSaid.removeAll()
        self.currentSessionStats = SessionStats()
        
        logger.info("🎯 Coaching session started - target: \(targetReps) reps")
        
        // Initial encouragement
        await sendCoachingMessage("Let's crush this set! Focus on your form.")
    }
    
    func endSession() async {
        isSessionActive = false
        
        if let startTime = sessionStartTime {
            let duration = Date().timeIntervalSince(startTime)
            logger.info("✅ Coaching session ended - duration: \(duration)s, reps: \(repHistory.count)")
        }
        
        // Session summary
        let completionRate = Double(repHistory.count) / Double(targetReps)
        let avgQuality = repHistory.isEmpty ? 0.0 : repHistory.map(\.qualityScore).reduce(0, +) / Double(repHistory.count)
        
        let summaryMessage = generateSessionSummary(
            repsCompleted: repHistory.count,
            targetReps: targetReps,
            avgQuality: avgQuality,
            completionRate: completionRate
        )
        
        await sendCoachingMessage(summaryMessage)
    }
    
    // MARK: - Rep Processing
    
    func processRep(context: CoachingContext) async {
        guard isSessionActive else { return }
        
        let repAnalysis = analyzeRep(context: context)
        repHistory.append(repAnalysis)
        
        // Update session stats
        updateSessionStats(with: repAnalysis)
        
        // Determine if coaching is needed
        let coachingDecision = shouldProvideCoaching(
            repAnalysis: repAnalysis,
            context: context
        )
        
        if let message = coachingDecision.message {
            await sendCoachingMessage(message)
        }
        
        logger.info("📊 Rep \(context.currentRep) processed - quality: \(repAnalysis.qualityScore:.2f), coaching: \(coachingDecision.message != nil)")
    }
    
    private func analyzeRep(context: CoachingContext) -> RepAnalysis {
        let timestamp = Date()
        
        // Calculate time since last rep
        let timeSinceLastRep: TimeInterval
        if let lastRep = repHistory.last {
            timeSinceLastRep = timestamp.timeIntervalSince(lastRep.timestamp)
        } else {
            timeSinceLastRep = 0
        }
        
        // Analyze rep quality components
        let qualityBreakdown = analyzeRepQuality(context: context)
        
        // Detect trends
        let trends = detectTrends(newQuality: context.repQuality)
        
        return RepAnalysis(
            repNumber: context.currentRep,
            timestamp: timestamp,
            qualityScore: context.repQuality,
            qualityBreakdown: qualityBreakdown,
            timeSinceLastRep: timeSinceLastRep,
            trends: trends,
            cognitiveState: context.cognitiveState
        )
    }
    
    private func analyzeRepQuality(context: CoachingContext) -> QualityBreakdown {
        // Simulate quality analysis based on rep score
        let baseQuality = context.repQuality
        
        return QualityBreakdown(
            depth: min(1.0, baseQuality + Double.random(in: -0.1...0.1)),
            speed: min(1.0, baseQuality + Double.random(in: -0.15...0.15)),
            smoothness: min(1.0, baseQuality + Double.random(in: -0.05...0.05)),
            symmetry: min(1.0, baseQuality + Double.random(in: -0.1...0.1))
        )
    }
    
    private func detectTrends(newQuality: Double) -> [RepTrend] {
        guard repHistory.count >= trendAnalysisWindow - 1 else { return [] }
        
        let recentQualities = repHistory.suffix(trendAnalysisWindow - 1).map(\.qualityScore) + [newQuality]
        var trends: [RepTrend] = []
        
        // Declining quality trend
        if recentQualities.count >= 3 {
            let isDecline = recentQualities[0] > recentQualities[1] && recentQualities[1] > recentQualities[2]
            if isDecline && recentQualities[2] < 0.7 {
                trends.append(.decliningQuality)
            }
        }
        
        // Shallow reps trend
        let shallowCount = recentQualities.filter { $0 < 0.6 }.count
        if shallowCount >= 2 {
            trends.append(.shallowReps)
        }
        
        // Fatigue indicators
        if newQuality < 0.5 && repHistory.count > targetReps / 2 {
            trends.append(.fatigueDetected)
        }
        
        return trends
    }
    
    // MARK: - Coaching Decision Logic
    
    private func shouldProvideCoaching(
        repAnalysis: RepAnalysis,
        context: CoachingContext
    ) -> CoachingDecision {
        
        // Check minimum interval
        let timeSinceLastCoaching = Date().timeIntervalSince(lastCoachingTime)
        guard timeSinceLastCoaching >= minCoachingInterval else {
            return CoachingDecision(shouldCoach: false, reason: .tooSoon, message: nil)
        }
        
        // Check for milestones
        if let milestone = checkMilestones(context: context) {
            return CoachingDecision(
                shouldCoach: true,
                reason: .milestone,
                message: generateMilestoneMessage(milestone, context: context)
            )
        }
        
        // Check for trends that need addressing
        if !repAnalysis.trends.isEmpty {
            let trendMessage = generateTrendMessage(trends: repAnalysis.trends, context: context)
            return CoachingDecision(
                shouldCoach: true,
                reason: .trendCorrection,
                message: trendMessage
            )
        }
        
        // Check for exceptional performance (outlier)
        if repAnalysis.qualityScore > 0.9 && context.currentRep > 1 {
            let avgQuality = repHistory.dropLast().map(\.qualityScore).reduce(0, +) / Double(repHistory.count - 1)
            if repAnalysis.qualityScore > avgQuality + 0.2 {
                return CoachingDecision(
                    shouldCoach: true,
                    reason: .exceptionalPerformance,
                    message: generateExceptionalMessage(context: context)
                )
            }
        }
        
        // Check for pause detection
        if repAnalysis.timeSinceLastRep > pauseDetectionThreshold {
            return CoachingDecision(
                shouldCoach: true,
                reason: .pauseDetected,
                message: generatePauseMessage(context: context)
            )
        }
        
        // Default: stay silent (great coaches know when NOT to speak)
        return CoachingDecision(shouldCoach: false, reason: .staySilent, message: nil)
    }
    
    private func checkMilestones(context: CoachingContext) -> CoachingMilestone? {
        let progress = Double(context.currentRep) / Double(context.targetReps)
        
        // Halfway point
        if progress >= 0.5 && !milestonesSaid.contains(.halfway) {
            milestonesSaid.insert(.halfway)
            return .halfway
        }
        
        // Three quarters
        if progress >= 0.75 && !milestonesSaid.contains(.threeQuarters) {
            milestonesSaid.insert(.threeQuarters)
            return .threeQuarters
        }
        
        // Final push
        if context.targetReps - context.currentRep <= 2 && !milestonesSaid.contains(.finalPush) {
            milestonesSaid.insert(.finalPush)
            return .finalPush
        }
        
        return nil
    }
    
    // MARK: - Message Generation
    
    private func generateMilestoneMessage(_ milestone: CoachingMilestone, context: CoachingContext) -> String {
        switch milestone {
        case .halfway:
            return "Halfway there! You're crushing it!"
        case .threeQuarters:
            return "Three quarters done - stay strong!"
        case .finalPush:
            return "Final push! You've got this!"
        }
    }
    
    private func generateTrendMessage(trends: [RepTrend], context: CoachingContext) -> String {
        if trends.contains(.decliningQuality) {
            return "Focus on your form - quality over speed!"
        } else if trends.contains(.shallowReps) {
            return "Go deeper on those squats - full range of motion!"
        } else if trends.contains(.fatigueDetected) {
            return "Feeling tired? Take a breath and reset your form."
        }
        return "Keep up the great work!"
    }
    
    private func generateExceptionalMessage(context: CoachingContext) -> String {
        let messages = [
            "Perfect form on that one!",
            "That's how it's done!",
            "Excellent depth and control!",
            "Beautiful rep!"
        ]
        return messages.randomElement() ?? "Great job!"
    }
    
    private func generatePauseMessage(context: CoachingContext) -> String {
        let messages = [
            "Take your time - quality matters more than speed.",
            "Good pause - reset and continue when ready.",
            "Breathing break? Smart move!"
        ]
        return messages.randomElement() ?? "Ready when you are!"
    }
    
    private func generateSessionSummary(
        repsCompleted: Int,
        targetReps: Int,
        avgQuality: Double,
        completionRate: Double
    ) -> String {
        
        if completionRate >= 1.0 {
            if avgQuality > 0.8 {
                return "Outstanding set! \(repsCompleted) perfect reps with excellent form!"
            } else {
                return "Great job completing all \(repsCompleted) reps! Focus on form next time."
            }
        } else if completionRate >= 0.8 {
            return "Solid effort! \(repsCompleted) out of \(targetReps) reps completed."
        } else {
            return "Good start! \(repsCompleted) reps done. Build up gradually."
        }
    }
    
    private func sendCoachingMessage(_ message: String) async {
        lastCoachingTime = Date()
        coachingMessagePublisher.send(message)
        logger.info("🗣️ Coaching message: '\(message)'")
    }
    
    // MARK: - Session Stats
    
    private func updateSessionStats(with repAnalysis: RepAnalysis) {
        currentSessionStats.totalReps = repHistory.count
        currentSessionStats.averageQuality = repHistory.map(\.qualityScore).reduce(0, +) / Double(repHistory.count)
        currentSessionStats.bestRep = repHistory.max(by: { $0.qualityScore < $1.qualityScore })?.qualityScore ?? 0.0
        currentSessionStats.worstRep = repHistory.min(by: { $0.qualityScore < $1.qualityScore })?.qualityScore ?? 0.0
        
        if let sessionStart = sessionStartTime {
            currentSessionStats.sessionDuration = Date().timeIntervalSince(sessionStart)
        }
    }
}

// MARK: - Supporting Types

struct RepAnalysis {
    let repNumber: Int
    let timestamp: Date
    let qualityScore: Double
    let qualityBreakdown: QualityBreakdown
    let timeSinceLastRep: TimeInterval
    let trends: [RepTrend]
    let cognitiveState: CognitiveState
}

struct QualityBreakdown {
    let depth: Double
    let speed: Double
    let smoothness: Double
    let symmetry: Double
}

enum RepTrend {
    case decliningQuality
    case shallowReps
    case fatigueDetected
}

enum CoachingMilestone {
    case halfway
    case threeQuarters
    case finalPush
}

struct CoachingDecision {
    let shouldCoach: Bool
    let reason: CoachingReason
    let message: String?
}

enum CoachingReason {
    case milestone
    case trendCorrection
    case exceptionalPerformance
    case pauseDetected
    case tooSoon
    case staySilent
}

struct SessionStats {
    var totalReps: Int = 0
    var averageQuality: Double = 0.0
    var bestRep: Double = 0.0
    var worstRep: Double = 0.0
    var sessionDuration: TimeInterval = 0.0
}
