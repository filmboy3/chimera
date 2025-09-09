import SwiftUI
import CoreMotion

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @State private var showingSettings = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Header
                headerView
                
                // Status Cards
                statusCardsView
                
                // Main Control Area
                mainControlView
                
                // Rep Counter Display
                repCounterView
                
                // Coaching Feedback
                coachingFeedbackView
                
                Spacer()
            }
            .padding()
            .navigationTitle("Chimera Coach")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Settings") {
                        showingSettings = true
                    }
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView()
            }
        }
    }
    
    // MARK: - Header View
    
    private var headerView: some View {
        VStack(spacing: 8) {
            Text("Project Chimera Ascendant")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)
            
            Text("Embodied AI Coach")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.primary)
        }
    }
    
    // MARK: - Status Cards
    
    private var statusCardsView: some View {
        HStack(spacing: 16) {
            StatusCard(
                title: "Connection",
                value: connectionStatusText,
                color: connectionStatusColor,
                icon: "antenna.radiowaves.left.and.right"
            )
            
            StatusCard(
                title: "Audio",
                value: appState.isAudioSessionActive ? "Active" : "Idle",
                color: appState.isAudioSessionActive ? .green : .gray,
                icon: "speaker.wave.2"
            )
            
            StatusCard(
                title: "Cognitive",
                value: String(format: "%.1f", appState.cognitiveState.focusLevel),
                color: cognitiveStateColor,
                icon: "brain.head.profile"
            )
        }
    }
    
    // MARK: - Main Control View
    
    private var mainControlView: some View {
        VStack(spacing: 20) {
            // Workout State Indicator
            workoutStateIndicator
            
            // Control Buttons
            controlButtonsView
        }
        .padding(.vertical)
    }
    
    private var workoutStateIndicator: some View {
        HStack {
            Circle()
                .fill(workoutStateColor)
                .frame(width: 12, height: 12)
            
            Text(workoutStateText)
                .font(.headline)
                .fontWeight(.medium)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(workoutStateColor.opacity(0.1))
        .cornerRadius(20)
    }
    
    private var controlButtonsView: some View {
        HStack(spacing: 20) {
            switch appState.workoutState {
            case .idle, .completed:
                Button("Start Workout") {
                    Task {
                        await appState.startWorkout(targetReps: 10)
                    }
                }
                .buttonStyle(PrimaryButtonStyle())
                
            case .countdown:
                Button("Starting...") { }
                    .buttonStyle(PrimaryButtonStyle())
                    .disabled(true)
                
            case .exercising:
                Button("Pause") {
                    Task {
                        await appState.pauseWorkout()
                    }
                }
                .buttonStyle(SecondaryButtonStyle())
                
                Button("Stop") {
                    Task {
                        await appState.stopWorkout()
                    }
                }
                .buttonStyle(DestructiveButtonStyle())
                
            case .paused:
                Button("Resume") {
                    Task {
                        await appState.resumeWorkout()
                    }
                }
                .buttonStyle(PrimaryButtonStyle())
                
                Button("Stop") {
                    Task {
                        await appState.stopWorkout()
                    }
                }
                .buttonStyle(DestructiveButtonStyle())
            }
        }
    }
    
    // MARK: - Rep Counter View
    
    private var repCounterView: some View {
        VStack(spacing: 12) {
            Text("Reps")
                .font(.headline)
                .foregroundColor(.secondary)
            
            HStack(alignment: .lastTextBaseline, spacing: 8) {
                Text("\(appState.currentRepCount)")
                    .font(.system(size: 64, weight: .bold, design: .rounded))
                    .foregroundColor(.primary)
                
                Text("/ \(appState.targetReps)")
                    .font(.title2)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
            }
            
            // Progress Bar
            ProgressView(value: Double(appState.currentRepCount), total: Double(appState.targetReps))
                .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                .scaleEffect(y: 2)
            
            // Rep Quality Score
            if appState.repQualityScore > 0 {
                HStack {
                    Text("Quality:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(String(format: "%.1f%%", appState.repQualityScore * 100))
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(qualityScoreColor)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }
    
    // MARK: - Coaching Feedback View
    
    private var coachingFeedbackView: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "message.badge.filled.fill")
                    .foregroundColor(.blue)
                
                Text("AI Coach")
                    .font(.headline)
                    .fontWeight(.semibold)
            }
            
            if !appState.lastCoachingMessage.isEmpty {
                Text(appState.lastCoachingMessage)
                    .font(.body)
                    .foregroundColor(.primary)
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(12)
            } else {
                Text("Ready to coach you through your workout!")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .italic()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 2, x: 0, y: 1)
    }
    
    // MARK: - Computed Properties
    
    private var connectionStatusText: String {
        switch appState.connectionStatus {
        case .disconnected: return "Offline"
        case .connecting: return "Connecting"
        case .connected: return "Connected"
        case .error: return "Error"
        }
    }
    
    private var connectionStatusColor: Color {
        switch appState.connectionStatus {
        case .disconnected: return .gray
        case .connecting: return .orange
        case .connected: return .green
        case .error: return .red
        }
    }
    
    private var workoutStateText: String {
        switch appState.workoutState {
        case .idle: return "Ready"
        case .countdown: return "Starting..."
        case .exercising: return "Exercising"
        case .paused: return "Paused"
        case .completed: return "Completed"
        }
    }
    
    private var workoutStateColor: Color {
        switch appState.workoutState {
        case .idle: return .gray
        case .countdown: return .orange
        case .exercising: return .green
        case .paused: return .yellow
        case .completed: return .purple
        }
    }
    
    private var cognitiveStateColor: Color {
        let focus = appState.cognitiveState.focusLevel
        if focus > 0.8 { return .green }
        else if focus > 0.5 { return .orange }
        else { return .red }
    }
    
    private var qualityScoreColor: Color {
        let score = appState.repQualityScore
        if score > 0.8 { return .green }
        else if score > 0.6 { return .orange }
        else { return .red }
    }
}

// MARK: - Status Card Component

struct StatusCard: View {
    let title: String
    let value: String
    let color: Color
    let icon: String
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            Text(value)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(color)
            
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Button Styles

struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .fontWeight(.semibold)
            .foregroundColor(.white)
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(Color.blue)
            .cornerRadius(12)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .fontWeight(.semibold)
            .foregroundColor(.blue)
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(Color.blue.opacity(0.1))
            .cornerRadius(12)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

struct DestructiveButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .fontWeight(.semibold)
            .foregroundColor(.white)
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(Color.red)
            .cornerRadius(12)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List {
                Section("Workout Settings") {
                    HStack {
                        Text("Target Reps")
                        Spacer()
                        Text("10")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Rest Time")
                        Spacer()
                        Text("60s")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("Audio Settings") {
                    Toggle("Voice Coaching", isOn: .constant(true))
                    Toggle("Sound Effects", isOn: .constant(true))
                }
                
                Section("Model Settings") {
                    HStack {
                        Text("Perception Model")
                        Spacer()
                        Text("QuantumLeap v3")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Coaching Engine")
                        Spacer()
                        Text("Sesame v2")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("About") {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0 (Phase 1)")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Build")
                        Spacer()
                        Text("Chimera Ascendant PoC")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(AppState())
}
