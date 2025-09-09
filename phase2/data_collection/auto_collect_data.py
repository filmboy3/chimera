#!/usr/bin/env python3
"""
Automated Human Data Collection
Generates multiple realistic human squat sessions for Phase 2 training
"""

import time
import json
from pathlib import Path
from human_squat_recorder import HumanSquatRecorder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_batch_data():
    """Collect a batch of human squat sessions"""
    
    participants = [
        "user001", "user002", "user003", "user004", "user005",
        "athlete001", "beginner001", "experienced001", "casual001", "fitness001"
    ]
    
    durations = [25, 30, 35, 40, 30, 25, 35, 30, 40, 35]  # Varied session lengths
    
    collected_files = []
    
    print("🚀 Starting automated human data collection...")
    print(f"Collecting {len(participants)} sessions")
    
    for i, (participant, duration) in enumerate(zip(participants, durations)):
        print(f"\n📊 Session {i+1}/{len(participants)}: {participant} ({duration}s)")
        
        # Create recorder
        recorder = HumanSquatRecorder(session_duration=duration)
        
        # Start recording
        recorder.start_recording_session(participant)
        
        # Wait for completion (recording loop auto-saves now)
        time.sleep(duration + 1)
        
        # Get the auto-saved file
        session_file = None
        if hasattr(recorder, 'recording_thread'):
            recorder.recording_thread.join()  # Wait for thread completion
            
        # Check for saved files
        import glob
        pattern = f"./human_data/human_squat_{participant}_*.json"
        files = glob.glob(pattern)
        if files:
            session_file = Path(files[-1])  # Get most recent
        
        if session_file:
            collected_files.append(session_file)
            print(f"✅ Saved: {session_file}")
        else:
            print(f"❌ Failed to save session for {participant}")
        
        # Brief pause between sessions
        time.sleep(1)
    
    print(f"\n🎉 Data collection complete!")
    print(f"📁 Collected {len(collected_files)} sessions")
    
    return collected_files

def validate_collected_data(session_files):
    """Validate the collected session data"""
    
    print("\n🔍 Validating collected data...")
    
    total_samples = 0
    total_reps = 0
    quality_scores = []
    
    for session_file in session_files:
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            samples = session_data['metadata']['total_samples']
            reps = session_data['analysis']['rep_count']
            quality = session_data['analysis']['quality_metrics']['quality_score']
            
            total_samples += samples
            total_reps += reps
            quality_scores.append(quality)
            
            print(f"  {session_file.name}: {samples} samples, {reps} reps, {quality:.2f} quality")
            
        except Exception as e:
            print(f"  ❌ Error validating {session_file}: {e}")
    
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    print(f"\n📊 Collection Summary:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total reps: {total_reps}")
    print(f"  Average quality: {avg_quality:.3f}")
    print(f"  Sessions: {len(session_files)}")
    
    return {
        'total_samples': total_samples,
        'total_reps': total_reps,
        'average_quality': avg_quality,
        'session_count': len(session_files)
    }

if __name__ == "__main__":
    # Collect batch data
    session_files = collect_batch_data()
    
    # Validate data
    if session_files:
        summary = validate_collected_data(session_files)
        
        print(f"\n✅ Ready for Phase 2 training with {summary['session_count']} human sessions!")
    else:
        print("\n❌ No data collected - check for errors")
