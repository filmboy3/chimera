# Project Chimera Ascendant

**The World's First Embodied AI Coach**

A revolutionary system combining multi-modal perception (IMU + barometric + audio) with cognitive coaching AI to create personalized, real-time fitness guidance.

## Architecture Overview

### Parallel Development Strategy
- **QuantumLeap v3**: Multi-modal perception engine (physics-based simulation → pose estimation)
- **Sesame v2**: Cognitive coaching core (fatigue detection → contextual dialogue)

### Core Innovation
- **Synthetic Data Supremacy**: 500K+ MuJoCo-generated samples with placement invariance
- **Embodied Cognition**: Real-time fusion of motion + audio cues for intelligent coaching
- **Sub-120ms Latency**: Sensor-to-cue pipeline optimized for responsive feedback

## Repository Structure

```
ChimeraAscendant/
├── quantumleap-v3/           # Perception Engine
│   ├── data_generation/      # MuJoCo synthetic data pipeline
│   ├── models/              # CNN→VQ-VAE→Transformer architecture
│   ├── training/            # PyTorch Lightning training scripts
│   └── evaluation/          # Benchmarking against MobilePoser
├── sesame-v2/               # Cognitive Core
│   ├── audio_pipeline/      # PANNs sound classification
│   ├── intent_recognition/  # MobileBERT intent classification
│   ├── cognitive_modulator/ # Fatigue/focus state estimation
│   └── llm_endpoint/        # Server-side coaching LLM
├── ios_app/                 # iOS Integration
│   ├── UnifiedAudioEngine/  # Single-owner audio session management
│   ├── CoreMLIntegration/   # On-device model inference
│   └── UI/                  # Minimal squat workout interface
├── infrastructure/          # DevOps & Cloud
│   ├── docker/              # Unified development environment
│   ├── cloud_setup/         # Vast.ai/Lambda Labs provisioning
│   └── deployment/          # Model deployment scripts
└── evaluation/              # Validation & Benchmarking
    ├── ablation_studies/    # Placement invariance validation
    ├── benchmarks/          # Performance vs existing solutions
    └── user_studies/        # Post-PoC validation protocols
```

## 90-Day PoC Milestones

### Phase 1: Foundational ML & Data Supremacy (Weeks 1-4)
- [x] M1.1: Unified monorepo & cloud infrastructure
- [ ] M1.2: Multi-modal data engine with 500K+ samples

### Phase 2: Minimum Viable Coach & Integration (Weeks 5-8)
- [ ] M2.1.A: QLv3 model implementation & training
- [ ] M2.1.B: Stable audio pipeline & on-device classifiers
- [ ] M2.2.B: Cognitive modulator & server-side LLM

### Phase 3: PoC Hardening & Pitch Assets (Weeks 9-12)
- [ ] M3.1: Integration, polish, and demo video recording
- [ ] M3.2: Provisional patent applications

## Success Criteria

### Quantitative KPIs
- **Perception**: Surpass MobilePoser benchmarks (MPJPE <8.0cm, MPJRE <20°, Jitter <0.5×10²°/s³)
- **Latency**: <120ms sensor-to-cue pipeline
- **Placement Invariance**: ≤+1.0cm MPJPE delta between pocket positions
- **Audio Stability**: 10-minute sessions with zero audio session errors

### Demo Requirements
1. Voice-activated squat workout initiation
2. Real-time form correction with sub-120ms latency
3. Placement invariance demonstration (phone movement in pocket)
4. Adaptive coaching tone based on detected fatigue progression

## Technology Stack

### ML/AI
- **Simulation**: MuJoCo (Apache 2.0) for physics-based data generation
- **Training**: PyTorch Lightning + Weights & Biases experiment tracking
- **Architecture**: CNN→VQ-VAE→Transformer with multi-task learning
- **Deployment**: Core ML (iOS) + serverless LLM endpoint

### Infrastructure
- **Cloud**: Vast.ai/Lambda Labs spot instances for training (<$500 budget)
- **Containerization**: Docker with unified development environment
- **Monitoring**: W&B for ML experiments, basic cloud provider monitoring

### Mobile
- **Platform**: iOS 17+ (iPhone 13+ recommended)
- **Audio**: UnifiedAudioEngine with strict state machine (.playAndRecord session)
- **Sensors**: Core Motion (IMU + barometer), Core ML (on-device inference)
- **Threading**: GCD with separate queues for sensors, ML, and UI

## Getting Started

### Prerequisites
- macOS with Xcode 15+
- Docker Desktop
- Recent iPhone (A14 Bionic+) for testing
- Cloud GPU account (Vast.ai or Lambda Labs)

### Quick Start
```bash
# Clone and setup
git clone <repo-url>
cd ChimeraAscendant

# Launch unified development environment
./scripts/setup_dev_environment.sh

# Generate initial synthetic dataset
cd quantumleap-v3/data_generation
python generate_squat_dataset.py --samples 50000 --output ./data/

# Start training
cd ../training
python train_qlv3.py --config configs/squat_baseline.yaml
```

## License & IP Strategy

- **Code**: Proprietary (all rights reserved)
- **Patents**: Two provisional applications planned
  1. Multi-modal synthetic data generation with placement invariance
  2. Real-time cognitive state fusion for proactive AI coaching
- **Dependencies**: Apache 2.0 and MIT licensed components only

## Contact

**Project Lead**: [Your Name]  
**Technical Questions**: [Email]  
**Business Inquiries**: [Email]

---

*Building the future of embodied AI, one rep at a time.*
