#!/bin/bash

# Project Chimera Ascendant - Development Environment Setup
# Single command to bootstrap the entire development environment

set -e

echo "🚀 Setting up Project Chimera Ascendant development environment..."

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed. Please install Docker Desktop first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed."
    exit 1
fi

# Create directory structure
echo "📁 Creating project directory structure..."
mkdir -p quantumleap-v3/{data_generation,models,training,evaluation}
mkdir -p sesame-v2/{audio_pipeline,intent_recognition,cognitive_modulator,llm_endpoint}
mkdir -p ios_app/{UnifiedAudioEngine,CoreMLIntegration,UI}
mkdir -p infrastructure/{cloud_setup,deployment}
mkdir -p evaluation/{ablation_studies,benchmarks,user_studies}

# Create initial config files
echo "⚙️ Creating configuration files..."

# QuantumLeap v3 training config
cat > quantumleap-v3/training/configs/squat_baseline.yaml << EOF
# QuantumLeap v3 Baseline Configuration
model:
  name: "QLv3_Squat_Baseline"
  architecture:
    cnn_channels: [64, 128, 256]
    vq_codebook_size: 512
    vq_commitment_cost: 0.25
    transformer_layers: 6
    transformer_heads: 8
    transformer_dim: 512
  
data:
  batch_size: 32
  sequence_length: 100  # 1 second at 100Hz
  num_workers: 4
  
training:
  max_epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-5
  gradient_clip_val: 1.0
  
loss_weights:
  pose_regression: 1.0
  classification: 0.5
  vq_commitment: 0.25
  
logging:
  project: "chimera-ascendant"
  entity: "quantumleap-v3"
  log_every_n_steps: 50
EOF

# Data generation config
cat > quantumleap-v3/data_generation/config.yaml << EOF
# Synthetic Data Generation Configuration
dataset:
  name: "squat_dataset_v1"
  total_samples: 500000
  exercise_type: "squat"
  
simulation:
  timestep: 0.01  # 100Hz
  duration: 30.0  # 30 seconds per sample
  
augmentation:
  placement_randomization:
    enabled: true
    position_variance: 0.05  # 5cm variance
    orientation_variance: 15  # 15 degree variance
  
  barometer_simulation:
    enabled: true
    noise_std: 0.1  # Pascal
    drift_rate: 0.01  # Pa/s
  
  fatigue_modeling:
    enabled: true
    jitter_increase_rate: 0.02  # per rep
    max_jitter_multiplier: 3.0
  
form_errors:
  knee_valgus:
    probability: 0.15
    severity_range: [0.1, 0.4]
  
  forward_lean:
    probability: 0.12
    severity_range: [0.1, 0.3]
  
  insufficient_depth:
    probability: 0.18
    severity_range: [0.1, 0.5]
EOF

# Build and start development environment
echo "🐳 Building Docker development environment..."
cd docker
docker-compose build

echo "🎯 Development environment ready!"
echo ""
echo "Next steps:"
echo "1. Start the development container:"
echo "   cd docker && docker-compose up -d"
echo ""
echo "2. Enter the container:"
echo "   docker-compose exec chimera-dev bash"
echo ""
echo "3. Generate initial dataset:"
echo "   cd quantumleap-v3/data_generation"
echo "   python generate_squat_dataset.py --config config.yaml"
echo ""
echo "4. Start training:"
echo "   cd ../training"
echo "   python train_qlv3.py --config configs/squat_baseline.yaml"
echo ""
echo "🔥 Ready to build the future of embodied AI coaching!"
