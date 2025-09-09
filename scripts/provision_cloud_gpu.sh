#!/bin/bash

# Project Chimera Ascendant - Cloud GPU Provisioning Script
# Automated setup for budget-conscious training on Vast.ai or Lambda Labs

set -e

echo "🌩️ Provisioning cloud GPU environment for Project Chimera Ascendant..."

# Configuration
INSTANCE_TYPE="RTX_4090"  # Target GPU type
MAX_PRICE_PER_HOUR="0.50"  # Budget constraint
MIN_GPU_RAM="24"  # GB
DOCKER_IMAGE="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"

# Check if vastai CLI is installed
if ! command -v vastai &> /dev/null; then
    echo "📦 Installing Vast.ai CLI..."
    pip install vastai
    echo "Please set your Vast.ai API key:"
    echo "vastai set api-key YOUR_API_KEY_HERE"
    echo "Then run this script again."
    exit 1
fi

# Search for available instances
echo "🔍 Searching for available GPU instances..."
vastai search offers \
    --type=bid \
    --order=score- \
    --gpu_name="RTX_4090" \
    --gpu_ram=">=${MIN_GPU_RAM}" \
    --dph="<${MAX_PRICE_PER_HOUR}" \
    --num_gpus=1 \
    --verified=true

echo ""
echo "💡 To create an instance, run:"
echo "vastai create instance <INSTANCE_ID> --image ${DOCKER_IMAGE} --disk 50"
echo ""
echo "📋 Setup commands to run on the instance:"
echo "1. Clone the repository:"
echo "   git clone <your-repo-url> /workspace/ChimeraAscendant"
echo ""
echo "2. Install dependencies:"
echo "   cd /workspace/ChimeraAscendant"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Install MuJoCo:"
echo "   wget https://github.com/deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz"
echo "   tar -xzf mujoco-2.3.7-linux-x86_64.tar.gz -C /opt/"
echo "   export MUJOCO_PATH=/opt/mujoco-2.3.7"
echo "   export LD_LIBRARY_PATH=\$MUJOCO_PATH/lib:\$LD_LIBRARY_PATH"
echo ""
echo "4. Generate dataset:"
echo "   cd quantumleap-v3/data_generation"
echo "   python generate_squat_dataset.py --config config.yaml --samples 50000"
echo ""
echo "5. Start training:"
echo "   cd ../training"
echo "   python train_qlv3.py --config configs/squat_baseline.yaml"
echo ""
echo "💰 Estimated cost for 50K samples + training: ~$25-50"
echo "💰 Estimated cost for full 500K dataset: ~$200-400"
