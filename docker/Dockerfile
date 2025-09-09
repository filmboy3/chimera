# Unified Development Environment for Project Chimera Ascendant
# Includes MuJoCo, PyTorch, and all ML dependencies

FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MuJoCo
WORKDIR /opt
RUN wget https://github.com/deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz \
    && tar -xzf mujoco-2.3.7-linux-x86_64.tar.gz \
    && rm mujoco-2.3.7-linux-x86_64.tar.gz

# Set MuJoCo environment variables
ENV MUJOCO_PATH=/opt/mujoco-2.3.7
ENV LD_LIBRARY_PATH=$MUJOCO_PATH/lib:$LD_LIBRARY_PATH

# Install Python ML dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Create working directory
WORKDIR /workspace

# Set up entry point
CMD ["/bin/bash"]
