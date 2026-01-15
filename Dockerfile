# RunPod Serverless Dockerfile for LTX-2
# Generates video WITH synchronized audio automatically!

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && git lfs install

# Create working directory
WORKDIR /app

# Copy project files
COPY packages/ ./packages/
COPY pyproject.toml ./
COPY handler.py ./

# Install uv for faster package management
RUN pip install uv

# Install PyTorch 2.7 (required by ltx-core)
RUN uv pip install --system torch==2.7.0 torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ltx packages and dependencies
RUN uv pip install --system \
    einops \
    numpy \
    transformers \
    safetensors \
    accelerate \
    "scipy>=1.14" \
    av \
    tqdm \
    pillow \
    sentencepiece

# Install the local packages
RUN uv pip install --system -e ./packages/ltx-core -e ./packages/ltx-pipelines --no-deps

# Install RunPod and HuggingFace tools
RUN uv pip install --system runpod huggingface-hub hf-transfer

# Create directories for models
RUN mkdir -p /models /runpod-volume

# Set environment variables
ENV ENABLE_FP8="true"

# Run the handler
CMD ["python", "-u", "handler.py"]
