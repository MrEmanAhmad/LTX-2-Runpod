# RunPod Serverless Dockerfile for LTX-2
# Models are automatically downloaded from HuggingFace on first run

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
    HF_HOME="/models/.cache/huggingface"

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

# Install the packages
RUN uv pip install --system torch==2.7.0 torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN uv pip install --system -e ./packages/ltx-core -e ./packages/ltx-pipelines
RUN uv pip install --system runpod av pillow boto3 huggingface-hub hf-transfer

# Enable fast HuggingFace downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create directories for models
RUN mkdir -p /models

# Set the model path environment variables
ENV MODEL_PATH="/models/ltx-2-19b-distilled-fp8.safetensors" \
    SPATIAL_UPSAMPLER_PATH="/models/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
    GEMMA_PATH="/models/gemma-3-12b-it-qat-q4_0-unquantized" \
    ENABLE_FP8="true"

# Expose port (for local testing)
EXPOSE 8000

# Run the handler
CMD ["python", "-u", "handler.py"]
