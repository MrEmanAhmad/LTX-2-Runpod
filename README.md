# LTX-2

[![Website](https://img.shields.io/badge/Website-LTX-181717?logo=google-chrome)](https://ltx.io)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=vercel)](https://app.ltx.studio/ltx-2-playground/i2v)
[![Paper](https://img.shields.io/badge/Paper-PDF-EC1C24?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/abs/2601.03233)
[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord)](https://discord.gg/ltxplatform)
[![RunPod](https://api.runpod.io/badge/MrEmanAhmad/LTX-2-Runpod)](https://console.runpod.io/hub/MrEmanAhmad/LTX-2-Runpod)

**LTX-2** is the first DiT-based audio-video foundation model that contains all core capabilities of modern video generation in one model: synchronized audio and video, high fidelity, multiple performance modes, production-ready outputs, API access, and open access.

<div align="center">
  <video src="https://github.com/user-attachments/assets/4414adc0-086c-43de-b367-9362eeb20228" width="70%" poster=""> </video>
</div>

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2

# Set up the environment
uv sync --frozen
source .venv/bin/activate
```

### Required Models

Download the following models from the [LTX-2 HuggingFace repository](https://huggingface.co/Lightricks/LTX-2):

**LTX-2 Model Checkpoint** (choose and download one of the following)
  * [`ltx-2-19b-dev-fp8.safetensors`](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-19b-dev-fp8.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors)

  * [`ltx-2-19b-dev.safetensors`](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-19b-dev.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors)
  * [`ltx-2-19b-distilled.safetensors`](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-19b-distilled.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled.safetensors)
  * [`ltx-2-19b-distilled-fp8.safetensors`](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-19b-distilled-fp8.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8.safetensors)

**Spatial Upscaler** - Required for current two-stage pipeline implementations in this repository
  * [`ltx-2-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-spatial-upscaler-x2-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors)

**Temporal Upscaler** - Supported by the model and will be required for future pipeline implementations
  * [`ltx-2-temporal-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-temporal-upscaler-x2-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors)

**Distilled LoRA** - Required for current two-stage pipeline implementations in this repository (except DistilledPipeline and ICLoraPipeline)
  * [`ltx-2-19b-distilled-lora-384.safetensors`](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-19b-distilled-lora-384.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors)

**Gemma Text Encoder** (download all assets from the repository)
  * [`Gemma 3`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main)

**LoRAs**
  * [`LTX-2-19b-IC-LoRA-Canny-Control`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Canny-Control) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors)
  * [`LTX-2-19b-IC-LoRA-Depth-Control`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Depth-Control) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors)
  * [`LTX-2-19b-IC-LoRA-Detailer`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors)
  * [`LTX-2-19b-IC-LoRA-Pose-Control`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-In`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Left`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Out`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Right`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Jib-Down`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Jib-Up`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Static`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors)

### Available Pipelines

* **[TI2VidTwoStagesPipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py)** - Production-quality text/image-to-video with 2x upsampling (recommended)
* **[TI2VidOneStagePipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py)** - Single-stage generation for quick prototyping
* **[DistilledPipeline](packages/ltx-pipelines/src/ltx_pipelines/distilled.py)** - Fastest inference with 8 predefined sigmas
* **[ICLoraPipeline](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py)** - Video-to-video and image-to-video transformations
* **[KeyframeInterpolationPipeline](packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py)** - Interpolate between keyframe images

### ⚡ Optimization Tips

* **Use DistilledPipeline** - Fastest inference with only 8 predefined sigmas (8 steps stage 1, 4 steps stage 2)
* **Enable FP8 transformer** - Enables lower memory footprint: `--enable-fp8` (CLI) or `fp8transformer=True` (Python)
* **Install attention optimizations** - Use xFormers (`uv sync --extra xformers`) or [Flash Attention 3](https://github.com/Dao-AILab/flash-attention) for Hopper GPUs
* **Use gradient estimation** - Reduce inference steps from 40 to 20-30 while maintaining quality (see [pipeline documentation](packages/ltx-pipelines/README.md#denoising-loop-optimization))
* **Skip memory cleanup** - If you have sufficient VRAM, disable automatic memory cleanup between stages for faster processing
* **Choose single-stage pipeline** - Use `TI2VidOneStagePipeline` for faster generation when high resolution isn't required

## ✍️ Prompting for LTX-2

When writing prompts, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. Think like a cinematographer describing a shot list. Keep within 200 words. For best results, build your prompts using this structure:

- Start with main action in a single sentence
- Add specific details about movements and gestures
- Describe character/object appearances precisely
- Include background and environment details
- Specify camera angles and movements
- Describe lighting and colors
- Note any changes or sudden events

For additional guidance on writing a prompt please refer to <https://ltx.video/blog/how-to-prompt-for-ltx-2>

### Automatic Prompt Enhancement

LTX-2 pipelines support automatic prompt enhancement via an `enhance_prompt` parameter.

## 🔌 ComfyUI Integration

To use our model with ComfyUI, please follow the instructions at <https://github.com/Lightricks/ComfyUI-LTXVideo/>.

## 📦 Packages

This repository is organized as a monorepo with three main packages:

* **[ltx-core](packages/ltx-core/)** - Core model implementation, inference stack, and utilities
* **[ltx-pipelines](packages/ltx-pipelines/)** - High-level pipeline implementations for text-to-video, image-to-video, and other generation modes
* **[ltx-trainer](packages/ltx-trainer/)** - Training and fine-tuning tools for LoRA, full fine-tuning, and IC-LoRA

Each package has its own README and documentation. See the [Documentation](#-documentation) section below.

## 📚 Documentation

Each package includes comprehensive documentation:

* **[LTX-Core README](packages/ltx-core/README.md)** - Core model implementation, inference stack, and utilities
* **[LTX-Pipelines README](packages/ltx-pipelines/README.md)** - High-level pipeline implementations and usage guides
* **[LTX-Trainer README](packages/ltx-trainer/README.md)** - Training and fine-tuning documentation with detailed guides

## ☁️ RunPod Serverless Deployment

Deploy LTX-2 as a serverless API on RunPod for scalable video generation.

### Prerequisites

1. A [RunPod account](https://runpod.io)
2. Docker installed locally (for building the image)
3. Model files downloaded (see [Required Models](#required-models))

### Quick Deploy

#### Step 1: Prepare Model Storage

Upload your models to a network volume or cloud storage (S3, HuggingFace, etc.). Required files:

```
/models/
├── ltx-2-19b-distilled-fp8.safetensors    # Main model (FP8 recommended)
├── ltx-2-spatial-upscaler-x2-1.0.safetensors
└── gemma-3-12b-it-qat-q4_0-unquantized/   # Gemma text encoder directory
```

#### Step 2: Build and Push Docker Image

```bash
# Build the image
docker build -t your-dockerhub/ltx2-serverless:latest .

# Push to Docker Hub (or your preferred registry)
docker push your-dockerhub/ltx2-serverless:latest
```

#### Step 3: Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:
   - **Container Image**: `your-dockerhub/ltx2-serverless:latest`
   - **GPU**: RTX 4090 (24GB) or A100 (40GB+) recommended
   - **Container Disk**: 20GB+
   - **Volume**: Mount your models volume to `/models`
4. Set Environment Variables:
   ```
   MODEL_PATH=/models/ltx-2-19b-distilled-fp8.safetensors
   SPATIAL_UPSAMPLER_PATH=/models/ltx-2-spatial-upscaler-x2-1.0.safetensors
   GEMMA_PATH=/models/gemma-3-12b-it-qat-q4_0-unquantized
   ENABLE_FP8=true
   ```

### API Reference

#### Request Schema

```json
{
  "input": {
    "prompt": "A majestic eagle soaring through clouds...",
    "seed": 42,
    "height": 544,
    "width": 960,
    "num_frames": 97,
    "frame_rate": 25.0,
    "enhance_prompt": false,
    "images": [
      {
        "image": "<base64-encoded-image>",
        "frame_index": 0,
        "strength": 1.0
      }
    ]
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description of the video to generate |
| `seed` | int | 42 | Random seed for reproducibility |
| `height` | int | 544 | Video height (must be divisible by 64) |
| `width` | int | 960 | Video width (must be divisible by 64) |
| `num_frames` | int | 97 | Number of frames (must be k×8+1: 9, 17, 25, ..., 97) |
| `frame_rate` | float | 25.0 | Video frame rate |
| `enhance_prompt` | bool | false | Use AI to enhance the prompt |
| `images` | array | [] | Conditioning images for image-to-video |

#### Response Schema

```json
{
  "video": "<base64-encoded-mp4>",
  "seed": 42,
  "prompt": "A majestic eagle...",
  "duration": 3.88
}
```

### Example: Python Client

```python
import requests
import base64
import time

RUNPOD_API_KEY = "your-api-key"
ENDPOINT_ID = "your-endpoint-id"

def generate_video(prompt, **kwargs):
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    payload = {
        "input": {
            "prompt": prompt,
            "seed": kwargs.get("seed", 42),
            "height": kwargs.get("height", 544),
            "width": kwargs.get("width", 960),
            "num_frames": kwargs.get("num_frames", 97),
            "frame_rate": kwargs.get("frame_rate", 25.0),
        }
    }
    
    # Submit job
    response = requests.post(url, json=payload, headers=headers)
    job_id = response.json()["id"]
    
    # Poll for completion
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    while True:
        status = requests.get(status_url, headers=headers).json()
        if status["status"] == "COMPLETED":
            # Decode and save video
            video_b64 = status["output"]["video"]
            with open("output.mp4", "wb") as f:
                f.write(base64.b64decode(video_b64))
            return status["output"]
        elif status["status"] == "FAILED":
            raise Exception(status.get("error", "Job failed"))
        time.sleep(2)

# Generate a video
result = generate_video(
    "A serene Japanese garden with cherry blossoms falling gently, "
    "koi fish swimming in a crystal clear pond, soft morning light "
    "filtering through the trees."
)
print(f"Generated {result['duration']:.1f}s video")
```

### Example: Image-to-Video

```python
import base64

# Load and encode your image
with open("input_image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

result = generate_video(
    prompt="The scene comes alive with gentle movement...",
    images=[{
        "image": image_b64,
        "frame_index": 0,
        "strength": 1.0
    }]
)
```

### Local Testing

Test the handler locally before deploying:

```bash
# Set environment variables
export MODEL_PATH=/path/to/ltx-2-19b-distilled-fp8.safetensors
export SPATIAL_UPSAMPLER_PATH=/path/to/ltx-2-spatial-upscaler-x2-1.0.safetensors
export GEMMA_PATH=/path/to/gemma-3-12b-it-qat-q4_0-unquantized

# Run the handler
python handler.py
```

### GPU Recommendations

| GPU | VRAM | Performance | Notes |
|-----|------|-------------|-------|
| RTX 4090 | 24GB | Good | Use FP8, smaller resolutions |
| A100 40GB | 40GB | Excellent | Full resolution support |
| A100 80GB | 80GB | Best | Batch processing capable |
| H100 | 80GB | Best | Fastest inference |

### Cost Optimization Tips

1. **Use FP8 models** - Reduces VRAM usage, enabling smaller GPU tiers
2. **Use DistilledPipeline** - Faster inference = lower cost per video
3. **Set appropriate idle timeout** - Balance between cold start time and cost
4. **Use Flash Workers** - For predictable workloads with consistent traffic