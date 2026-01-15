"""
RunPod Serverless Handler for LTX-2 Video Generation
Generates video WITH synchronized audio automatically!

LTX-2 is the first DiT model that generates audio AND video together.
"""

import base64
import io
import os
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

print("=" * 60)
print("🚀 LTX-2 Video + Audio Generator Starting...")
print("=" * 60)
print(f"📦 Python: {sys.version.split()[0]}")

try:
    import runpod
    print(f"✅ RunPod SDK loaded")
except ImportError as e:
    print(f"❌ Failed to import runpod: {e}")
    raise

try:
    import torch
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  WARNING: CUDA not available!")
except ImportError as e:
    print(f"❌ Failed to import torch: {e}")
    raise

# Test critical imports early
try:
    from ltx_pipelines import DistilledPipeline
    from ltx_core.model.video_vae import TilingConfig
    print("✅ LTX packages loaded")
except ImportError as e:
    print(f"❌ Failed to import LTX packages: {e}")
    traceback.print_exc()
    raise

print("=" * 60)

# Model paths - priority order for RunPod Serverless
def get_model_paths():
    """Get model paths, preferring network volume for persistence."""
    
    # Check if network volume is actually mounted (not just directory exists)
    runpod_vol = Path("/runpod-volume")
    workspace_vol = Path("/workspace")
    
    # Priority 1: /runpod-volume if it's a mount point (serverless network volume)
    if runpod_vol.exists() and runpod_vol.is_mount():
        base = runpod_vol / "models"
        print(f"📂 Using network volume: {base}")
    # Priority 2: /workspace if it's a mount point (GPU pod volume)
    elif workspace_vol.exists() and workspace_vol.is_mount():
        base = workspace_vol / "models"
        print(f"📂 Using workspace volume: {base}")
    # Priority 3: Container disk fallback
    else:
        base = Path("/models")
        print(f"📂 Using container disk: {base}")
    
    base.mkdir(parents=True, exist_ok=True)
    
    # Set HuggingFace cache to same volume for persistence
    cache_dir = base.parent / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    print(f"📂 HF cache: {cache_dir}")
    
    return {
        "model": base / "ltx-2-19b-distilled-fp8.safetensors",
        "upsampler": base / "ltx-2-spatial-upscaler-x2-1.0.safetensors", 
        "gemma": base / "gemma-3-12b-it-qat-q4_0-unquantized",
        "base": base
    }


# HuggingFace repos
HF_LTX_REPO = "Lightricks/LTX-2"
HF_GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

# Pipeline singleton
_pipeline = None


def download_models():
    """Download models from HuggingFace if not present."""
    from huggingface_hub import hf_hub_download, snapshot_download
    
    paths = get_model_paths()
    base = paths["base"]
    
    # Get HuggingFace token for gated models (Gemma requires this)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("🔑 HuggingFace token found")
    else:
        print("⚠️  No HF_TOKEN found - gated models may fail to download")

    # Download LTX-2 model
    if not paths["model"].exists():
        print("⬇️  Downloading LTX-2 model (~10GB)...")
        hf_hub_download(
            repo_id=HF_LTX_REPO,
            filename="ltx-2-19b-distilled-fp8.safetensors",
            local_dir=str(base),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        print("✅ LTX-2 model ready")
    else:
        print(f"✅ LTX-2 model exists: {paths['model']}")

    # Download Spatial Upsampler
    if not paths["upsampler"].exists():
        print("⬇️  Downloading Spatial Upsampler (~500MB)...")
        hf_hub_download(
            repo_id=HF_LTX_REPO,
            filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            local_dir=str(base),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        print("✅ Spatial Upsampler ready")
    else:
        print(f"✅ Spatial Upsampler exists: {paths['upsampler']}")

    # Download Gemma (GATED MODEL - requires token!)
    if not paths["gemma"].exists() or not any(paths["gemma"].iterdir() if paths["gemma"].exists() else []):
        print("⬇️  Downloading Gemma text encoder (~12GB)...")
        if not hf_token:
            print("❌ ERROR: Gemma is a gated model - HF_TOKEN required!")
            print("   Set HF_TOKEN environment variable with your HuggingFace token")
            raise ValueError("HF_TOKEN required for Gemma model download")
        snapshot_download(
            repo_id=HF_GEMMA_REPO,
            local_dir=str(paths["gemma"]),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        print("✅ Gemma text encoder ready")
    else:
        print(f"✅ Gemma exists: {paths['gemma']}")

    print("🎉 All models ready!")
    return paths


def get_pipeline():
    """Get or create the pipeline singleton."""
    global _pipeline
    
    if _pipeline is None:
        print("=" * 50)
        print("🚀 Initializing LTX-2 Pipeline")
        print("=" * 50)
        
        paths = download_models()
        
        enable_fp8 = os.environ.get("ENABLE_FP8", "true").lower() == "true"

        print(f"📦 Loading pipeline (FP8={enable_fp8})...")
        
        _pipeline = DistilledPipeline(
            checkpoint_path=str(paths["model"]),
            spatial_upsampler_path=str(paths["upsampler"]),
            gemma_root=str(paths["gemma"]),
            loras=[],
            fp8transformer=enable_fp8,
        )
        
        print("✅ Pipeline ready!")
        print("=" * 50)

    return _pipeline


def download_image(url: str, output_path: str) -> None:
    """Download image from URL."""
    print(f"⬇️  Downloading: {url[:60]}...")
    urllib.request.urlretrieve(url, output_path)


def save_base64_image(b64_string: str, output_path: str) -> None:
    """Decode and save base64 image."""
    from PIL import Image
    
    # Remove data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    
    data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(data))
    img.save(output_path)


def file_to_base64(path: str) -> str:
    """Read file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@torch.inference_mode()
def handler(job: dict) -> dict:
    """
    LTX-2 Video + Audio Generation
    
    INPUT:
    {
      "input": {
        "prompt": "description of video",  // REQUIRED
        "image_url": "https://...",        // Optional: image-to-video
        "seed": 42,                        // Optional
        "width": 960,                      // Optional (divisible by 64)
        "height": 544,                     // Optional (divisible by 64)
        "num_frames": 97,                  // Optional (must be k*8+1)
        "frame_rate": 25.0,                // Optional
        "generate_audio": true,            // Optional
        "enhance_prompt": false            // Optional
      }
    }
    
    OUTPUT:
    {
      "video": "<base64 MP4 with audio>",
      "seed": 42,
      "duration": 3.88,
      "has_audio": true
    }
    """
    try:
        inp = job.get("input", {})

        # Required parameter
        prompt = inp.get("prompt")
        if not prompt:
            return {"error": "Missing required 'prompt' parameter"}

        # Optional parameters with defaults
        seed = int(inp.get("seed", 42))
        width = int(inp.get("width", 960))
        height = int(inp.get("height", 544))
        num_frames = int(inp.get("num_frames", 97))
        frame_rate = float(inp.get("frame_rate", 25.0))
        generate_audio = str(inp.get("generate_audio", True)).lower() in ("true", "1", "yes")
        enhance_prompt = str(inp.get("enhance_prompt", False)).lower() in ("true", "1", "yes")

        # Validate resolution (must be divisible by 64)
        if width % 64 != 0 or height % 64 != 0:
            return {"error": f"Resolution {width}x{height} must be divisible by 64. Try 960x544 or 512x512."}
        
        # Validate frames (must be k*8+1)
        if (num_frames - 1) % 8 != 0:
            valid = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97]
            return {"error": f"num_frames must be k*8+1. Valid values: {valid}. Got: {num_frames}"}

        # Process input images
        images = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Handle image_url (string or list)
            img_url = inp.get("image_url")
            if img_url:
                urls = [img_url] if isinstance(img_url, str) else img_url
                for i, url in enumerate(urls):
                    if url and url.startswith("http"):
                        path = tmp / f"img_{i}.png"
                        download_image(url, str(path))
                        images.append((str(path), 0, 1.0))

            # Handle images array
            for i, img in enumerate(inp.get("images", [])):
                if isinstance(img, dict):
                    content = img.get("image", "")
                    frame_idx = int(img.get("frame_index", 0))
                    strength = float(img.get("strength", 1.0))
                else:
                    content = img
                    frame_idx = 0
                    strength = 1.0

                if content:
                    path = tmp / f"arr_img_{i}.png"
                    if content.startswith("http"):
                        download_image(content, str(path))
                    else:
                        save_base64_image(content, str(path))
                    images.append((str(path), frame_idx, strength))

            # Get pipeline
            pipeline = get_pipeline()

            # Setup tiling
            from ltx_core.model.video_vae import get_video_chunks_number
            tiling = TilingConfig.default()
            chunks = get_video_chunks_number(num_frames, tiling)

            duration = num_frames / frame_rate

            print("=" * 50)
            print("🎬 Generating Video + Audio")
            print("=" * 50)
            print(f"📐 {width}x{height} | {num_frames} frames | {duration:.1f}s")
            print(f"💬 {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
            print(f"🎲 Seed: {seed} | 🎵 Audio: {generate_audio}")
            if images:
                print(f"🖼️  {len(images)} input image(s)")
            print("=" * 50)

            # Generate!
            video, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=tiling,
                enhance_prompt=enhance_prompt,
            )

            # Encode output
            from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
            from ltx_pipelines.utils.media_io import encode_video

            out_path = tmp / "output.mp4"
            encode_video(
                video=video,
                fps=int(frame_rate),
                audio=audio if generate_audio else None,
                audio_sample_rate=AUDIO_SAMPLE_RATE if generate_audio else None,
                output_path=str(out_path),
                video_chunks_number=chunks,
            )

            result = {
                "video": file_to_base64(str(out_path)),
                "seed": seed,
                "prompt": prompt,
                "duration": duration,
                "has_audio": generate_audio and audio is not None,
            }

            print("=" * 50)
            print(f"✅ Done! {duration:.1f}s video" + (" + audio" if result["has_audio"] else ""))
            print("=" * 50)

            return result

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start RunPod handler
runpod.serverless.start({"handler": handler})
