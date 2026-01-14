"""
RunPod Serverless Handler for LTX-2 Video Generation
Supports text-to-video and image-to-video generation with synchronized audio.
Models are automatically downloaded from HuggingFace on first run.

🎵 AUDIO IS AUTOMATICALLY GENERATED - LTX-2 is the first DiT model
that generates synchronized audio AND video together!
"""

import base64
import io
import os
import tempfile
import traceback
import urllib.request
from pathlib import Path

import runpod
import torch

# Model paths - check multiple locations
VOLUME_PATH = Path("/runpod-volume/models")
LOCAL_PATH = Path("/models")

def get_model_paths():
    """Get model paths, preferring volume storage."""
    # Check if volume exists and has models
    if VOLUME_PATH.exists():
        base = VOLUME_PATH
    else:
        base = LOCAL_PATH
    
    base.mkdir(parents=True, exist_ok=True)
    
    return {
        "model": base / "ltx-2-19b-distilled-fp8.safetensors",
        "upsampler": base / "ltx-2-spatial-upscaler-x2-1.0.safetensors", 
        "gemma": base / "gemma-3-12b-it-qat-q4_0-unquantized",
        "base": base
    }

# HuggingFace model info
HF_LTX_REPO = "Lightricks/LTX-2"
HF_GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

# Lazy load the pipeline
_pipeline = None


def download_models():
    """Download models from HuggingFace if not already present."""
    from huggingface_hub import hf_hub_download, snapshot_download
    
    paths = get_model_paths()
    base = paths["base"]
    
    print(f"📂 Using model directory: {base}")

    # Download LTX-2 main model
    if not paths["model"].exists():
        print(f"⬇️  Downloading LTX-2 model (~10GB)...")
        hf_hub_download(
            repo_id=HF_LTX_REPO,
            filename="ltx-2-19b-distilled-fp8.safetensors",
            local_dir=str(base),
            local_dir_use_symlinks=False,
        )
        print("✅ LTX-2 model downloaded")
    else:
        print(f"✅ LTX-2 model found at {paths['model']}")

    # Download Spatial Upsampler
    if not paths["upsampler"].exists():
        print(f"⬇️  Downloading Spatial Upsampler (~500MB)...")
        hf_hub_download(
            repo_id=HF_LTX_REPO,
            filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            local_dir=str(base),
            local_dir_use_symlinks=False,
        )
        print("✅ Spatial Upsampler downloaded")
    else:
        print(f"✅ Spatial Upsampler found at {paths['upsampler']}")

    # Download Gemma text encoder
    if not paths["gemma"].exists():
        print(f"⬇️  Downloading Gemma text encoder (~12GB)...")
        snapshot_download(
            repo_id=HF_GEMMA_REPO,
            local_dir=str(paths["gemma"]),
            local_dir_use_symlinks=False,
        )
        print("✅ Gemma text encoder downloaded")
    else:
        print(f"✅ Gemma text encoder found at {paths['gemma']}")

    print("🎉 All models ready!")
    return paths


def get_pipeline():
    """Lazy load the pipeline."""
    global _pipeline
    if _pipeline is None:
        paths = download_models()
        
        from ltx_pipelines import DistilledPipeline
        
        enable_fp8 = os.environ.get("ENABLE_FP8", "true").lower() == "true"

        print("🚀 Loading LTX-2 pipeline...")
        print(f"   Model: {paths['model']}")
        print(f"   Upsampler: {paths['upsampler']}")
        print(f"   Gemma: {paths['gemma']}")
        print(f"   FP8: {enable_fp8}")

        _pipeline = DistilledPipeline(
            checkpoint_path=str(paths["model"]),
            spatial_upsampler_path=str(paths["upsampler"]),
            gemma_root=str(paths["gemma"]),
            loras=[],
            fp8transformer=enable_fp8,
        )
        print("✅ Pipeline loaded!")

    return _pipeline


def download_image_from_url(url: str, output_path: str) -> None:
    """Download an image from URL."""
    print(f"⬇️  Downloading image from {url[:50]}...")
    urllib.request.urlretrieve(url, output_path)


def save_image_from_base64(base64_string: str, output_path: str) -> None:
    """Save a base64 encoded image to file."""
    from PIL import Image
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path)


def encode_to_base64(file_path: str) -> str:
    """Read file and encode to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@torch.inference_mode()
def handler(job: dict) -> dict:
    """
    LTX-2 Video + Audio Generation Handler
    
    🎵 GENERATES SYNCHRONIZED AUDIO AUTOMATICALLY!
    
    ════════════════════════════════════════════════════════
    INPUT (Compatible with n8n workflows)
    ════════════════════════════════════════════════════════
    {
      "input": {
        "prompt": "A cat playing piano",     // Required
        "image_url": "https://...",          // Optional: for image-to-video
        "seed": 42,                          // Optional (default: 42)
        "width": 960,                        // Optional (default: 960, must be ÷64)
        "height": 544,                       // Optional (default: 544, must be ÷64)  
        "num_frames": 97,                    // Optional (default: 97, must be k*8+1)
        "frame_rate": 25.0,                  // Optional (default: 25.0)
        "enhance_prompt": false,             // Optional: AI prompt enhancement
        "generate_audio": true,              // Optional: generate audio (default: true)
        "return_audio_separate": false       // Optional: return audio as separate base64
      }
    }
    
    ════════════════════════════════════════════════════════
    OUTPUT
    ════════════════════════════════════════════════════════
    {
      "video": "<base64 MP4 with embedded audio>",
      "audio": "<base64 WAV>",               // Only if return_audio_separate=true
      "seed": 42,
      "prompt": "...",
      "duration": 3.88,
      "has_audio": true
    }
    """
    try:
        job_input = job.get("input", {})

        # Required
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "Missing required parameter: prompt"}

        # Optional parameters
        seed = int(job_input.get("seed", 42))
        height = int(job_input.get("height", 544))
        width = int(job_input.get("width", 960))
        num_frames = int(job_input.get("num_frames", 97))
        frame_rate = float(job_input.get("frame_rate", 25.0))
        enhance_prompt = bool(job_input.get("enhance_prompt", False))
        generate_audio = bool(job_input.get("generate_audio", True))
        return_audio_separate = bool(job_input.get("return_audio_separate", False))

        # Validation
        if height % 64 != 0 or width % 64 != 0:
            return {"error": f"Resolution ({width}x{height}) must be divisible by 64"}
        
        if (num_frames - 1) % 8 != 0:
            return {"error": f"num_frames must be k*8+1 (9,17,25,...97). Got: {num_frames}"}

        # Process images
        images = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Handle image_url (single URL or list)
            image_url = job_input.get("image_url")
            if image_url:
                urls = image_url if isinstance(image_url, list) else [image_url]
                for idx, url in enumerate(urls):
                    if url and isinstance(url, str) and url.startswith("http"):
                        img_path = temp_path / f"url_img_{idx}.png"
                        download_image_from_url(url, str(img_path))
                        images.append((str(img_path), 0, 1.0))

            # Handle images array
            for idx, img_data in enumerate(job_input.get("images", [])):
                if isinstance(img_data, dict):
                    content = img_data.get("image", "")
                    frame_idx = int(img_data.get("frame_index", 0))
                    strength = float(img_data.get("strength", 1.0))
                else:
                    content, frame_idx, strength = img_data, 0, 1.0

                if content:
                    img_path = temp_path / f"img_{idx}.png"
                    if content.startswith("http"):
                        download_image_from_url(content, str(img_path))
                    else:
                        save_image_from_base64(content, str(img_path))
                    images.append((str(img_path), frame_idx, strength))

            # Load pipeline
            pipeline = get_pipeline()

            # Tiling config
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
            tiling_config = TilingConfig.default()
            video_chunks = get_video_chunks_number(num_frames, tiling_config)

            duration = num_frames / frame_rate
            
            print("=" * 50)
            print("🎬 LTX-2 Video + Audio Generation")
            print("=" * 50)
            print(f"📐 Resolution: {width}x{height}")
            print(f"🎞️  Frames: {num_frames} @ {frame_rate}fps = {duration:.1f}s")
            print(f"💬 Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            print(f"🎲 Seed: {seed}")
            print(f"🎵 Audio: {'Yes' if generate_audio else 'No'}")
            if images:
                print(f"🖼️  Input images: {len(images)}")
            print("=" * 50)

            # Generate video + audio
            video, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=enhance_prompt,
            )

            # Encode output
            from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
            from ltx_pipelines.utils.media_io import encode_video

            output_path = temp_path / "output.mp4"
            encode_video(
                video=video,
                fps=int(frame_rate),
                audio=audio if generate_audio else None,
                audio_sample_rate=AUDIO_SAMPLE_RATE if generate_audio else None,
                output_path=str(output_path),
                video_chunks_number=video_chunks,
            )

            response = {
                "video": encode_to_base64(str(output_path)),
                "seed": seed,
                "prompt": prompt,
                "duration": duration,
                "has_audio": generate_audio and audio is not None,
            }

            # Separate audio if requested
            if return_audio_separate and audio is not None:
                import torchaudio
                audio_path = temp_path / "output.wav"
                torchaudio.save(str(audio_path), audio.cpu(), AUDIO_SAMPLE_RATE)
                response["audio"] = encode_to_base64(str(audio_path))

            print("=" * 50)
            print(f"✅ Complete! {duration:.1f}s video" + (" + audio" if response["has_audio"] else ""))
            print("=" * 50)

            return response

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start handler
runpod.serverless.start({"handler": handler})
