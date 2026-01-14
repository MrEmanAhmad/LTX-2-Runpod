"""
RunPod Serverless Handler for LTX-2 Video Generation
Supports text-to-video and image-to-video generation with synchronized audio.
Models are automatically downloaded from HuggingFace on first run.

Audio is automatically generated and embedded in the video output.
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

# Model paths
MODELS_DIR = Path("/models")
MODEL_PATH = MODELS_DIR / "ltx-2-19b-distilled-fp8.safetensors"
SPATIAL_UPSAMPLER_PATH = MODELS_DIR / "ltx-2-spatial-upscaler-x2-1.0.safetensors"
GEMMA_PATH = MODELS_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"

# HuggingFace model info
HF_LTX_REPO = "Lightricks/LTX-2"
HF_GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

# Lazy load the pipeline components
_pipeline = None


def download_models():
    """Download models from HuggingFace if not already present."""
    from huggingface_hub import hf_hub_download, snapshot_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download LTX-2 main model
    if not MODEL_PATH.exists():
        print(f"Downloading LTX-2 model to {MODEL_PATH}...")
        hf_hub_download(
            repo_id=HF_LTX_REPO,
            filename="ltx-2-19b-distilled-fp8.safetensors",
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print("✓ LTX-2 model downloaded")
    else:
        print(f"✓ LTX-2 model already exists at {MODEL_PATH}")

    # Download Spatial Upsampler
    if not SPATIAL_UPSAMPLER_PATH.exists():
        print(f"Downloading Spatial Upsampler to {SPATIAL_UPSAMPLER_PATH}...")
        hf_hub_download(
            repo_id=HF_LTX_REPO,
            filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print("✓ Spatial Upsampler downloaded")
    else:
        print(f"✓ Spatial Upsampler already exists at {SPATIAL_UPSAMPLER_PATH}")

    # Download Gemma text encoder
    if not GEMMA_PATH.exists():
        print(f"Downloading Gemma text encoder to {GEMMA_PATH}...")
        snapshot_download(
            repo_id=HF_GEMMA_REPO,
            local_dir=str(GEMMA_PATH),
            local_dir_use_symlinks=False,
        )
        print("✓ Gemma text encoder downloaded")
    else:
        print(f"✓ Gemma text encoder already exists at {GEMMA_PATH}")

    print("All models ready!")


def get_pipeline():
    """Lazy load the pipeline to avoid loading on cold start until needed."""
    global _pipeline
    if _pipeline is None:
        # Download models if needed
        download_models()

        from ltx_pipelines import DistilledPipeline

        # Allow environment variable overrides
        checkpoint_path = os.environ.get("MODEL_PATH", str(MODEL_PATH))
        spatial_upsampler_path = os.environ.get("SPATIAL_UPSAMPLER_PATH", str(SPATIAL_UPSAMPLER_PATH))
        gemma_path = os.environ.get("GEMMA_PATH", str(GEMMA_PATH))
        enable_fp8 = os.environ.get("ENABLE_FP8", "true").lower() == "true"

        print(f"Loading LTX-2 pipeline...")
        print(f"  Model: {checkpoint_path}")
        print(f"  Spatial Upsampler: {spatial_upsampler_path}")
        print(f"  Gemma: {gemma_path}")
        print(f"  FP8: {enable_fp8}")

        _pipeline = DistilledPipeline(
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_path,
            loras=[],
            fp8transformer=enable_fp8,
        )
        print("Pipeline loaded successfully!")

    return _pipeline


def download_image_from_url(url: str, output_path: str) -> None:
    """Download an image from a URL and save it locally."""
    print(f"Downloading image from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"✓ Image saved to {output_path}")


def save_image_from_base64(base64_string: str, output_path: str) -> None:
    """Save a base64 encoded image to a file."""
    from PIL import Image

    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path)


def encode_video_to_base64(video_path: str) -> str:
    """Read a video file and encode it to base64."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_audio_to_base64(audio_path: str) -> str:
    """Read an audio file and encode it to base64."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@torch.inference_mode()
def handler(job: dict) -> dict:
    """
    RunPod handler for LTX-2 video generation with synchronized audio.

    ═══════════════════════════════════════════════════════════════════
    INPUT SCHEMA (Compatible with n8n workflows)
    ═══════════════════════════════════════════════════════════════════

    {
        "input": {
            // === REQUIRED ===
            "prompt": str,                    // Text prompt describing the video/audio

            // === OPTIONAL - Video Settings ===
            "seed": int,                      // Random seed (default: 42)
            "height": int,                    // Video height (default: 544, must be divisible by 64)
            "width": int,                     // Video width (default: 960, must be divisible by 64)
            "num_frames": int,                // Number of frames (default: 97, must be k*8+1)
            "frame_rate": float,              // Frame rate (default: 25.0)
            "enhance_prompt": bool,           // AI prompt enhancement (default: false)

            // === OPTIONAL - Image Conditioning (Image-to-Video) ===
            // Method 1: Single image URL (for n8n compatibility)
            "image_url": str | [str],         // URL(s) to conditioning image(s)

            // Method 2: Detailed image config
            "images": [
                {
                    "image": str,             // Base64 encoded image OR URL
                    "frame_index": int,       // Frame index to condition on (default: 0)
                    "strength": float         // Conditioning strength (default: 1.0)
                }
            ],

            // === OPTIONAL - Audio Settings ===
            "generate_audio": bool,           // Generate audio (default: true)
            "return_audio_separate": bool     // Return audio as separate base64 (default: false)
        }
    }

    ═══════════════════════════════════════════════════════════════════
    OUTPUT SCHEMA
    ═══════════════════════════════════════════════════════════════════

    {
        "video": str,              // Base64 encoded MP4 video (with embedded audio if generated)
        "audio": str | null,       // Base64 encoded WAV audio (only if return_audio_separate=true)
        "seed": int,               // Seed used for generation
        "prompt": str,             // Prompt used (may be enhanced)
        "duration": float,         // Video duration in seconds
        "has_audio": bool          // Whether audio was generated
    }
    """
    try:
        job_input = job.get("input", {})

        # Extract parameters with defaults
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "Missing required parameter: prompt"}

        seed = int(job_input.get("seed", 42))
        height = int(job_input.get("height", 544))
        width = int(job_input.get("width", 960))
        num_frames = int(job_input.get("num_frames", 97))
        frame_rate = float(job_input.get("frame_rate", 25.0))
        enhance_prompt = bool(job_input.get("enhance_prompt", False))
        generate_audio = bool(job_input.get("generate_audio", True))
        return_audio_separate = bool(job_input.get("return_audio_separate", False))

        # Validate resolution for two-stage pipeline (must be divisible by 64)
        if height % 64 != 0 or width % 64 != 0:
            return {
                "error": f"Resolution ({height}x{width}) must be divisible by 64 for the two-stage pipeline"
            }

        # Validate frame count (must be k*8 + 1)
        if (num_frames - 1) % 8 != 0:
            return {
                "error": f"num_frames must be k*8+1 (e.g., 9, 17, 25, ..., 97). Got: {num_frames}"
            }

        # Process conditioning images from multiple input formats
        images = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Method 1: Handle image_url parameter (n8n compatibility)
            image_url = job_input.get("image_url")
            if image_url:
                # Can be a single URL or list of URLs
                urls = image_url if isinstance(image_url, list) else [image_url]
                for idx, url in enumerate(urls):
                    if url and isinstance(url, str) and url.startswith("http"):
                        image_path = temp_path / f"url_image_{idx}.png"
                        download_image_from_url(url, str(image_path))
                        images.append((str(image_path), 0, 1.0))  # frame_index=0, strength=1.0

            # Method 2: Handle images array parameter
            input_images = job_input.get("images", [])
            for idx, img_data in enumerate(input_images):
                if isinstance(img_data, dict):
                    image_content = img_data.get("image", "")
                    frame_index = int(img_data.get("frame_index", 0))
                    strength = float(img_data.get("strength", 1.0))
                else:
                    # Backwards compatibility: just a string (base64 or URL)
                    image_content = img_data
                    frame_index = 0
                    strength = 1.0

                if image_content:
                    image_path = temp_path / f"input_image_{idx}.png"

                    # Check if it's a URL or base64
                    if isinstance(image_content, str) and image_content.startswith("http"):
                        download_image_from_url(image_content, str(image_path))
                    else:
                        save_image_from_base64(image_content, str(image_path))

                    images.append((str(image_path), frame_index, strength))

            # Get the pipeline (lazy loaded, downloads models if needed)
            pipeline = get_pipeline()

            # Configure tiling for memory efficiency
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

            tiling_config = TilingConfig.default()
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

            print("=" * 60)
            print("LTX-2 Video + Audio Generation")
            print("=" * 60)
            print(f"Resolution: {width}x{height}")
            print(f"Frames: {num_frames} @ {frame_rate}fps")
            print(f"Duration: {num_frames / frame_rate:.2f}s")
            print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
            print(f"Seed: {seed}")
            print(f"Generate Audio: {generate_audio}")
            if images:
                print(f"Conditioning Images: {len(images)}")
            print("=" * 60)

            # Generate video (audio is always generated by the pipeline)
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

            # Prepare output
            from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
            from ltx_pipelines.utils.media_io import encode_video

            # Save video with embedded audio
            output_path = temp_path / "output.mp4"
            encode_video(
                video=video,
                fps=int(frame_rate),
                audio=audio if generate_audio else None,
                audio_sample_rate=AUDIO_SAMPLE_RATE if generate_audio else None,
                output_path=str(output_path),
                video_chunks_number=video_chunks_number,
            )

            # Encode video to base64
            video_base64 = encode_video_to_base64(str(output_path))

            duration = num_frames / frame_rate

            # Prepare response
            response = {
                "video": video_base64,
                "seed": seed,
                "prompt": prompt,
                "duration": duration,
                "has_audio": generate_audio and audio is not None,
            }

            # Optionally return audio separately
            if return_audio_separate and audio is not None:
                import torchaudio
                audio_path = temp_path / "output.wav"
                torchaudio.save(str(audio_path), audio.cpu(), AUDIO_SAMPLE_RATE)
                response["audio"] = encode_audio_to_base64(str(audio_path))

            print("=" * 60)
            print(f"✓ Generation complete!")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Audio: {'Yes' if response['has_audio'] else 'No'}")
            print("=" * 60)

            return response

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
