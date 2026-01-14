"""
RunPod Serverless Handler for LTX-2 Video Generation
Supports text-to-video and image-to-video generation with audio.
"""

import base64
import io
import os
import tempfile
import traceback
from pathlib import Path

import runpod
import torch

# Lazy load the pipeline components
_pipeline = None


def get_pipeline():
    """Lazy load the pipeline to avoid loading on cold start until needed."""
    global _pipeline
    if _pipeline is None:
        from ltx_pipelines import DistilledPipeline

        checkpoint_path = os.environ.get(
            "MODEL_PATH", "/models/ltx-2-19b-distilled-fp8.safetensors"
        )
        spatial_upsampler_path = os.environ.get(
            "SPATIAL_UPSAMPLER_PATH", "/models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
        )
        gemma_path = os.environ.get(
            "GEMMA_PATH", "/models/gemma-3-12b-it-qat-q4_0-unquantized"
        )
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


@torch.inference_mode()
def handler(job: dict) -> dict:
    """
    RunPod handler for LTX-2 video generation.

    Input schema:
    {
        "input": {
            "prompt": str,                    # Required: Text prompt for generation
            "seed": int,                      # Optional: Random seed (default: 42)
            "height": int,                    # Optional: Video height (default: 544, must be divisible by 64)
            "width": int,                     # Optional: Video width (default: 960, must be divisible by 64)
            "num_frames": int,                # Optional: Number of frames (default: 97, must be k*8+1)
            "frame_rate": float,              # Optional: Frame rate (default: 25.0)
            "enhance_prompt": bool,           # Optional: Use prompt enhancement (default: false)
            "images": [                       # Optional: List of conditioning images
                {
                    "image": str,             # Base64 encoded image or URL
                    "frame_index": int,       # Frame index to condition on (default: 0)
                    "strength": float         # Conditioning strength (default: 1.0)
                }
            ]
        }
    }

    Output schema:
    {
        "video": str,           # Base64 encoded MP4 video with audio
        "seed": int,            # Seed used for generation
        "prompt": str,          # Prompt used (may be enhanced)
        "duration": float       # Video duration in seconds
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

        # Process conditioning images
        images = []
        input_images = job_input.get("images", [])

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for idx, img_data in enumerate(input_images):
                if isinstance(img_data, dict):
                    image_content = img_data.get("image", "")
                    frame_index = int(img_data.get("frame_index", 0))
                    strength = float(img_data.get("strength", 1.0))
                else:
                    # Backwards compatibility: just a base64 string
                    image_content = img_data
                    frame_index = 0
                    strength = 1.0

                if image_content:
                    image_path = temp_path / f"input_image_{idx}.png"
                    save_image_from_base64(image_content, str(image_path))
                    images.append((str(image_path), frame_index, strength))

            # Get the pipeline (lazy loaded)
            pipeline = get_pipeline()

            # Configure tiling for memory efficiency
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

            tiling_config = TilingConfig.default()
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

            print(f"Generating video: {width}x{height}, {num_frames} frames @ {frame_rate}fps")
            print(f"Prompt: {prompt}")
            print(f"Seed: {seed}")
            if images:
                print(f"Conditioning images: {len(images)}")

            # Generate video
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

            # Save to temp file
            output_path = temp_path / "output.mp4"

            from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
            from ltx_pipelines.utils.media_io import encode_video

            encode_video(
                video=video,
                fps=int(frame_rate),
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=str(output_path),
                video_chunks_number=video_chunks_number,
            )

            # Encode video to base64
            video_base64 = encode_video_to_base64(str(output_path))

            duration = num_frames / frame_rate

            print(f"Generation complete! Duration: {duration:.2f}s")

            return {
                "video": video_base64,
                "seed": seed,
                "prompt": prompt,
                "duration": duration,
            }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# For local testing
if __name__ == "__main__":
    # Check if running in RunPod environment
    if os.environ.get("RUNPOD_POD_ID"):
        runpod.serverless.start({"handler": handler})
    else:
        # Local testing mode
        print("Running in local test mode...")
        print("To test, call: python handler.py")

        # Simple test
        test_input = {
            "input": {
                "prompt": "A serene mountain landscape with flowing clouds and gentle wind rustling through pine trees.",
                "seed": 42,
                "height": 544,
                "width": 960,
                "num_frames": 25,
                "frame_rate": 25.0,
            }
        }

        result = handler(test_input)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Success! Generated {result['duration']:.2f}s video")
            # Save the video locally
            with open("test_output.mp4", "wb") as f:
                f.write(base64.b64decode(result["video"]))
            print("Saved to test_output.mp4")

