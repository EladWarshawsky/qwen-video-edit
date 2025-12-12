import os
import torch
from PIL import Image
from qwen_vid2vid.pipeline import QwenVid2VidPipeline
from diffusers.utils import load_image, export_to_video

from transformers import BitsAndBytesConfig

# Configuration
MODEL_ID = "Qwen/Qwen-Image-Edit"
PROMPT = "A van gogh style painting of a car driving on the road"
INPUT_VIDEO_PATH = "path/to/your/input_video" # Or a folder of images
OUTPUT_DIR = "./output_vid2vid"
FRAMES_TO_PROCESS = 8  # Keep low for testing VRAM usage

def load_frames(path):
    # Mock loader - expects path to a folder of images or single image for testing
    if os.path.isdir(path):
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg'))])
        return [Image.open(f).convert("RGB") for f in files]
    else:
        # Fallback for testing: duplicate one image to simulate video
        img = load_image(path).convert("RGB")
        return [img] * FRAMES_TO_PROCESS

def main():
    # 1. Load Pipeline
    # Note: Qwen-Image is large (20B). Requires significant VRAM (A100/H100 recommended) in fp16.
    # We use 4-bit quantization to fit inconsumer cards (like 24GB/40GB VRAM).
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    pipe = QwenVid2VidPipeline.from_pretrained(
        MODEL_ID, 
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # pipe.to("cuda") # Removed: device_map="auto" handles this with quantization
    
    # Optional: Enable CPU Offload if VRAM is tight
    # pipe.enable_model_cpu_offload() 

    # 2. Prepare Data
    frames = load_frames(INPUT_VIDEO_PATH)[:FRAMES_TO_PROCESS]
    
    # 3. Run Inference
    # Note: strength determines how much we edit. 
    # 0.4-0.6 is usually good for style transfer while keeping structure.
    output_frames = pipe(
        video=frames,
        prompt=PROMPT,
        num_inference_steps=30,
        strength=0.6,
        guidance_scale=4.0,
        seed=1234
    )

    # 4. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, frame in enumerate(output_frames):
        frame.save(os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png"))
    
    # export_to_video(output_frames, os.path.join(OUTPUT_DIR, "result.mp4"), fps=8)
    print(f"Saved results to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
