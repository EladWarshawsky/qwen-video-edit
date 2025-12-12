import os
import torch
from PIL import Image
from qwen_vid2vid.pipeline import QwenVid2VidPipeline
from diffusers.utils import load_image, export_to_video

from transformers import BitsAndBytesConfig

import imageio
import numpy as np

# Configuration
MODEL_ID = "Qwen/Qwen-Image-Edit"
PROMPT = "A van gogh style painting of a car driving on the road"
INPUT_VIDEO_PATH = "without_first_frame.mp4" # User's file
OUTPUT_DIR = "./output_vid2vid"
FRAMES_TO_PROCESS = 8  # Keep low for testing VRAM usage

def read_video_frames(video_path, max_frames=30):
    """
    Reads frames from an mp4 video file using imageio.
    """
    frames = []
    try:
        reader = imageio.get_reader(video_path)
        for i, im in enumerate(reader):
            if i >= max_frames:
                break
            # Imageio returns numpy arrays, convert to PIL
            frames.append(Image.fromarray(im))
        reader.close()
    except Exception as e:
        raise ValueError(f"Could not open video file with imageio: {video_path}. Error: {e}")
        
    return frames

def load_frames(path):
    # Check if path is a video file
    if os.path.isfile(path) and path.endswith(('.mp4', '.avi', '.mov')):
        return read_video_frames(path, max_frames=FRAMES_TO_PROCESS)
    
    # Expects path to a folder of images
    elif os.path.isdir(path):
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg'))])
        return [Image.open(f).convert("RGB") for f in files]
    else:
        # Fallback for testing: try to load as single image
        try:
             img = load_image(path).convert("RGB")
             return [img] * FRAMES_TO_PROCESS
        except Exception:
             # Just return a dummy list if nothing found, to fail gracefully later or here
             raise ValueError(f"Path is not a folder, video file, or valid image: {path}")

from transformers import BitsAndBytesConfig, AutoModelForCausalLM

def main():
    # 1. Load Pipeline
    # Note: Qwen-Image is large (20B).
    # We use 4-bit quantization to fit in consumer cards (like 24GB/40GB VRAM).
    
    # Define 4-bit config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the specific transformer model with quantization
    # Note: Qwen-Image-Edit likely uses a visual language model structure.
    # However, QwenImageEditPipeline in diffusers might expect a specific class.
    # Based on standard diffusers usage for large models (like PixArt or SD3), 
    # we can try passing the quantization config if the library supports it.
    # Since the previous attempt failed, it means this specific pipeline class 
    # doesn't support 'quantization_config' in from_pretrained yet.
    
    # We will try to rely on 'device_map="auto"' and low_cpu_mem_usage to fit.
    # But for 4-bit, we really need the config. 
    # Workaround: Use the 'transformer' argument if possible.
    # We need to know the class. 'QwenImageEditPipeline' has a 'transformer'.
    # If we don't know the exact class, we can try generic AutoModel or similar from transformers
    # but that might not match the expected interface in diffusers.
    
    # ALTERNATIVE: Use the `load_in_4bit=True` kwargs which are sometimes passed to the underlying model loader
    # provided the model itself supports it.
    
    # Let's try loading without explicit quantization_config first OR capture the error?
    # No, the user explicitly asked for optimization.
    
    # Let's try loading the pipeline components.
    # Since I cannot easily determine the transformer class without running code (which fails),
    # I will try to use the 'device_map' and 'start with CPU' approach to at least run.
    # BUT, the user wants 4-bit.
    
    # Let's use the 'low_cpu_mem_usage' and verify if 'load_in_4bit' works as a kwargs directly?
    # Some diffusers pipelines forward kwargs.
    
    print("Loading model with 4-bit quantization config (workaround)...")
    
    # Attempt 2: Load components? No, too complex without class name.
    # Attempt 3: Just rely on `device_map="auto"` and `torch_dtype=torch.bfloat16` 
    # but use `quantization_config` in the generic `from_pretrained` if we treat it as a model?
    # No, `QwenVid2VidPipeline` is a DiffusionPipeline.
    
    # Let's revert to checking if 'diffusers' has 'BitsAndBytesConfig' support.
    # It does not seem to have it in 'utils'. 
    # Maybe we need to pass it to the 'transformer' argument as a loaded model.
    # I'll guess the transformer loading. 
    # If "Qwen/Qwen-Image-Edit" is a huggingface hub model, it has a config.json.
    # If I search the model card... step 28 said "ovedrive/qwen-image-edit-4bit".
    # Using the pre-quantized model is the key!
    
    # Switching to the pre-quantized model ID.
    MODEL_ID_QUANTIZED = "ovedrive/qwen-image-edit-4bit" # Found in search
    
    try:
        print(f"Attempting to load quantized model: {MODEL_ID_QUANTIZED}")
        pipe = QwenVid2VidPipeline.from_pretrained(
            MODEL_ID_QUANTIZED, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to load quantized model directly: {e}")
        print("Falling back to original model with device_map='balanced'...")
        pipe = QwenVid2VidPipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16,
            device_map="balanced" 
        )

    # pipe.to("cuda") # Handled by device_map
    
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
