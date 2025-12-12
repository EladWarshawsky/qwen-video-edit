import torch
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from typing import List, Union, Optional
from PIL import Image
from .attention import QwenVid2VidAttnProcessor

class QwenVid2VidPipeline(QwenImageEditPipeline):
    
    def enable_vid2vid(self):
        """
        Swaps the default Attention Processors with QwenVid2VidAttnProcessor.
        """
        # Iterate over all modules in the transformer
        for name, module in self.transformer.named_modules():
            # Qwen uses 'attn' inside QwenImageTransformerBlock
            if module.__class__.__name__ == "Attention":
                # We specifically want to patch the processor
                if not isinstance(module.processor, QwenVid2VidAttnProcessor):
                    module.processor = QwenVid2VidAttnProcessor()
        print("Vid2Vid Attention Processors Enabled.")

    def set_temporal_mode(self, mode: str):
        """
        Updates the mode of all QwenVid2VidAttnProcessor instances.
        Modes: "standard", "record_anchor", "attend_temporal"
        """
        for module in self.transformer.modules():
            if hasattr(module, "processor") and isinstance(module.processor, QwenVid2VidAttnProcessor):
                module.processor.mode = mode

    def update_prev_cache(self):
        """
        Moves the current 'prev' cache to the next step. 
        In the Processor implementation, 'prev' is set during forward.
        This explicitly tells processors to lock the current frame as 'previous' for the next frame.
        """
        # Actually, since we process frame-by-frame (full denoising loop per frame),
        # we need to capture the *final* latent or key/values of the previous frame.
        # However, vid2vid-zero typically shares K/V *during* the denoising steps.
        # This implies we need to store K/V for *every timestep* if we want exact reconstruction.
        # 
        # SIMPLIFICATION for 20B Model:
        # We effectively treat the generation of Frame N as:
        # "Edit Frame N, but force it to align with Frame 0's structure".
        # We only really need the Anchor. The "Previous Frame" logic is complex 
        # because timesteps might not align perfectly if schedulers are stochastic.
        # 
        # We will focus on ANCHOR consistency first.
        pass

    @torch.no_grad()
    def __call__(
        self,
        video: List[Image.Image],
        prompt: str,
        num_inference_steps: int = 30,
        strength: float = 0.6, # Denoising strength
        guidance_scale: float = 4.0,
        seed: int = 42,
        **kwargs,
    ):
        if not isinstance(video, list):
            raise ValueError("Input must be a list of PIL Images (video frames)")

        # 1. Initialize Pipeline
        self.enable_vid2vid()
        
        # Handle device for generator - 'meta' device (offloading) cannot be used for generator
        device = self.device
        if str(device) == 'meta' or str(device) == 'accelerator':
            # Fallback to cuda if available, else cpu
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Even if 'cpu', offloaded models might prefer 'cuda' for generator if running on gpu eventually
        # But for 'balanced', components move. 
        # Safest is usually 'cpu' for generator seeds to be reproducible across devices, 
        # OR 'cuda' for speed/standard.
        # Given the error, we force a real device.
        if device.type == 'meta': 
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = torch.Generator(device=device).manual_seed(seed)
        
        output_frames = []

        print(f"Processing Video with {len(video)} frames...")

        # ---------------------------------------------------------
        # Frame 0: The Anchor
        # ---------------------------------------------------------
        print("Processing Anchor Frame (0)...")
        self.set_temporal_mode("record_anchor")
        
        # We use a slightly lower strength for the first frame to adhere to prompt 
        # but keep structure, or user defined strength.
        frame_0_out = super().__call__(
            image=video[0],
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            strength=strength,
            true_cfg_scale=guidance_scale,
            generator=generator,
            output_type="pil",
            return_dict=True,
            **kwargs
        ).images[0]
        
        output_frames.append(frame_0_out)

        # ---------------------------------------------------------
        # Subsequent Frames: Attend to Anchor
        # ---------------------------------------------------------
        self.set_temporal_mode("attend_temporal")
        
        for i in range(1, len(video)):
            print(f"Processing Frame {i}/{len(video)}...")
            
            # Reset generator for temporal consistency (optional, but standard in vid2vid)
            # vid2vid-zero uses the SAME seed for noise for all frames to aid consistency
            
            # Re-use the safe device determined above
            generator = torch.Generator(device=device).manual_seed(seed)
            
            frame_out = super().__call__(
                image=video[i],
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                strength=strength, 
                true_cfg_scale=guidance_scale,
                generator=generator,
                output_type="pil",
                return_dict=True,
                **kwargs
            ).images[0]
            
            output_frames.append(frame_out)

        return output_frames
