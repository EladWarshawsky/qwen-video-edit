import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False


class QwenVid2VidAttnProcessor:
    """
    Wraps the default QwenDoubleStreamAttnProcessor2_0 to enable Vid2Vid-Zero 
    temporal consistency by attending to an anchor frame.
    Supports SageAttention acceleration implicitly if installed.
    """
    def __init__(self):
        self.mode = "standard"
        self.anchor_img_key = None
        self.anchor_img_value = None
        self.prev_img_key = None
        self.prev_img_value = None
        self.use_anchor = False 
        self.use_previous_frame = True 

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask=None,
        image_rotary_emb=None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states")

        seq_txt = encoder_hidden_states.shape[1]

        # 1. Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # 2. Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # 3. Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # 4. Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # 5. Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # --- VID2VID-ZERO INJECTION POINT ---
        
        # Save current keys/values for future frames if needed
        current_img_key = img_key.clone()
        current_img_value = img_value.clone()

        if self.mode == "record_anchor":
            self.anchor_img_key = current_img_key.detach()
            self.anchor_img_value = current_img_value.detach()
            # Also reset previous for the start of a sequence
            self.prev_img_key = current_img_key.detach()
            self.prev_img_value = current_img_value.detach()

        elif self.mode == "attend_temporal":
            kv_list_k = [img_key]
            kv_list_v = [img_value]

            # Add Anchor (Conditional)
            if self.use_anchor and self.anchor_img_key is not None:
                kv_list_k.insert(0, self.anchor_img_key)
                kv_list_v.insert(0, self.anchor_img_value)
            
            # Add Previous Frame (Optional but recommended for smoothness)
            if self.use_previous_frame and self.prev_img_key is not None:
                kv_list_k.append(self.prev_img_key)
                kv_list_v.append(self.prev_img_value)

            # Concatenate along sequence dimension (dim 1)
            img_key = torch.cat(kv_list_k, dim=1)
            img_value = torch.cat(kv_list_v, dim=1)

        # ------------------------------------

        # 6. Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # 7. Compute joint attention
        if SAGE_ATTENTION_AVAILABLE:
            # SageAttention Optimization
            # Layout is NHD: (Batch, Seq, Heads, Dim) because of unflatten(-1) earlier without transpose
            joint_hidden_states = sageattn(
                joint_query,
                joint_key,
                joint_value,
                tensor_layout="NHD", 
                is_causal=False 
            )
        else:
            joint_hidden_states = dispatch_attention_fn(
                joint_query,
                joint_key,
                joint_value,
                attn_mask=None, 
                dropout_p=0.0,
                is_causal=False,
            )

        # 8. Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # 9. Split attention outputs back
        # The query length didn't change, so the output length is still (seq_txt + seq_img_original)
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # 10. Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
