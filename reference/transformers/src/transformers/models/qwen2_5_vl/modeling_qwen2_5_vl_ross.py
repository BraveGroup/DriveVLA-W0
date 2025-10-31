"""
Qwen2.5-VL with ROSS (Raw-pixel Observation State Separation) support.
This module extends the base Qwen2.5-VL model to support extracting hidden states
for image and action tokens based on provided spans.
"""

import torch
from torch import nn
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass
from einops import rearrange
from diffusers import AutoencoderKL

from .modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLCausalLMOutputWithPast,
)
from .configuration_qwen2_5_vl import Qwen2_5_VLConfig
from ...utils import ModelOutput
from .modeling_ross.denoiser_sd import RossStableDiffusionXOmni

## 直接copy
@dataclass
class Qwen2_5_VLROSSOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    action_hidden_states: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    raw_pixel_values_vae: Optional[torch.FloatTensor] = None


class Qwen2_5_VLConfigROSS(Qwen2_5_VLConfig):
    """Extended config with ROSS support."""
    
    def __init__(
        self,
        enable_ross: bool = False,
        extract_image_hidden: bool = True,
        extract_action_hidden: bool = True,
        sd_model_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.enable_ross = enable_ross
        self.extract_image_hidden = extract_image_hidden
        self.extract_action_hidden = extract_action_hidden
        self.sd_model_path = sd_model_path


class Qwen2_5_VLForConditionalGenerationROSS(Qwen2_5_VLForConditionalGeneration):

    config_class = Qwen2_5_VLConfigROSS
    
    def __init__(self, config: Qwen2_5_VLConfigROSS):
        super().__init__(config)
        self.enable_ross = getattr(config, 'enable_ross', True)
        self.extract_image_hidden = getattr(config, 'extract_image_hidden', True)
        self.extract_action_hidden = getattr(config, 'extract_action_hidden', True)
        self.denoiser = RossStableDiffusionXOmni(
            unet_path=getattr(config, 'sd_model_path', 'pretrained_models/stable-diffusion-v1-5/unet'),
            z_channel=getattr(config, 'hidden_size', 3584),
            mlp_depth=2,
            n_patches=180,
        )
        # 优先使用更安全的 safetensors 格式，如果不可用则使用 pickle 格式
        vae_path = getattr(config, 'sd_model_path', 'pretrained_models/stable-diffusion-v1-5/unet').replace('/unet', '/vae')
        try:
            # 尝试不使用 pickle 加载 (safetensors)
            self.vae = AutoencoderKL.from_pretrained(vae_path, use_safetensors=True)
        except:
            # 如果失败，则使用 pickle 格式
            self.vae = AutoencoderKL.from_pretrained(vae_path, allow_pickle=True)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae_shift_factor = self.vae.config.shift_factor if self.vae.config.shift_factor is not None else 0.
        self.vae_scaling_factor = self.vae.config.scaling_factor if self.vae.config.scaling_factor is not None else 1.
        
        self.post_init()
    
    def extract_hidden_with_masks(
        self,
        hidden: torch.Tensor,  # [B, L, C]
        image_masks: Optional[torch.Tensor] = None,  # [B, T, N, L] boolean
        action_masks: Optional[torch.Tensor] = None,  # [B, T, L] boolean
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract hidden states using structured boolean masks.
        
        Args:
            hidden: [B, L, C] hidden states
            image_masks: [B, T, N, L] boolean mask for image tokens (T time steps, N images per step)
            action_masks: [B, T, L] boolean mask for action tokens (T time steps)
            
        Returns:
            image_hidden: [B, T, N, L_img, C] extracted image hidden states
            action_hidden: [B, T, L_act, C] extracted action hidden states
        """
        B, L, C = hidden.shape
        image_hidden = None
        action_hidden = None
        
        # Extract image hidden states
        if image_masks is not None:
            # image_masks: [B, T, N, L]
            B_mask, T, N, L_mask = image_masks.shape
            
            # Ensure mask length matches hidden length
            if L_mask != L:
                if L_mask > L:
                    image_masks = image_masks[..., :L]
                    L_mask = L
                else:
                    # Pad mask to match hidden length
                    pad_len = L - L_mask
                    image_masks = torch.nn.functional.pad(image_masks, (0, pad_len), value=False)
                    L_mask = L
            
            # Extract tokens for each time step and image
            image_hidden_list = []
            for t in range(T):
                time_step_images = []
                for n in range(N):
                    batch_images = []
                    for b in range(B):
                        mask = image_masks[b, t, n]  # [L]
                        if mask.any():
                            # Extract tokens where mask is True
                            extracted = hidden[b][mask]  # [num_tokens, C]
                            batch_images.append(extracted)
                        else:
                            # No tokens for this image, use empty tensor
                            batch_images.append(torch.zeros(0, C, device=hidden.device, dtype=hidden.dtype))
                    
                    # Pad to same length within each (t, n)
                    max_tokens = max(img.shape[0] for img in batch_images) if batch_images else 0
                    if max_tokens > 0:
                        padded_batch = []
                        for img in batch_images:
                            if img.shape[0] < max_tokens:
                                pad_len = max_tokens - img.shape[0]
                                img = torch.cat([img, torch.zeros(pad_len, C, device=img.device, dtype=img.dtype)], dim=0)
                            padded_batch.append(img)
                        time_step_images.append(torch.stack(padded_batch, dim=0))  # [B, L_img, C]
                    else:
                        time_step_images.append(torch.zeros(B, 0, C, device=hidden.device, dtype=hidden.dtype))
                
                # Stack images for this time step: [N, B, L_img, C]
                if time_step_images:
                    time_tensor = torch.stack(time_step_images, dim=0)  # [N, B, L_img, C]
                    time_tensor = time_tensor.permute(1, 0, 2, 3)  # [B, N, L_img, C]
                    image_hidden_list.append(time_tensor)
            
            if image_hidden_list:
                # 先在时间维统一 L_img 再堆叠
                max_l_img = max(t.size(2) for t in image_hidden_list)  # [B, N, L_img, C]
                padded_time = []
                for t_tensor in image_hidden_list:
                    cur_l = t_tensor.size(2)
                    if cur_l < max_l_img:
                        pad_len = max_l_img - cur_l
                        # 仅在 L 维（倒数第二维）右侧补零
                        t_tensor = torch.nn.functional.pad(t_tensor, (0, 0, 0, pad_len), value=0)
                    padded_time.append(t_tensor)
                # Stack time steps: [B, T, N, L_img, C]
                image_hidden = torch.stack(padded_time, dim=1)
        
        # Extract action hidden states
        if action_masks is not None:
            # action_masks: [B, T, L]
            B_mask, T, L_mask = action_masks.shape
            
            # Ensure mask length matches hidden length
            if L_mask != L:
                if L_mask > L:
                    action_masks = action_masks[..., :L]
                    L_mask = L
                else:
                    # Pad mask to match hidden length
                    pad_len = L - L_mask
                    action_masks = torch.nn.functional.pad(action_masks, (0, pad_len), value=False)
                    L_mask = L
            
            # Extract tokens for each time step
            action_hidden_list = []
            for t in range(T):
                batch_actions = []
                for b in range(B):
                    mask = action_masks[b, t]  # [L]
                    if mask.any():
                        # Extract tokens where mask is True
                        extracted = hidden[b][mask]  # [num_tokens, C]
                        batch_actions.append(extracted)
                    else:
                        # No tokens for this action, use empty tensor
                        batch_actions.append(torch.zeros(0, C, device=hidden.device, dtype=hidden.dtype))
                
                # Pad to same length within each time step
                max_tokens = max(act.shape[0] for act in batch_actions) if batch_actions else 0
                if max_tokens > 0:
                    padded_batch = []
                    for act in batch_actions:
                        if act.shape[0] < max_tokens:
                            pad_len = max_tokens - act.shape[0]
                            act = torch.cat([act, torch.zeros(pad_len, C, device=act.device, dtype=act.dtype)], dim=0)
                        padded_batch.append(act)
                    action_hidden_list.append(torch.stack(padded_batch, dim=0))  # [B, L_act, C]
                else:
                    action_hidden_list.append(torch.zeros(B, 0, C, device=hidden.device, dtype=hidden.dtype))
            
            if action_hidden_list:
                # 先在时间维统一 L_act 再堆叠
                max_l_act = max(t.size(1) for t in action_hidden_list)  # [B, L_act, C]
                padded_time = []
                for t_tensor in action_hidden_list:
                    cur_l = t_tensor.size(1)
                    if cur_l < max_l_act:
                        pad_len = max_l_act - cur_l
                        # 仅在 L 维（倒数第二维）右侧补零
                        t_tensor = torch.nn.functional.pad(t_tensor, (0, 0, 0, pad_len), value=0)
                    padded_time.append(t_tensor)
                # Stack time steps: [B, T, L_act, C]
                action_hidden = torch.stack(padded_time, dim=1)
        
        return image_hidden, action_hidden
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # ROSS-specific inputs
        raw_pixel_values_vae: Optional[torch.Tensor] = None,  # [B, T, N, C, H, W]
        image_token_masks: Optional[torch.Tensor] = None,  # [B, T, N, L] boolean mask
        action_future_masks: Optional[torch.Tensor] = None,  # [B, T, L] boolean mask
        **kwargs,
    ) -> Union[Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLROSSOutput]:

        output_hidden_states = True
        
        # Call parent forward

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        
        # Extract hidden states if requested
        image_hidden = None
        action_hidden = None
        last_hidden = None
        
        last_hidden = outputs.hidden_states[-1]  # [B, L, C]
        
        # 如果 collator 已经提供了按 batch 拼接且长度对齐的张量，直接使用；
        # 否则在这里进行一次性填充并拼接为张量。
        def _pad_and_cat_list(mask_list: list, target_len: int) -> torch.Tensor:
            padded_list = []
            for m in mask_list:
                pad = target_len - m.shape[-1]
                if pad > 0:
                    m = torch.nn.functional.pad(m, (0, pad), mode='constant', value=0)
                padded_list.append(m)
            return torch.cat(padded_list, dim=0)

        if image_token_masks is not None:
            if isinstance(image_token_masks, (list, tuple)):
                img_max_len = max(m.shape[-1] for m in image_token_masks)
            else:
                img_max_len = image_token_masks.shape[-1]
        else:
            img_max_len = 0

        if action_future_masks is not None:
            if isinstance(action_future_masks, (list, tuple)):
                act_max_len = max(m.shape[-1] for m in action_future_masks)
            else:
                act_max_len = action_future_masks.shape[-1]
        else:
            act_max_len = 0

        target_length = max(img_max_len, act_max_len)

        if image_token_masks is not None and isinstance(image_token_masks, (list, tuple)):
            image_token_masks = _pad_and_cat_list(image_token_masks, target_length)
        # 若为张量则保持不变

        if action_future_masks is not None and isinstance(action_future_masks, (list, tuple)):
            action_future_masks = _pad_and_cat_list(action_future_masks, target_length)
        # 若为张量则保持不变
            
        # Extract hidden states using masks
        if (self.extract_image_hidden or self.extract_action_hidden) and (image_token_masks is not None or action_future_masks is not None):
            image_hidden, action_hidden = self.extract_hidden_with_masks(
                last_hidden,
                image_masks=image_token_masks if self.extract_image_hidden else None,
                action_masks=action_future_masks if self.extract_action_hidden else None,
            )

        ### conditions from t to **predict** t+1
        assert image_hidden.shape[1] == 2, "Currently only support 2 images for each sample"
        assert action_hidden.shape[1] == 2, "Currently only support 2 images for each sample"
        assert raw_pixel_values_vae.shape[0] == 2, "Currently only support 2 images for each sample"

        image_hidden = image_hidden[:, 0].squeeze()             # [bsz, seq_len, dim]
        action_hidden = action_hidden.mean(2)[:, 0].squeeze()   # [bsz, dim]
        cond_pixel_values = raw_pixel_values_vae[0]
        raw_pixel_values_vae = raw_pixel_values_vae[1]          # [bsz, 3, h, w]

        ### conditions from t to **reconstruct** t
        # image_hidden = image_hidden.flatten(0, 1).squeeze()                                 # [bsz * t, seq_len, dim]
        # action_hidden = action_hidden.mean(2).flatten(0, 1)                                 # [bsz * t, dim]
        # raw_pixel_values_vae = raw_pixel_values_vae.permute(1, 0, 2, 3, 4).flatten(0, 1)    # [bsz * t, 3, h, w]

        raw_pixel_values_vae = torch.nn.functional.interpolate(raw_pixel_values_vae, size=(280, 504), mode='bilinear', align_corners=False)

        with torch.no_grad():
            posterior = self.vae.encode(raw_pixel_values_vae).latent_dist
            z_q = (posterior.sample() - self.vae_shift_factor) * self.vae_scaling_factor

        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_hidden = self.denoiser.ln_pre_a(action_hidden)
            image_hidden = self.denoiser.ln_pre(image_hidden)
            # image_hidden = image_hidden + self.denoiser.pos_embed
            image_hidden = rearrange(image_hidden, 'b (h w) c -> b c h w', h=10, w=18)
            ross_loss = self.denoiser(z=image_hidden.float(), target=z_q.float(), z_a=action_hidden.float())
            # NO actions for **reconstruction**
            # ross_loss = self.denoiser(z=image_hidden.float(), target=z_q.float(), z_a=None)


        self._last_logs = {
            "action_loss": float(outputs.loss.detach().cpu()),
            "ross_loss": float(ross_loss.mean().detach().cpu()),
        } 

        outputs.loss = outputs.loss + ross_loss.mean()
        
        # Return ROSS output
        return Qwen2_5_VLROSSOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas if hasattr(outputs, 'rope_deltas') else None,
            image_hidden_states=image_hidden.float(),
            action_hidden_states=action_hidden.float(),
            last_hidden_state=last_hidden,
            raw_pixel_values_vae=[cond_pixel_values, raw_pixel_values_vae],
        )
    
    # @classmethod
    # def from_pretrained(cls, *args, **kwargs):
    #     """Load pretrained model (no extra ROSS-side wiring here)."""
    #     return super().from_pretrained(*args, **kwargs)


__all__ = [
    "Qwen2_5_VLConfigROSS",
    "Qwen2_5_VLForConditionalGenerationROSS",
    "Qwen2_5_VLROSSOutput",
]
