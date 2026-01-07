# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from https://github.com/huggingface/transformers/blob/52daf4ec768fb9ffe84a0c373834172a7c54aecc/src/transformers/models/llama/modeling_llama.py
#
""" PyTorch Emu3 model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch, json
import os
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_emu3 import Emu3Config


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

import sys
sys.path.append("/share/project/yuqi.wang/OmniSim")
from models.policy_head.noise_schedulers import FlowMatchingScheduler

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Emu3Config"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.emu3.modeling_emu3._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.emu3.modeling_emu3._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.emu3.modeling_emu3.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class Emu3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Emu3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(Emu3RMSNorm)


class Emu3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device="cpu").float().to(device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)).to(torch.float32).to(device=device)
        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Emu3LinearScalingRotaryEmbedding(Emu3RotaryEmbedding):
    """Emu3RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class Emu3DynamicNTKScalingRotaryEmbedding(Emu3RotaryEmbedding):
    """Emu3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Emu3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Emu3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Emu3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # self.head_dim = self.hidden_size // self.num_heads
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     raise ValueError(
        #         f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        #         f" and `num_heads`: {self.num_heads})."
        #     )

        # modify here
        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = Emu3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = Emu3LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = Emu3DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Emu3FlashAttention2(Emu3Attention):
    """
    Emu3 flash attention module. This module inherits from `Emu3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Emu3FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (Emu3RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in Emu3FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class Emu3SdpaAttention(Emu3Attention):
    """
    Emu3 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Emu3Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Emu3Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Emu3Model is using Emu3SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


EMU3_ATTENTION_CLASSES = {
    "eager": Emu3Attention,
    "flash_attention_2": Emu3FlashAttention2,
    "sdpa": Emu3SdpaAttention,
}


class Emu3DecoderLayer(nn.Module):
    def __init__(self, config: Emu3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.attention_dropout)
        self.self_attn = EMU3_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = Emu3MLP(config)
        self.input_layernorm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


EMU3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Emu3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Emu3 Model outputting raw hidden-states without any specific head on top.",
    EMU3_START_DOCSTRING,
)
class Emu3PreTrainedModel(PreTrainedModel):
    config_class = Emu3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Emu3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


EMU3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Emu3 Model outputting raw hidden-states without any specific head on top.",
    EMU3_START_DOCSTRING,
)
class Emu3Model(Emu3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Emu3DecoderLayer`]

    Args:
        config: Emu3Config
    """

    def __init__(self, config: Emu3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.dropout = nn.Dropout(config.attention_dropout)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Emu3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = self.dropout(inputs_embeds)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Emu3ForCausalLM(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Emu3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
        >>> from transformers.generation.configuration_utils import GenerationConfig
        >>> from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
        >>> from transformers import Emu3Processor
        >>> from PIL import Image

        >>> model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_EMU3_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> image_processor = AutoImageProcessor.from_pretrained(PATH_TO_CONVERTED_IMAGE_PROCESSER)
        >>> image_tokenizer = AutoModel.from_pretrained(PATH_TO_CONVERTED_TOKENIZER_WEIGHTS).eval()
        >>> processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

        >>> # Generation
        >>> prompt = "An Emu in cartoon style, it is wearing sunglasses."

        >>> pos_inputs = processor(text=prompt, mode='G', ratio="4:3", image_area=model.config.image_area, return_tensors="pt")
        >>> neg_inputs = processor(text="", mode='G', ratio="4:3", image_area=model.config.image_area, return_tensors="pt")

        >>> GENERATION_CONFIG = GenerationConfig(
        >>>     use_cache=True,
        >>>     eos_token_id=model.config.eos_token_id,
        >>>     pad_token_id=model.config.pad_token_id,
        >>>     max_new_tokens=40960,
        >>>     do_sample=True,
        >>>     top_k=2048,
        >>> )

        >>> h, w = pos_inputs.image_size[0]
        >>> constrained_fn = processor.build_prefix_constrained_fn(h, w)
        >>> logits_processor = LogitsProcessorList([
        >>>     UnbatchedClassifierFreeGuidanceLogitsProcessor(
        >>>         classifier_free_guidance, 
        >>>         model,
        >>>         unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
        >>>     ),
        >>>     PrefixConstrainedLogitsProcessor(
        >>>         constrained_fn,
        >>>         num_beams=1,
        >>>     ),
        >>> ])

        >>> outputs = model.generate(pos_inputs.input_ids.to("cuda:0"), GENERATION_CONFIG, logits_processor=logits_processor)
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> mm_list = processor.decode(outputs[0])

        >>> # Understanding
        >>> prompt = "Provide a one-sentence caption for the provided image."
        >>> image = Image.open(TEST_IMAGE_PATH)

        >>> inputs = processor(text=text, image=image, mode='U', padding_side="left", padding="longest", return_tensors="pt")
        >>> input_ids = inputs.input_ids.to("cuda:0")
        >>> GENERATION_CONFIG = GenerationConfig(
        >>>     pad_token_id=tokenizer.pad_token_id,
        >>>     bos_token_id=tokenizer.bos_token_id,
        >>>     eos_token_id=tokenizer.eos_token_id,
        >>> )

        >>> outputs = model.generate(input_ids, GENERATION_CONFIG, max_new_tokens=100)
        >>> outputs = outputs[:, input_ids.shape[-1]:]
        >>> answer = processor.batch_decode(outputs, skip_special_tokens=True)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


class ActionProjector(nn.Module):
    def __init__(self, in_channels, dim):
        super(ActionProjector, self).__init__()
        # Initialize the linear layers W1, W2, W3
        self.W1 = nn.Linear(in_channels, dim)
        self.W2 = nn.Linear(dim + dim, dim)  # Concatenating 2 encodings (dim + dim)
        self.W3 = nn.Linear(dim, dim)
        self.nonlinearity = nn.SiLU()  # swish
        
        # Initialize the weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Use Xavier initialization for the linear layer weights
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W3.weight)
        
        # Initialize the biases to zeros
        if self.W1.bias is not None:
            nn.init.zeros_(self.W1.bias)
        if self.W2.bias is not None:
            nn.init.zeros_(self.W2.bias)
        if self.W3.bias is not None:
            nn.init.zeros_(self.W3.bias)

    def forward(self, x, tau):
        """
        Forward pass through the ActionProjector.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, dim)
            tau (torch.Tensor): Timestep tensor, shape (batch_size, seq_len, dim)

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, dim)
        """
        # Apply linear transformation W1 to each element in the sequence (along dim=2)
        out1 = self.W1(x)  # Shape: (batch_size, seq_len, dim)

        # Concatenate out1 and tau along the last dimension
        out2 = self.W2(torch.cat([out1, tau], dim=-1))  # Shape: (batch_size, seq_len, dim)

        # Apply linear transformation W3
        out3 = self.W3(self.nonlinearity(out2))  # Shape: (batch_size, seq_len, dim)

        return out3

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Emu3MoE(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        
        # Base model (the same as in Emu3ForCausalLM)
        self.model = Emu3Model(config)
        self.vocab_size = config.vocab_size

        if hasattr(config, "vision_loss_weight"):
            self.use_weight = True
            self.vision_loss_weight = config.vision_loss_weight
            self.eov_token_id = config.eov_token_id
            self.bov_token_id = config.bov_token_id
        else:
            self.use_weight = False

        self.action_experts = config.action_experts if hasattr(config, "action_experts") else False
        if self.action_experts:
            action_config = Emu3Config.from_dict(config.action_config)
            self.vision_loss_weight = action_config.vision_loss_weight
            self.action_projector = ActionProjector(config.action_dim, action_config.hidden_size)
            self.action_layers = nn.ModuleList(
                [Emu3DecoderLayer(action_config, layer_idx) for layer_idx in range(action_config.num_hidden_layers)]
            )
            self.action_decoder = FinalLayer(action_config.hidden_size, config.action_dim)
            # self.rf = FlowMatchingScheduler(sample_method="uniform", s = 1.0)
            self.rf = FlowMatchingScheduler(sample_method="beta", s = 1.0)
            self.tau_emb = SinusoidalPosEmb(action_config.hidden_size)
        
        # Output head (same as Emu3ForCausalLM)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Example output will be the same as in Emu3ForCausalLM, with the inclusion of MoE-based processing.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        seq_len = hidden_states.shape[1]

        # processing action
        if action is not None and self.action_experts and self.training:
            # Generate noise with the same shape and data type as the action tensor
            noise = torch.randn_like(action, dtype=action.dtype)

            # Sample tau values and ensure the data type matches the noise tensor
            tau = self.rf.sample_t(noise.shape[0]).to(noise.dtype)

            noise_action = self.rf.add_noise(action, noise, tau)

            # Use forward_action to compute predictions and updated hidden states
            velo_pred, hidden_states_refine = self.forward_action(noise_action, tau, hidden_states)

            # flow matching loss
            loss_action = F.mse_loss(noise - action, velo_pred)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if self.use_weight:
                weights = torch.ones(self.config.vocab_size)
                vision_token_range = range(self.bov_token_id,self.eov_token_id+1)
                weights[vision_token_range] = self.vision_loss_weight
                loss_fct = CrossEntropyLoss(weight=weights.to(logits.device))

            else:
                loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            loss = loss_fct(shift_logits, shift_labels)
            if action is not None and self.action_experts:
                loss += loss_action * self.vision_loss_weight
            # loss = loss_action
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def forward_action(self, z, t, cond):

        # Embed the sampled tau values and adjust the data type to match the noise tensor
        tau_emb = self.tau_emb(t).to(z.dtype)

        # Repeat tau embeddings along the action dimension to match the input shape
        tau_emb = tau_emb.repeat(1, z.shape[1], 1)

        seq_len = cond.shape[1]

        # Compute action embeddings using the action projector and the tau embeddings
        action_hidden_states = self.action_projector(z, tau_emb)

        # Concat in sequence dimension
        action_hidden_states = torch.cat([cond, action_hidden_states], dim=1)
        # transformer layers
        for action_layer in self.action_layers:
            action_hidden_states = action_layer(
                action_hidden_states
            )[0]
        hidden_states, action_hidden_states = action_hidden_states[:, :seq_len, :], action_hidden_states[:, seq_len:, :]
        velo_pred = self.action_decoder(action_hidden_states, tau_emb)

        return velo_pred, hidden_states

    def generate_action(self, outputs, sample_steps = 20, frames = 8, action_dim = 7):

        input_ids = outputs
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        hidden_states = outputs[0]

        # action generation 
        z = torch.randn((batch_size, frames, action_dim), dtype=hidden_states.dtype).to(hidden_states.device)
        dt = 1.0 / sample_steps

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * batch_size).to(hidden_states.device)

            velo_pred, hidden_states_i = self.forward_action(z, t, cond = hidden_states)

            z = z - dt * velo_pred
        
        return z

# ============================================================================
# Emu3Pi0 Model - Pi0.5 style Action Expert Integration
# ============================================================================

class Emu3Pi0SharedLayer:
    """
    A dedicated helper class for shared VLM-Action Expert layer processing.
    It is not an nn.Module to avoid duplicated parameter registration.
    This is compatible with gradient checkpointing.
    """

    def __init__(self, vlm_layer, action_layer):
        self.vlm_layer = vlm_layer
        self.action_layer = action_layer

    def __call__(
            self,
            current_vlm_h: torch.Tensor,
            current_action_h: torch.Tensor,
            vlm_position_ids_original: torch.Tensor,
            combined_attention_mask_4d: torch.Tensor,
            vlm_seq_len: int,
            action_seq_len: int,
            batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single layer of both VLM and Action Expert with shared attention.
        """
        return self.forward(
            current_vlm_h,
            current_action_h,
            vlm_position_ids_original,
            combined_attention_mask_4d,
            vlm_seq_len,
            action_seq_len,
            batch_size,
        )

    def forward(
            self,
            current_vlm_h: torch.Tensor,
            current_action_h: torch.Tensor,
            vlm_position_ids_original: torch.Tensor,
            combined_attention_mask_4d: torch.Tensor,
            vlm_seq_len: int,
            action_seq_len: int,
            batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single layer of both VLM and Action Expert with shared attention.
        """
        _num_heads = self.vlm_layer.self_attn.num_heads
        _head_dim = self.vlm_layer.self_attn.head_dim
        _num_key_value_heads = self.vlm_layer.self_attn.num_key_value_heads
        _num_key_value_groups = self.vlm_layer.self_attn.num_key_value_groups
        _hidden_size_attn_output = _num_heads * _head_dim

        residual_vlm = current_vlm_h
        residual_action = current_action_h

        normed_vlm_h = self.vlm_layer.input_layernorm(current_vlm_h)
        normed_action_h = self.action_layer.input_layernorm(current_action_h)

        q_vlm = self.vlm_layer.self_attn.q_proj(normed_vlm_h)
        k_vlm = self.vlm_layer.self_attn.k_proj(normed_vlm_h)
        v_vlm = self.vlm_layer.self_attn.v_proj(normed_vlm_h)

        q_action = self.action_layer.self_attn.q_proj(normed_action_h)
        k_action = self.action_layer.self_attn.k_proj(normed_action_h)
        v_action = self.action_layer.self_attn.v_proj(normed_action_h)

        q_vlm = q_vlm.view(batch_size, vlm_seq_len, _num_heads, _head_dim).transpose(1, 2)
        k_vlm = k_vlm.view(batch_size, vlm_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)
        v_vlm = v_vlm.view(batch_size, vlm_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)

        q_action = q_action.view(batch_size, action_seq_len, _num_heads, _head_dim).transpose(1, 2)
        k_action = k_action.view(batch_size, action_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)
        v_action = v_action.view(batch_size, action_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)

        # Apply RoPE only to VLM part, not to action expert
        cos, sin = self.vlm_layer.self_attn.rotary_emb(v_vlm, seq_len=vlm_seq_len)
        q_vlm_rope, k_vlm_rope = apply_rotary_pos_emb(q_vlm, k_vlm, cos, sin, vlm_position_ids_original,
                                                      unsqueeze_dim=1)

        # Action expert does not use RoPE - keep original q_action, k_action  [2400, 2408] -> [1600, 1608]
        combined_q = torch.cat([q_vlm_rope, q_action], dim=2)
        combined_k = torch.cat([k_vlm_rope, k_action], dim=2)
        combined_v = torch.cat([v_vlm, v_action], dim=2)

        combined_k_repeated = repeat_kv(combined_k, _num_key_value_groups)
        combined_v_repeated = repeat_kv(combined_v, _num_key_value_groups)

        combined_q = combined_q.contiguous()
        combined_k_repeated = combined_k_repeated.contiguous()
        combined_v_repeated = combined_v_repeated.contiguous()

        attn_output_combined = torch.nn.functional.scaled_dot_product_attention(
            combined_q,
            combined_k_repeated,
            combined_v_repeated,
            attn_mask=combined_attention_mask_4d,
            dropout_p=0.0,
            is_causal=False,
        )

        attn_output_combined = attn_output_combined.transpose(1, 2).contiguous()
        total_seq_len = vlm_seq_len + action_seq_len
        attn_output_combined = attn_output_combined.reshape(batch_size, total_seq_len, _hidden_size_attn_output)

        attn_out_vlm = attn_output_combined[:, :vlm_seq_len, :]
        attn_out_action = attn_output_combined[:, vlm_seq_len:, :]

        attn_out_vlm_proj = self.vlm_layer.self_attn.o_proj(attn_out_vlm)
        current_vlm_h = residual_vlm + self.vlm_layer.dropout(attn_out_vlm_proj)

        residual_vlm_mlp = current_vlm_h
        normed_vlm_for_mlp = self.vlm_layer.post_attention_layernorm(current_vlm_h)
        mlp_out_vlm = self.vlm_layer.mlp(normed_vlm_for_mlp)
        current_vlm_h = residual_vlm_mlp + self.vlm_layer.dropout(mlp_out_vlm)

        attn_out_action_proj = self.action_layer.self_attn.o_proj(attn_out_action)
        current_action_h = residual_action + self.action_layer.dropout(attn_out_action_proj)

        residual_action_mlp = current_action_h
        normed_action_for_mlp = self.action_layer.post_attention_layernorm(current_action_h)
        mlp_out_action = self.action_layer.mlp(normed_action_for_mlp)
        current_action_h = residual_action_mlp + self.action_layer.dropout(mlp_out_action)

        return current_vlm_h, current_action_h

    def forward_with_cache(
            self,
            current_action_h: torch.Tensor,
            action_attention_mask: torch.Tensor,
            vlm_seq_len: int,
            action_seq_len: int,
            batch_size: int,
            vlm_k_rope_cached: torch.Tensor,
            vlm_v_cached: torch.Tensor,
            action_query_valid_1d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a single layer for the Action Expert using cached VLM K/V.
        In this optimized path, only the Action Expert's Query is computed and used
        to attend to the combined VLM (cached) and Action (real-time) K/V pairs.
        """
        _num_heads = self.vlm_layer.self_attn.num_heads
        _head_dim = self.vlm_layer.self_attn.head_dim
        _num_key_value_heads = self.vlm_layer.self_attn.num_key_value_heads
        _num_key_value_groups = self.vlm_layer.self_attn.num_key_value_groups
        _hidden_size_attn_output = _num_heads * _head_dim

        residual_action = current_action_h
        normed_action_h = self.action_layer.input_layernorm(current_action_h)

        q_action = self.action_layer.self_attn.q_proj(normed_action_h)
        k_action = self.action_layer.self_attn.k_proj(normed_action_h)
        v_action = self.action_layer.self_attn.v_proj(normed_action_h)

        q_action = q_action.view(batch_size, action_seq_len, _num_heads, _head_dim).transpose(1, 2)
        k_action = k_action.view(batch_size, action_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)
        v_action = v_action.view(batch_size, action_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)

        # Use cached VLM K/V and combine with real-time Action K/V
        combined_k = torch.cat([vlm_k_rope_cached, k_action], dim=2)
        combined_v = torch.cat([vlm_v_cached, v_action], dim=2)

        combined_k_repeated = repeat_kv(combined_k, _num_key_value_groups)
        combined_v_repeated = repeat_kv(combined_v, _num_key_value_groups)

        q_action = q_action.contiguous()
        combined_k_repeated = combined_k_repeated.contiguous()
        combined_v_repeated = combined_v_repeated.contiguous()

        # Attention with Action Q and combined K/V
        attn_output_action = torch.nn.functional.scaled_dot_product_attention(
            q_action,
            combined_k_repeated,
            combined_v_repeated,
            attn_mask=action_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        attn_output_action = attn_output_action.transpose(1, 2).contiguous()
        attn_output_action = attn_output_action.reshape(batch_size, action_seq_len, _hidden_size_attn_output)

        # Zero-out outputs for padded action query positions to avoid numerical artifacts
        if action_query_valid_1d is not None:
            qv = action_query_valid_1d.to(attn_output_action.dtype).unsqueeze(-1)  # [B, A, 1]
            attn_output_action = attn_output_action * qv

        # Process Action path
        attn_out_action_proj = self.action_layer.self_attn.o_proj(attn_output_action)
        current_action_h = residual_action + self.action_layer.dropout(attn_out_action_proj)

        residual_action_mlp = current_action_h
        normed_action_for_mlp = self.action_layer.post_attention_layernorm(current_action_h)
        mlp_out_action = self.action_layer.mlp(normed_action_for_mlp)
        current_action_h = residual_action_mlp + self.action_layer.dropout(mlp_out_action)

        return current_action_h


class Emu3Pi0(Emu3PreTrainedModel):
    """
    Emu3Pi0 Model combining Emu3MoE (VLM) and Action Expert using Pi0-style approach.
    
    This model follows the Pi0 paradigm where VLM and action expert are processed
    layer by layer with concatenated Q/K/V states for shared attention.
    """
    _tied_weights_keys = ["vlm.lm_head.weight"]

    def __init__(self, config, pretrain_vlm_path):
        super().__init__(config)

        # Store configs for different components
        self.vlm_config = getattr(config, 'vlm_config', config)
        self.action_config = getattr(config, 'action_config', config)

        # VLM and Action Expert
        self.vlm, loading_info = Emu3MoE.from_pretrained(
            pretrain_vlm_path,
            attn_implementation="sdpa",
            torch_dtype=self.config.torch_dtype,
            output_loading_info=True
        )
        print("Missing keys in loaded VLM:", loading_info["missing_keys"])
        print("Unexpected keys in loaded VLM:", loading_info["unexpected_keys"])
        print("Mismatched sizes in loaded VLM:", loading_info.get("mismatched_keys", "N/A"))

        # Create Action Expert - 独立的Emu3Model
        self.action_expert = Emu3Model(self.action_config)

        # Action specific components
        action_dim = getattr(config, 'action_dim', 3)
        action_hidden_size = self.action_config.hidden_size

        # Training configurations
        self.train_action_only = getattr(config, 'train_action_only', True)
        self.action_loss_weight = getattr(config, 'action_loss_weight', 1.0)
        self.action_frames = getattr(config, 'action_frames', 8)
        self.pre_action_frames = getattr(self.action_config, 'pre_action_frames', 3)
        self.action_sample_steps = getattr(config, 'action_sample_steps', 10)

        state_input_dim = self.pre_action_frames * action_dim + 4  # 4 for cmd one-hot
        self.state_projector = nn.Sequential(
            nn.Linear(state_input_dim, action_hidden_size),
            nn.SiLU(),
            nn.Linear(action_hidden_size, action_hidden_size),
        )

        self.action_projector = ActionProjector(action_dim, action_hidden_size, action_frames=self.action_frames)
        self.action_decoder = FinalLayer(action_hidden_size, action_dim)

        # Flow matching scheduler
        self.rf = FlowMatchingScheduler(sample_method="beta", s=1.0)
        self.tau_emb = SinusoidalPosEmb(action_hidden_size)

        # Create shared layer modules for gradient checkpointing
        # These are now plain Python objects, not nn.Modules, to avoid registration issues.
        self.shared_layers = [
            Emu3Pi0SharedLayer(vlm_layer, action_layer)
            for vlm_layer, action_layer in zip(self.vlm.model.layers, self.action_expert.layers)
        ]

        # Gradient checkpointing configuration
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            action: torch.Tensor,
            pre_action: torch.Tensor,
            cmd: torch.Tensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Forward pass combining Emu3MoE (VLM) and Action Expert using pi0-style shared attention.
        This method is primarily designed for training.

        Args:
            action (`torch.Tensor` of shape `(batch_size, action_frames, action_dim)`):
                Action sequence for training. The model will compute action loss using flow matching,
                with shared attention between VLM and action expert.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from reference.Emu3.emu3.mllm.modeling_emu3 import Emu3Pi0
        >>> from reference.Emu3.emu3.mllm.configuration_emu3 import Emu3Config

        >>> # Training with actions (interleaved shared attention)
        >>> config = Emu3Config() # Ensure VLM and Action configs are compatible for shared attention
        >>> model = Emu3Pi0(config)
        >>> input_ids = torch.randint(0, 1000, (2, 32))
        >>> action_gt = torch.randn(2, model.action_frames, config.action_dim if hasattr(config, 'action_dim') else 7)
        >>> outputs = model(input_ids=input_ids, action=action_gt, labels=input_ids)
        >>> loss = outputs.loss  # Combined VLM loss + action loss
        ```
        """

        # use_cache is not typically used in this shared attention training path, set to False by default or ignored.
        use_cache = False  # Explicitly setting use_cache to False for this training-focused forward pass
        return_dict = return_dict if return_dict is not None else self.vlm_config.use_return_dict

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Initial VLM embeddings
        if inputs_embeds is None:
            vlm_initial_hidden_states = self.vlm.model.embed_tokens(input_ids)
        else:
            vlm_initial_hidden_states = inputs_embeds  # Assuming these are VLM embeddings

        vlm_seq_len = vlm_initial_hidden_states.shape[1]  # 2400

        noise = torch.randn_like(action, dtype=action.dtype, device=device)
        tau_values = self.rf.sample_t(noise.shape[0]).to(noise.dtype).to(device)
        noisy_action = self.rf.add_noise(action, noise, tau_values)

        action_frames_len = noisy_action.shape[1]  # 8
        tau_emb = self.tau_emb(tau_values).to(noisy_action.dtype)
        tau_emb_expanded = tau_emb.unsqueeze(1).expand(-1, action_frames_len, -1)

        action_hidden_states_no_state = self.action_projector(noisy_action, tau_emb_expanded)

        state_input = torch.cat([pre_action.view(batch_size, -1), cmd], dim=1)
        state_token_embedding = self.state_projector(state_input).unsqueeze(1)  # (bs, 1, h)

        action_initial_hidden_states = torch.cat([state_token_embedding, action_hidden_states_no_state], dim=1)
        action_seq_len = action_initial_hidden_states.shape[1]

        current_vlm_h = vlm_initial_hidden_states
        current_action_h = action_initial_hidden_states

        num_layers = len(self.vlm.model.layers)  # 32 layers

        vlm_position_ids_original = position_ids
        if vlm_position_ids_original is None:
            vlm_position_ids_original = torch.arange(vlm_seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(
                batch_size, -1)

        combined_attention_mask_4d = self.create_causal_style_attention_mask(
            vlm_seq_len, action_seq_len, attention_mask,
            input_ids,
            batch_size, device, current_vlm_h.dtype
        )

        # Check for gradient checkpointing
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # Layer-wise processing with gradient checkpointing support
        for layer_idx in range(num_layers):
            shared_layer = self.shared_layers[layer_idx]

            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing with proper nn.Module
                current_vlm_h, current_action_h = self._gradient_checkpointing_func(
                    shared_layer.__call__,
                    current_vlm_h,
                    current_action_h,
                    vlm_position_ids_original,
                    combined_attention_mask_4d,
                    vlm_seq_len,
                    action_seq_len,
                    batch_size,
                )
            else:
                # Normal forward pass without checkpointing
                current_vlm_h, current_action_h = shared_layer(
                    current_vlm_h,
                    current_action_h,
                    vlm_position_ids_original,
                    combined_attention_mask_4d,
                    vlm_seq_len,
                    action_seq_len,
                    batch_size,
                )

        final_vlm_hidden_states_for_lm_head = self.vlm.model.norm(current_vlm_h)
        final_action_hidden_for_decode = self.action_expert.norm(current_action_h)

        velo_t_pred = self.action_decoder(final_action_hidden_for_decode[:, 1:, :], tau_emb_expanded)

        action_loss = F.mse_loss(noise - action, velo_t_pred)

        self.action_loss_weight = 1.0
        self.vlm_loss_weight = 0.0
        logits = None

        if self.vlm_loss_weight > 0.0:
            # --- LM Head and Loss Calculation ---
            if self.vlm_config.pretraining_tp > 1:
                lm_head_slices = self.vlm.lm_head.weight.split(
                    self.vlm_config.vocab_size // self.vlm_config.pretraining_tp, dim=0)
                logits = [F.linear(final_vlm_hidden_states_for_lm_head, lm_head_slices[i]) for i in
                          range(self.vlm_config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.vlm.lm_head(final_vlm_hidden_states_for_lm_head)
            logits = logits.float()

            vlm_loss = None
            use_vision_loss = True
            if use_vision_loss:
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    weights = torch.ones(self.config.vocab_size)
                    vision_token_range = range(self.bov_token_id, self.eov_token_id + 1)
                    weights[vision_token_range] = self.vision_loss_weight
                    loss_fct = CrossEntropyLoss(weight=weights.to(logits.device))

                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    vlm_loss = loss_fct(shift_logits, shift_labels)
            else:
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # 只计算Future Action部分的loss [boa_pos, eoa_pos)
                    boa_token_id = 151844  # BOA token ID
                    eoa_token_id = 151845  # EOA token ID
                    vlm_ids = input_ids[:, :vlm_seq_len]

                    boa_positions = torch.argmax((vlm_ids == boa_token_id).float(), dim=1)  # [batch_size]
                    eoa_positions = torch.argmax((vlm_ids == eoa_token_id).float(), dim=1)  # [batch_size]

                    # 1) 在原始长�?vlm_seq_len 上构造全序列 mask
                    orig_indices = torch.arange(vlm_seq_len, device=labels.device).unsqueeze(0).expand(batch_size,
                                                                                                       -1)  # [B, vlm_seq_len]
                    full_mask = (orig_indices > boa_positions.unsqueeze(1)) & (
                                orig_indices <= eoa_positions.unsqueeze(1))  # [B, vlm_seq_len]
                    # 2) 丢掉�?0 位，对齐�?shift_labels (长度 vlm_seq_len-1)
                    future_action_mask = full_mask[:, 1:]  # [B, vlm_seq_len-1]
                    # 3) 应用�?shift_labels 上：非Future Action位置设为-100（CrossEntropyLoss会忽略）
                    masked_shift_labels = shift_labels.clone()
                    masked_shift_labels[~future_action_mask] = -100

                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, self.vlm_config.vocab_size)
                    masked_shift_labels = masked_shift_labels.view(-1)
                    masked_shift_labels = masked_shift_labels.to(shift_logits.device)
                    vlm_loss = loss_fct(shift_logits, masked_shift_labels)
            total_loss = self.action_loss_weight * action_loss + self.vlm_loss_weight * vlm_loss
        else:
            total_loss = self.action_loss_weight * action_loss

        # For this training-focused forward pass, past_kv, hidden_states (intermediate), and attentions are not returned.
        if not return_dict:
            output = (logits, None, None, None)  # logits, past_kv, hidden_states, attentions
            return (total_loss,) + output if total_loss is not None else output

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=None,  # Not computed/returned in this training path
            hidden_states=None,  # Not computed/returned in this training path
            attentions=None,  # Not computed/returned in this training path
        )

    @torch.no_grad()
    def sample_actions(
            self,
            input_ids: torch.LongTensor,
            pre_action: torch.Tensor,
            cmd: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            num_inference_steps: Optional[int] = None,
            action_frames: Optional[int] = None,
            action_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample actions using iterative denoising with shared VLM-Action Expert attention,
        mimicking the training forward pass and pi0-style Euler sampling.

        Args:
            input_ids: Input token ids for the VLM.
            pre_action: Past action sequence, used for state.
            cmd: Command one-hot vector, used for state.
            attention_mask: Attention mask for input_ids.
            position_ids: Position ids for input_ids.
            inputs_embeds: Optional pre-computed VLM embeddings.
            num_inference_steps: Number of denoising steps. Defaults to self.action_sample_steps.
            action_frames: Number of action frames to generate. Defaults to self.action_frames.
            action_dim: Dimension of the action. Defaults to self.action_config.action_dim or a fallback.

        Returns:
            torch.Tensor: Generated actions of shape [batch_size, action_frames, action_dim]
        """
        self.eval()  # Ensure model is in eval mode for inference

        _num_inference_steps = num_inference_steps if num_inference_steps is not None else self.action_sample_steps
        _action_frames = action_frames if action_frames is not None else self.action_frames
        _action_dim = action_dim if action_dim is not None else getattr(self.action_config, 'action_dim',
                                                                        getattr(self.config, 'action_dim', 3))

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Initial VLM embeddings
        if inputs_embeds is None:
            vlm_initial_hidden_states = self.vlm.model.embed_tokens(input_ids)
        else:
            vlm_initial_hidden_states = inputs_embeds

        vlm_seq_len = vlm_initial_hidden_states.shape[1]

        # Initialize random noise for actions
        z = torch.randn(batch_size, _action_frames, _action_dim, device=device, dtype=vlm_initial_hidden_states.dtype)

        state_input = torch.cat([pre_action.view(batch_size, -1), cmd], dim=1)
        state_token_embedding = self.state_projector(state_input).unsqueeze(1)

        # Time stepping according to pi0 reference
        dt = -1.0 / _num_inference_steps
        dt_tensor = torch.tensor(dt, dtype=vlm_initial_hidden_states.dtype, device=device)

        current_time = torch.tensor(1.0, dtype=vlm_initial_hidden_states.dtype, device=device)

        # Loop while current_time is greater than or equal to a very small positive number (or -dt/2 as in ref)
        while current_time >= -dt / 2:  # Condition based on time decreasing towards 0
            # Ensure t_tensor for tau_emb has batch_size dimension
            # current_time is scalar here, expand for batch
            t_tensor = current_time.expand(batch_size).to(z.dtype)

            # --- Start of equivalent to denoise_step(z, current_time) ---
            current_vlm_h = vlm_initial_hidden_states  # VLM embeddings are constant per step

            tau_emb = self.tau_emb(t_tensor).to(z.dtype)  # t_tensor should be [batch_size]
            tau_emb_expanded = tau_emb.unsqueeze(1).expand(-1, _action_frames, -1)
            action_hidden_states_no_state = self.action_projector(z, tau_emb_expanded)
            current_action_h = torch.cat([state_token_embedding, action_hidden_states_no_state], dim=1)

            num_layers = len(self.vlm.model.layers)

            vlm_position_ids_original = position_ids
            if vlm_position_ids_original is None:
                vlm_position_ids_original = torch.arange(vlm_seq_len, device=device, dtype=torch.long).unsqueeze(
                    0).expand(batch_size, -1)

            vlm_attention_mask_for_combined = attention_mask
            action_seq_len_with_state = current_action_h.shape[1]
            combined_attention_mask_4d = self.create_causal_style_attention_mask(
                vlm_seq_len, action_seq_len_with_state, vlm_attention_mask_for_combined,
                input_ids,
                batch_size, device, current_vlm_h.dtype
            )

            # Layer processing
            for layer_idx in range(num_layers):
                shared_layer = self.shared_layers[layer_idx]
                current_vlm_h, current_action_h = shared_layer(
                    current_vlm_h,
                    current_action_h,
                    vlm_position_ids_original,
                    combined_attention_mask_4d,
                    vlm_seq_len,
                    action_seq_len_with_state,
                    batch_size,
                )

            final_action_hidden_for_decode = self.action_expert.norm(current_action_h)
            velo_t_pred = self.action_decoder(final_action_hidden_for_decode[:, 1:, :], tau_emb_expanded)
            # --- End of equivalent to denoise_step ---

            # Euler step: z_t = z_{t+dt} + dt * v(z_{t+dt}, t+dt)
            # Since dt is negative, this effectively means z_new = z_current - abs(dt) * velo_t_pred
            z = z + dt * velo_t_pred  # dt is a negative scalar, velo_t_pred is [B, AF, AD]
            current_time += dt_tensor  # Update time

        return z

    @torch.no_grad()
    def sample_actions_with_kv_cache(
            self,
            input_ids: torch.LongTensor,
            pre_action: torch.Tensor,
            cmd: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            num_inference_steps: Optional[int] = None,
            action_frames: Optional[int] = None,
            action_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample actions using iterative denoising with a VLM KV-cache and action-only query.
        This highly optimized method pre-computes the VLM's Key and Value states and,
        during denoising, only computes the Action Expert's Query, which then attends
        to the combined VLM and Action K/V context.
        """
        self.eval()

        _num_inference_steps = num_inference_steps if num_inference_steps is not None else self.action_sample_steps
        _action_frames = action_frames if action_frames is not None else self.action_frames
        _action_dim = action_dim if action_dim is not None else getattr(self.action_config, 'action_dim',
                                                                        getattr(self.config, 'action_dim', 3))

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if inputs_embeds is None:
            vlm_initial_hidden_states = self.vlm.model.embed_tokens(input_ids)
        else:
            vlm_initial_hidden_states = inputs_embeds

        vlm_seq_len = vlm_initial_hidden_states.shape[1]

        vlm_position_ids_original = position_ids
        if vlm_position_ids_original is None:
            vlm_position_ids_original = torch.arange(vlm_seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(
                batch_size, -1)

        # --- Pre-compute and cache VLM K and V states ---
        vlm_k_rope_cache, vlm_v_cache = [], []

        current_vlm_h_for_cache = vlm_initial_hidden_states
        _num_heads = self.vlm.model.layers[0].self_attn.num_heads
        _head_dim = self.vlm.model.layers[0].self_attn.head_dim
        _num_key_value_heads = self.vlm.model.layers[0].self_attn.num_key_value_heads

        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        vlm_attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, vlm_seq_len), vlm_initial_hidden_states, 0
        )

        for vlm_layer in self.vlm.model.layers:
            normed_vlm_h = vlm_layer.input_layernorm(current_vlm_h_for_cache)

            q_vlm = vlm_layer.self_attn.q_proj(normed_vlm_h).view(batch_size, vlm_seq_len, _num_heads,
                                                                  _head_dim).transpose(1, 2)
            k_vlm = vlm_layer.self_attn.k_proj(normed_vlm_h).view(batch_size, vlm_seq_len, _num_key_value_heads,
                                                                  _head_dim).transpose(1, 2)
            v_vlm = vlm_layer.self_attn.v_proj(normed_vlm_h).view(batch_size, vlm_seq_len, _num_key_value_heads,
                                                                  _head_dim).transpose(1, 2)

            cos, sin = vlm_layer.self_attn.rotary_emb(v_vlm, seq_len=vlm_seq_len)
            _, k_vlm_rope = apply_rotary_pos_emb(q_vlm, k_vlm, cos, sin, vlm_position_ids_original, unsqueeze_dim=1)

            vlm_k_rope_cache.append(k_vlm_rope)
            vlm_v_cache.append(v_vlm)

            layer_outputs = vlm_layer(current_vlm_h_for_cache, attention_mask=vlm_attention_mask_4d,
                                      position_ids=vlm_position_ids_original)
            current_vlm_h_for_cache = layer_outputs[0]

        z = torch.randn(batch_size, _action_frames, _action_dim, device=device, dtype=vlm_initial_hidden_states.dtype)
        state_input = torch.cat([pre_action.view(batch_size, -1), cmd], dim=1)
        state_token_embedding = self.state_projector(state_input).unsqueeze(1)

        dt = -1.0 / _num_inference_steps
        current_time = torch.tensor(1.0, dtype=z.dtype, device=device)

        while current_time.item() >= -dt / 2:  # Ensure comparison is done with item() for scalar tensor
            t_tensor = current_time.expand(batch_size).to(z.dtype)
            tau_emb = self.tau_emb(t_tensor).to(z.dtype)
            tau_emb_expanded = tau_emb.unsqueeze(1).expand(-1, _action_frames, -1)

            action_hidden_states_no_state = self.action_projector(z, tau_emb_expanded)
            current_action_h = torch.cat([state_token_embedding.clone(), action_hidden_states_no_state], dim=1)
            action_seq_len_with_state = current_action_h.shape[1]

            combined_attention_mask_4d = self.create_causal_style_attention_mask(
                vlm_seq_len, action_seq_len_with_state, attention_mask,
                input_ids, batch_size, device, vlm_initial_hidden_states.dtype
            )
            action_attention_mask = combined_attention_mask_4d[:, :, vlm_seq_len:, :]

            for layer_idx in range(len(self.shared_layers)):
                shared_layer = self.shared_layers[layer_idx]

                current_action_h = shared_layer.forward_with_cache(
                    current_action_h,
                    action_attention_mask=action_attention_mask,
                    vlm_seq_len=vlm_seq_len,
                    action_seq_len=action_seq_len_with_state,
                    batch_size=batch_size,
                    vlm_k_rope_cached=vlm_k_rope_cache[layer_idx],
                    vlm_v_cached=vlm_v_cache[layer_idx],
                )

            final_action_hidden_for_decode = self.action_expert.norm(current_action_h)
            velo_t_pred = self.action_decoder(final_action_hidden_for_decode[:, 1:, :], tau_emb_expanded)

            z = z + dt * velo_t_pred
            current_time += dt

        return z

    @staticmethod
    def create_pi0_style_attention_mask(vlm_seq_len, action_seq_len, vlm_attention_mask_original,
                                        input_ids, batch_size, device, dtype, boa_token_id=151844, eoa_token_id=151845,
                                        visualize=False, save_path="attention_mask_vis.png"):
        """
        Create Pi0-style attention mask for combined sequence: [Text/Image+PreAction] [Future Action] [Action Expert]
        
        Attention Rules (按照用户提供的图表实�?:
        - Text/Image + Pre Action: 双向注意力，互相都能看到 (boa_pos之前的token)
        - Future Action: causal，能看到Text/Image+PreAction，看不到Action Expert (boa_pos到eoa_pos之间的token)
        - Action Expert: 双向，能看到Text/Image+PreAction，看不到Future Action (VLM之后的action_seq_len个token)
        
        序列组织�?
        - Text/Image + Pre Action: [0, boa_pos) - 双向注意�?
        - Future Action: [boa_pos, eoa_pos+1) - causal注意�?
        - Action Expert: [vlm_seq_len, vlm_seq_len + action_seq_len) - 双向注意�?
        
        Args:
            visualize: 是否生成可视化图�?
            save_path: 可视化图片保存路�?
        """
        # 目前这个attention mask还需要修改，直接报错
        raise NotImplementedError("Pi0-style attention mask is not implemented yet")
        total_seq_len = vlm_seq_len + action_seq_len

        # Initialize combined mask (additive, so 0 means attend, -inf means no attend)
        combined_mask = torch.full(
            (batch_size, 1, total_seq_len, total_seq_len),
            torch.finfo(dtype).min / 2,
            device=device,
            dtype=dtype
        )

        # 必须有vlm_attention_mask_original且为2D
        if vlm_attention_mask_original is None:
            raise ValueError("vlm_attention_mask_original is required for Pi0-style attention")
        if vlm_attention_mask_original.dim() != 2:
            raise ValueError(f"vlm_attention_mask_original must be 2D, got {vlm_attention_mask_original.dim()}D")

        # 找到boa和eoa位置 - 向量化操�?
        vlm_ids = input_ids[:, :vlm_seq_len]  # [batch_size, vlm_seq_len]

        # 检查每个样本是否包含BOA和EOA token
        boa_exists = (vlm_ids == boa_token_id).any(dim=1)  # [batch_size]
        eoa_exists = (vlm_ids == eoa_token_id).any(dim=1)  # [batch_size]

        assert boa_exists.all(), f"BOA token ({boa_token_id}) missing in some samples"
        assert eoa_exists.all(), f"EOA token ({eoa_token_id}) missing in some samples"

        boa_positions = torch.argmax((vlm_ids == boa_token_id).float(), dim=1)  # [batch_size]
        eoa_positions = torch.argmax((vlm_ids == eoa_token_id).float(), dim=1)  # [batch_size]

        # 创建序列位置索引
        seq_indices = torch.arange(vlm_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, VLM_S]

        # 定义各部分的mask - 向量�?
        boa_pos_expanded = boa_positions.unsqueeze(1).expand(-1, vlm_seq_len)  # [B, VLM_S]
        eoa_pos_expanded = eoa_positions.unsqueeze(1).expand(-1, vlm_seq_len)  # [B, VLM_S]

        # Text/Image + Pre Action: [0, boa_pos) - 双向注意�?
        text_pre_mask = seq_indices < boa_pos_expanded  # [B, VLM_S]

        # Future Action: [boa_pos, eoa_pos+1) - causal注意�? 
        future_mask = (seq_indices >= boa_pos_expanded) & (seq_indices <= eoa_pos_expanded)  # [B, VLM_S]

        # Action Expert: [vlm_seq_len, vlm_seq_len + action_seq_len) - 双向注意�?
        action_expert_start = vlm_seq_len
        action_expert_end = vlm_seq_len + action_seq_len

        # 1. Text/Image + Pre Action 内部: 双向注意�?+ padding
        text_pre_valid = text_pre_mask & vlm_attention_mask_original.bool()  # [B, VLM_S]
        text_pre_attend = text_pre_valid.unsqueeze(2) & text_pre_valid.unsqueeze(1)  # [B, VLM_S, VLM_S]
        combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len] = torch.where(
            text_pre_attend,
            torch.tensor(0.0, device=device, dtype=dtype),
            combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len]
        )

        # 2. Future Action 内部: causal注意�?+ padding
        future_valid = future_mask & vlm_attention_mask_original.bool()  # [B, VLM_S]
        # 创建causal mask: 只能看到当前和之前的
        future_seq_idx = torch.arange(vlm_seq_len, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, VLM_S]
        future_causal = future_seq_idx.transpose(1, 2) >= future_seq_idx  # [1, VLM_S, VLM_S] - causal mask
        future_attend = future_valid.unsqueeze(2) & future_valid.unsqueeze(1) & future_causal  # [B, VLM_S, VLM_S]
        combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len] = torch.where(
            future_attend,
            torch.tensor(0.0, device=device, dtype=dtype),
            combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len]
        )

        # 3. Action Expert 内部: 双向注意�?
        if action_seq_len > 0:
            combined_mask[:, :, action_expert_start:action_expert_end, action_expert_start:action_expert_end] = 0.0

        # 4. Future Action -> Text/Image + Pre Action
        # future_valid: [B, VLM_S] - True where tokens are Future Action
        # text_pre_valid: [B, VLM_S] - True where tokens are Text/Image + Pre Action
        future_query_mask = future_valid.unsqueeze(2).expand(-1, -1, vlm_seq_len)  # [B, VLM_S, VLM_S] query维度
        text_pre_key_mask = text_pre_valid.unsqueeze(1).expand(-1, vlm_seq_len, -1)  # [B, VLM_S, VLM_S] key维度
        future_to_text_pre = future_query_mask & text_pre_key_mask  # [B, VLM_S, VLM_S]
        combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len] = torch.where(
            future_to_text_pre,
            torch.tensor(0.0, device=device, dtype=dtype),
            combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len]
        )

        # 5. Action Expert -> Text/Image + Pre Action  
        if action_seq_len > 0:
            text_pre_for_expert = text_pre_valid.unsqueeze(1).expand(-1, action_seq_len, -1)  # [B, A_S, VLM_S]
            text_pre_for_expert = text_pre_for_expert.unsqueeze(1)  # [B, 1, A_S, VLM_S] 匹配combined_mask的维�?
            combined_mask[:, :, action_expert_start:action_expert_end, :vlm_seq_len] = torch.where(
                text_pre_for_expert,
                torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(torch.finfo(dtype).min / 2, device=device, dtype=dtype)
            )

        return combined_mask

    @staticmethod
    def create_causal_style_attention_mask(vlm_seq_len, action_seq_len, vlm_attention_mask_original,
                                           input_ids, batch_size, device, dtype, boa_token_id=151844,
                                           eoa_token_id=151845, action_is_causal=False,
                                           visualize=False, save_path="attention_mask_causal_vis.png",
                                           action_attention_mask_1d: Optional[torch.Tensor] = None):
        """
        Create Causal-style attention mask for combined sequence: [VLM Causal] [Action Expert]
        
        Attention Rules (Causal变体):
        - VLM全部 (Text/Image + Pre Action + Future Action): 全部causal attention（只能看到当前和之前的token），不能看到Action Expert
        - Action Expert: 双向或Causal attention，能看到Text/Image + Pre Action，不能看到Future Action (Future Action is defined as the content from the second BOA token onwards)
        
        序列组织�?
        - VLM全部: [0, vlm_seq_len) - 全部causal
        - Action Expert: [vlm_seq_len, vlm_seq_len + action_seq_len) - 双向或causal，但只能看到VLM中第二个BOA之前的内�?
        
        Args:
            action_is_causal: If True, the action expert part of the mask will be causal.
            visualize: 是否生成可视化图�?
            save_path: 可视化图片保存路�?
        """
        total_seq_len = vlm_seq_len + action_seq_len

        # Initialize combined mask (additive, so 0 means attend, -inf means no attend)
        combined_mask = torch.full(
            (batch_size, 1, total_seq_len, total_seq_len),
            torch.finfo(dtype).min / 2,
            device=device,
            dtype=dtype
        )

        # 必须有vlm_attention_mask_original且为2D
        if vlm_attention_mask_original is None:
            raise ValueError("vlm_attention_mask_original is required for Causal-style attention")
        if vlm_attention_mask_original.dim() != 2:
            raise ValueError(f"vlm_attention_mask_original must be 2D, got {vlm_attention_mask_original.dim()}D")

        vlm_ids = input_ids[:, :vlm_seq_len]

        # 检查每个样本是否包含BOA和EOA token
        is_boa = (vlm_ids == boa_token_id)
        boa_counts = is_boa.sum(dim=1)
        assert torch.all(boa_counts == 2), f"Expected 2 BOA tokens per sample, but found counts: {boa_counts}"

        # 找到第二个boa的位�?
        cumsum_boa = torch.cumsum(is_boa.float(), dim=1)
        is_second_boa = (cumsum_boa == 2) & is_boa
        second_boa_positions = torch.argmax(is_second_boa.float(), dim=1)  # [batch_size]

        # 1. VLM全部内部: causal attention + padding
        vlm_valid = vlm_attention_mask_original.bool()  # [B, VLM_S] - 使用原始padding mask
        vlm_seq_idx = torch.arange(vlm_seq_len, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, VLM_S]
        vlm_causal = vlm_seq_idx.transpose(1, 2) >= vlm_seq_idx  # [1, VLM_S, VLM_S] - causal mask
        vlm_attend = vlm_valid.unsqueeze(2) & vlm_valid.unsqueeze(1) & vlm_causal  # [B, VLM_S, VLM_S]
        combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len] = torch.where(
            vlm_attend,
            torch.tensor(0.0, device=device, dtype=dtype),
            combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len]
        )

        # 2. Action Expert 内部: 双向或causal注意�?
        action_expert_start = vlm_seq_len
        action_expert_end = vlm_seq_len + action_seq_len
        if action_seq_len > 0:
            if action_is_causal:
                action_seq_idx = torch.arange(action_seq_len, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, A_S]
                action_causal_mask = action_seq_idx.transpose(1, 2) >= action_seq_idx  # [1, A_S, A_S]
                action_mask_quadrant = torch.full(
                    (batch_size, 1, action_seq_len, action_seq_len), torch.finfo(dtype).min / 2, device=device,
                    dtype=dtype
                )
                action_mask_quadrant = torch.where(
                    action_causal_mask,
                    torch.tensor(0.0, device=device, dtype=dtype),
                    action_mask_quadrant
                )
                combined_mask[:, :, action_expert_start:action_expert_end,
                action_expert_start:action_expert_end] = action_mask_quadrant
            else:
                combined_mask[:, :, action_expert_start:action_expert_end, action_expert_start:action_expert_end] = 0.0

        # 3. Action Expert -> VLM (只能看到第二个BOA之前的部�?
        if action_seq_len > 0:
            seq_indices = torch.arange(vlm_seq_len, device=device).unsqueeze(0)  # [1, VLM_S]

            # Action Expert can see VLM content before the second BOA token.
            vlm_visible_to_action_expert_mask = seq_indices < second_boa_positions.unsqueeze(1)  # [B, VLM_S]

            # Also respect original padding
            vlm_visible_to_action_expert_mask = vlm_visible_to_action_expert_mask & vlm_attention_mask_original.bool()

            # Expand to fit the mask quadrant shape [B, 1, A_S, VLM_S]
            vlm_visible_to_action_expert_mask = vlm_visible_to_action_expert_mask.unsqueeze(1).expand(-1,
                                                                                                      action_seq_len,
                                                                                                      -1)
            vlm_visible_to_action_expert_mask = vlm_visible_to_action_expert_mask.unsqueeze(1)

            combined_mask[:, :, action_expert_start:action_expert_end, :vlm_seq_len] = torch.where(
                vlm_visible_to_action_expert_mask,
                torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(torch.finfo(dtype).min / 2, device=device, dtype=dtype)
            )

        # 4.（可选）屏蔽 Action 段中�?PAD（仅�?Query 屏蔽�?
        # 说明：在 causal 情况�?pad 统一在尾部时，非 pad �?Query 本就无法看到更后�?pad Key�?
        # 因此仅做行屏蔽即可，避免无效 Query 参与计算；列屏蔽可省略�?
        if action_seq_len > 0 and action_attention_mask_1d is not None:
            act_valid = action_attention_mask_1d.bool()
            row_valid = act_valid.unsqueeze(1).unsqueeze(-1)  # [B,1,A_S,1]
            combined_mask[:, :, action_expert_start:action_expert_end, :] = torch.where(
                row_valid,
                combined_mask[:, :, action_expert_start:action_expert_end, :],
                torch.tensor(torch.finfo(dtype).min / 2, device=device, dtype=dtype)
            )

        return combined_mask

    def get_input_embeddings(self):
        return self.vlm.model.embed_tokens

    def set_input_embeddings(self, value):
        self.vlm.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.vlm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.vlm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.vlm.model = decoder

    def get_decoder(self):
        return self.vlm.model


# ============================================================================
# Emu3QFormer Model - Q-Former style Action Expert Integration
# ============================================================================

class ActionDecoder(nn.Module):
    """
    Simple decoder to project action expert outputs to action space.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class Emu3QFormer(Emu3PreTrainedModel):
    """
    Emu3QFormer Model combining Emu3MoE (VLM) and a Q-Former-style Action Expert.

    This model uses learnable queries as input to the action expert and interacts
    with the VLM using the same shared attention mechanism as Emu3Pi0.
    """
    _tied_weights_keys = ["vlm.lm_head.weight"]

    def __init__(self, config, pretrain_vlm_path):
        super().__init__(config)

        # Store configs for different components
        self.vlm_config = getattr(config, 'vlm_config', config)
        self.action_config = getattr(config, 'action_config', config)

        # VLM and Action Expert
        self.vlm, loading_info = Emu3MoE.from_pretrained(
            pretrain_vlm_path,
            attn_implementation="sdpa",
            torch_dtype=self.config.torch_dtype,
            output_loading_info=True
        )
        print("Missing keys:", loading_info["missing_keys"])
        print("Unexpected keys:", loading_info["unexpected_keys"])
        print("Mismatched sizes:", loading_info.get("mismatched_keys", "N/A"))

        self.action_expert = Emu3Model(self.action_config)

        # Action specific components
        action_dim = getattr(config, 'action_dim', 3)
        action_hidden_size = self.action_config.hidden_size
        self.action_frames = getattr(config, 'action_frames', 8)
        self.pre_action_frames = getattr(self.action_config, 'pre_action_frames', 3)

        # Learnable queries for the action expert, similar to Q-Former
        # self.action_queries = nn.Parameter(torch.zeros(1, self.action_frames, action_hidden_size))
        self.action_queries = nn.Parameter(torch.zeros(1, 256, action_hidden_size))
        nn.init.normal_(self.action_queries, std=action_hidden_size ** -0.5)

        # State projector for pre_action and cmd
        pre_action_dim = getattr(config, 'action_dim', 3)
        state_input_dim = self.pre_action_frames * pre_action_dim + 4  # 4 for cmd one-hot
        self.state_projector = nn.Sequential(
            nn.Linear(state_input_dim, action_hidden_size),
            nn.SiLU(),
            nn.Linear(action_hidden_size, action_hidden_size),
        )

        # Action decoder
        self.action_decoder = ActionDecoder(action_hidden_size, action_dim)

        # Create shared layer modules for gradient checkpointing
        self.shared_layers = [
            Emu3Pi0SharedLayer(vlm_layer, action_layer)
            for vlm_layer, action_layer in zip(self.vlm.model.layers, self.action_expert.layers)
        ]

        # Training configurations
        self.train_action_only = getattr(config, 'train_action_only', True)
        self.action_loss_weight = getattr(config, 'action_loss_weight', 1.0)

        # Denorm stats - 从环境变量或配置中获取路径
        norm_stats_path = getattr(config, 'norm_stats_path', None)
        if norm_stats_path is None:
            norm_stats_path = os.environ.get('VLA_NORM_STATS', '/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/configs/normalizer_navsim_trainval/norm_stats.json')
        
        with open(norm_stats_path, "r") as f:
            cfg = json.load(f)
        self._action_low = torch.tensor(cfg["norm_stats"]["libero"]["q01"], dtype=torch.float32)
        self._action_high = torch.tensor(cfg["norm_stats"]["libero"]["q99"], dtype=torch.float32)

        # Anchor classification support - 从环境变量或配置中获取路径
        anchor_cluster_path = getattr(config, 'anchor_cluster_path', None)
        if anchor_cluster_path is None:
            anchor_cluster_path = os.environ.get('VLA_ANCHOR_CLUSTER_PATH', '/mnt/vdb1/yingyan.li/emu_vla_logs/cluster_centers_8192.npy')
        self._anchor_cluster_path = anchor_cluster_path
        
        anchor_metric_score_path = getattr(config, 'anchor_metric_score_path', None)
        if anchor_metric_score_path is None:
            anchor_metric_score_path = os.environ.get('VLA_ANCHOR_METRIC_SCORE_PATH', '/mnt/vdb1/yingyan.li/emu_vla_logs/formatted_pdm_score_8192.npy')
        self._anchor_metric_score_path = anchor_metric_score_path
        self._anchor_metric_keys = [
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
        ]
        self._anchor_reward_weight = torch.tensor([0.1, 1.0, 1.0, 1.0], dtype=torch.float32)
        self._anchor_softmax_temperature = 1.0
        self._anchor_metric_score_dict = None
        self._default_metric_scores = None
        self._anchor_setup_done = False
        self._anchor_enabled = True
        self.anchor_num_trajs = None
        self.anchor_head = None
        self.register_buffer("anchor_cluster_centers", None, persistent=False)
        self.anchor = getattr(config, "anchor", True)
        if self.anchor:
            self._ensure_anchor_setup()

        # Gradient checkpointing configuration
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _ensure_anchor_setup(self):
        if self._anchor_setup_done:
            return
        reference_weight = self.action_expert.layers[0].mlp.up_proj.weight
        target_device = reference_weight.device
        target_dtype = reference_weight.dtype
        cluster_np = np.load(self._anchor_cluster_path)  # [8192, 8, 3]
        cluster_tensor = torch.tensor(cluster_np, dtype=torch.float32)  # [8192, 8, 3]
        self.anchor_cluster_centers = cluster_tensor  # [8192, 8, 3]
        self.anchor_num_trajs = cluster_tensor.shape[0]
        self.anchor_policy_head = nn.Sequential(
            nn.Linear(self.action_config.hidden_size, self.action_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.action_config.hidden_size, 1),
        )

        pos_encode_dim_per_coord = 2 * 10
        cluster_centers_feature_dim = 8 * 3 * pos_encode_dim_per_coord
        self.anchor_mlp_planning = nn.Sequential(
            nn.Linear(cluster_centers_feature_dim, self.action_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.action_config.hidden_size, self.action_config.hidden_size),
        )

        anchor_tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.action_config.hidden_size,
            nhead=8,
            dim_feedforward=self.action_config.hidden_size,
            dropout=0.0,
            batch_first=True,
        )
        self.anchor_tf_decoder = nn.TransformerDecoder(anchor_tf_decoder_layer, 3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.action_config.hidden_size,
            nhead=8,
            dim_feedforward=self.action_config.hidden_size,
            dropout=0.0,
            batch_first=True,
        )
        self.anchor_cluster_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        metric_raw = np.load(self._anchor_metric_score_path, allow_pickle=True).item()
        processed = {}
        for key, value in metric_raw.items():
            traj_scores = value["trajectory_scores"][0]
            stacked = np.vstack([traj_scores[k] for k in self._anchor_metric_keys]).astype(np.float32)
            processed[key] = torch.tensor(stacked, dtype=torch.float32)
        self._anchor_metric_score_dict = processed

        self._anchor_setup_done = True

    def _anchor_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Positional encoding for anchors; x: [num_trajs, num_steps, 3] -> [num_trajs, encoded_dim]."""
        device = x.device
        dtype = x.dtype
        num_trajs, num_steps, channels = x.shape
        assert channels == 3, f"Expected 3 channels (x,y,yaw), got {channels}"
        L = int(getattr(self, 'anchor_pos_encode_multires', 10))
        x_flat = x.reshape(num_trajs, -1)
        freqs = (2.0 ** torch.arange(L, device=device, dtype=dtype)) * np.pi
        x_flat = x_flat.unsqueeze(-1)
        freqs = freqs.reshape(1, 1, L)
        scaled = x_flat * freqs
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)
        enc = enc.reshape(num_trajs, -1)
        return enc

    def _get_cluster_centers_feat(self, batch_size: int) -> tuple[torch.Tensor, int]:
        """Encode anchor trajectories to hidden features, returns ([B, N, H], N)."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        traj_data = self.anchor_cluster_centers.to(device).to(dtype)
        encoded = self._anchor_positional_encoding(traj_data)
        init_traj = encoded.unsqueeze(0).repeat(batch_size, 1, 1)
        cluster_feat = self.anchor_mlp_planning(init_traj)
        cluster_feat = self.anchor_cluster_encoder(cluster_feat)
        return cluster_feat, traj_data.shape[0]

    def _gather_metric_scores(self, token, device, dtype):
        if self._anchor_metric_score_dict is None:
            raise RuntimeError("Anchor metric score dictionary is not loaded.")
        tokens_hex = [bytes(row.tolist()).hex() for row in token]
        metric_tensors = []
        for tok in tokens_hex:
            tensor = self._anchor_metric_score_dict.get(tok)
            if tensor is None:
                raise ValueError(f"Missing metric scores for token {tok}.")
            metric_tensors.append(tensor)
        stacked = torch.stack([tensor.to(device=device, dtype=dtype) for tensor in metric_tensors], dim=0)
        return stacked

    def _compute_anchor_loss(self, action, token, logits):
        if action is None:
            raise ValueError("Anchor classification requires ground-truth actions for supervision.")
        eps = 1e-6
        device = logits.device
        dtype = logits.dtype
        working_dtype = torch.float32

        action_tensor = action.to(device=device, dtype=working_dtype)
        low = self._action_low.to(device, dtype=dtype)
        high = self._action_high.to(device, dtype=dtype)
        action_tensor = 0.5 * (action_tensor + 1.0) * (high - low) + low
        abs_action = Emu3AutoRegressive_GRPO._relative_to_absolute_se2(action_tensor)

        metric_scores = self._gather_metric_scores(token, device=device, dtype=working_dtype)
        S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = metric_scores.unbind(dim=1)

        S_NC = S_NC.clamp(eps, 1.0)
        S_DAC = S_DAC.clamp(eps, 1.0)
        S_EP = S_EP.clamp(eps, 1.0)
        S_TTC = S_TTC.clamp(eps, 1.0)
        S_COMFORT = S_COMFORT.clamp(eps, 1.0)

        cluster = self.anchor_cluster_centers.to(device=device, dtype=working_dtype)
        B = abs_action.shape[0]
        cluster_flat = cluster.view(cluster.shape[0], -1).unsqueeze(0).expand(B, -1, -1)
        target_traj = abs_action.view(B, -1).unsqueeze(1)
        l2_distances = torch.cdist(cluster_flat, target_traj, p=2).squeeze(-1)
        score_targets = torch.softmax(-l2_distances, dim=-1)

        reward_weights = self._anchor_reward_weight.to(device=device, dtype=working_dtype)
        combined_term = 5 * S_TTC + 2 * S_COMFORT + 5 * S_EP
        gt_final_scores = (
                reward_weights[0] * torch.log(score_targets.clamp(min=eps))
                + reward_weights[1] * torch.log(S_NC)
                + reward_weights[2] * torch.log(S_DAC)
                + reward_weights[3] * torch.log(combined_term.clamp(min=eps))
        )
        gt_final_probs = torch.softmax(gt_final_scores / self._anchor_softmax_temperature, dim=-1)

        pred_probs = torch.softmax(logits, dim=-1).to(working_dtype)
        gt_final_probs = gt_final_probs.to(working_dtype)

        # reverse_kl = torch.sum(
        #     pred_probs * (torch.log(pred_probs + eps) - torch.log(gt_final_probs + eps)),
        #     dim=-1,
        # )
        #
        # return reverse_kl.mean().to(dtype)

        reverse_kl = torch.sum(
            pred_probs * (torch.log(pred_probs + eps) - torch.log(gt_final_probs + eps)),
            dim=-1,
        )
        forward_kl = torch.sum(
            gt_final_probs * (torch.log(gt_final_probs + eps) - torch.log(pred_probs + eps)),
            dim=-1,
        )
        sym_kl = 0.5 * (forward_kl + reverse_kl)
        return sym_kl.mean().to(dtype)

        # forward_kl = torch.sum(
        #     gt_final_probs * (torch.log(gt_final_probs + eps) - torch.log(pred_probs + eps)),
        #     dim=-1,
        # )
        #
        # return forward_kl.mean().to(dtype)

    @property
    def anchor(self):
        return self._anchor_enabled

    @anchor.setter
    def anchor(self, value):
        enabled = bool(value)
        if enabled and not self._anchor_setup_done:
            self._ensure_anchor_setup()
        self._anchor_enabled = enabled

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            action: torch.Tensor,
            pre_action: torch.Tensor,
            cmd: torch.Tensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            token=None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Forward pass combining Emu3MoE (VLM) and Action Expert using pi0-style shared attention.
        This method is primarily designed for training.

        Args:
            action (`torch.Tensor` of shape `(batch_size, action_frames, action_dim)`):
                Ground truth action sequence for training. The model will compute L1 loss against the predicted action.

        Returns:
            CausalLMOutputWithPast or tuple: The model's output.
        """
        use_cache = False
        return_dict = return_dict if return_dict is not None else self.vlm_config.use_return_dict

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Initial VLM embeddings
        if inputs_embeds is None:
            vlm_initial_hidden_states = self.vlm.model.embed_tokens(input_ids)
        else:
            vlm_initial_hidden_states = inputs_embeds

        vlm_seq_len = vlm_initial_hidden_states.shape[1]

        # Initial action expert embeddings from learnable queries
        # Process state (pre_action + cmd) into a state token
        state_input = torch.cat([pre_action.view(batch_size, -1), cmd], dim=1)
        state_token_embedding = self.state_projector(state_input).unsqueeze(1)  # (bs, 1, h)

        action_queries_expanded = self.action_queries.expand(batch_size, -1, -1)
        action_initial_hidden_states = torch.cat([state_token_embedding, action_queries_expanded], dim=1)
        action_seq_len = action_initial_hidden_states.shape[1]

        current_vlm_h = vlm_initial_hidden_states
        current_action_h = action_initial_hidden_states

        num_layers = len(self.vlm.model.layers)

        vlm_position_ids_original = position_ids
        if vlm_position_ids_original is None:
            vlm_position_ids_original = torch.arange(vlm_seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(
                batch_size, -1)

        combined_attention_mask_4d = Emu3Pi0.create_causal_style_attention_mask(
            vlm_seq_len, action_seq_len, attention_mask,
            input_ids,
            batch_size, device, current_vlm_h.dtype
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        for layer_idx in range(num_layers):
            shared_layer = self.shared_layers[layer_idx]
            if self.gradient_checkpointing and self.training:
                current_vlm_h, current_action_h = self._gradient_checkpointing_func(
                    shared_layer.__call__,
                    current_vlm_h, current_action_h, vlm_position_ids_original,
                    combined_attention_mask_4d, vlm_seq_len, action_seq_len, batch_size
                )
            else:
                current_vlm_h, current_action_h = shared_layer(
                    current_vlm_h, current_action_h, vlm_position_ids_original,
                    combined_attention_mask_4d, vlm_seq_len, action_seq_len, batch_size
                )

        final_action_hidden_for_decode = self.action_expert.norm(current_action_h)  # [bs, 256, h]
        action_hidden = final_action_hidden_for_decode[:, 1:, :]

        if self.anchor:
            cluster_centers_feat, num_traj = self._get_cluster_centers_feat(batch_size)
            keyval = final_action_hidden_for_decode
            query_out = self.anchor_tf_decoder(cluster_centers_feat, keyval)
            trajectory_query = query_out
            logits = self.anchor_policy_head(trajectory_query).squeeze(-1)  # [B, num_traj]
            total_loss = self._compute_anchor_loss(action, token, logits)
        else:
            predicted_action = self.action_decoder(action_hidden)
            trajectory_loss = F.l1_loss(predicted_action, action)
            total_loss = self.action_loss_weight * trajectory_loss
            logits = None

        if not return_dict:
            output = (logits, None, None, None)
            return (total_loss,) + output if total_loss is not None else output

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def sample_actions(
            self,
            input_ids: torch.LongTensor,
            pre_action: torch.Tensor,
            cmd: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            action_frames: Optional[int] = None,
            action_dim: Optional[int] = None,
    ) -> torch.Tensor:
        r"""
        Sample actions in a single forward pass using the Q-Former-style action expert.

        Args:
            input_ids: Input token ids for the VLM.
            pre_action: Past action sequence, used for state.
            cmd: Command one-hot vector, used for state.
            attention_mask: Attention mask for input_ids.
            position_ids: Position ids for input_ids.
            inputs_embeds: Optional pre-computed VLM embeddings.
            action_frames: Number of action frames to generate.
            action_dim: Dimension of the action.

        Returns:
            torch.Tensor: Generated actions of shape [batch_size, action_frames, action_dim]
        """
        self.eval()

        _action_frames = action_frames if action_frames is not None else self.action_frames
        _action_dim = action_dim if action_dim is not None else getattr(self.action_config, 'action_dim',
                                                                        getattr(self.config, 'action_dim', 7))

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Initial VLM embeddings
        if inputs_embeds is None:
            vlm_initial_hidden_states = self.vlm.model.embed_tokens(input_ids)
        else:
            vlm_initial_hidden_states = inputs_embeds

        vlm_seq_len = vlm_initial_hidden_states.shape[1]

        # Initial action expert embeddings from learnable queries
        state_input = torch.cat([pre_action.view(batch_size, -1), cmd], dim=1)
        state_token_embedding = self.state_projector(state_input).unsqueeze(1)

        action_queries_expanded = self.action_queries.expand(batch_size, -1, -1)
        action_initial_hidden_states = torch.cat([state_token_embedding, action_queries_expanded], dim=1)
        action_seq_len = action_initial_hidden_states.shape[1]

        current_vlm_h = vlm_initial_hidden_states
        current_action_h = action_initial_hidden_states

        num_layers = len(self.vlm.model.layers)

        vlm_position_ids_original = position_ids
        if vlm_position_ids_original is None:
            vlm_position_ids_original = torch.arange(vlm_seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(
                batch_size, -1)

        combined_attention_mask_4d = Emu3Pi0.create_causal_style_attention_mask(
            vlm_seq_len, action_seq_len, attention_mask,
            input_ids,
            batch_size, device, current_vlm_h.dtype
        )

        for layer_idx in range(num_layers):
            shared_layer = self.shared_layers[layer_idx]
            current_vlm_h, current_action_h = shared_layer(
                current_vlm_h, current_action_h, vlm_position_ids_original,
                combined_attention_mask_4d, vlm_seq_len, action_seq_len, batch_size
            )

        final_action_hidden_for_decode = self.action_expert.norm(current_action_h)
        action_hidden = final_action_hidden_for_decode[:, 1:, :]

        if self.anchor:
            cluster_centers_feat, num_traj = self._get_cluster_centers_feat(batch_size)
            keyval = final_action_hidden_for_decode
            query_out = self.anchor_tf_decoder(cluster_centers_feat, keyval)
            trajectory_query = query_out
            logits = self.anchor_policy_head(trajectory_query).squeeze(-1)
            probs = torch.softmax(logits, dim=-1)  # [bs ,8192]
            top_indices = torch.argmax(probs, dim=-1)  # [bs]
            cluster = self.anchor_cluster_centers.to(device=device, dtype=action_hidden.dtype)  # [8192, 8, 3]
            predicted_action = cluster[top_indices]
        else:
            predicted_action = self.action_decoder(action_hidden)

        return predicted_action

    def get_input_embeddings(self):
        return self.vlm.model.embed_tokens

    def set_input_embeddings(self, value):
        self.vlm.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.vlm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.vlm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.vlm.model = decoder

    def get_decoder(self):
        return self.vlm.model


class Emu3AutoRegressive(Emu3PreTrainedModel):
    """
    Emu3AutoRegressive：结�?Emu3(VLM) 与自回归 Action Expert；兼�?HF `.generate()`�?
    训练时可全参联合优化；推理时使用 VLM-KV 快速缓存�?
    """
    _tied_weights_keys = [
        "vlm.model.embed_tokens.weight",
        "action_expert.embed_tokens.weight",
    ]

    def __init__(self, config, pretrain_vlm_path):
        super().__init__(config)

        # --- 组件 ---
        self.vlm_config = getattr(config, "vlm_config", config)
        self.action_config = getattr(config, "action_config", config)

        # VLM
        self.vlm, loading_info = Emu3MoE.from_pretrained(
            pretrain_vlm_path,
            attn_implementation="sdpa",
            torch_dtype=self.config.torch_dtype,
            output_loading_info=True,
        )
        print("Loading VLM for Emu3AutoRegressive...")
        print("Missing keys in VLM:", loading_info["missing_keys"])  # 可见即可
        print("Unexpected keys in VLM:", loading_info["unexpected_keys"])  # 可见即可
        print("Mismatched sizes in VLM:", loading_info.get("mismatched_keys", "N/A"))  # 可见即可

        # Action Expert（权重与 VLM 共享词嵌入）
        self.action_expert = Emu3Model(self.action_config)
        self.action_expert.embed_tokens = self.vlm.model.embed_tokens

        # LM 头（Action hidden -> vocab�?
        self.action_lm_head = nn.Linear(self.action_config.hidden_size, self.vlm_config.vocab_size, bias=False)

        # state projector：将 [pre_action, cmd] -> 1 �?state token
        pre_action_dim = getattr(config, "action_dim", 3)
        self.pre_action_frames = getattr(config, "pre_action_frames", 3)
        state_input_dim = self.pre_action_frames * pre_action_dim + 4  # 4 = cmd one-hot
        self.state_projector = nn.Sequential(
            nn.Linear(state_input_dim, self.action_config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.action_config.hidden_size, self.action_config.hidden_size),
        )

        # 投影：将 action_expert �?token-emb �?vlm_hidden -> action_hidden（若两者维度不同）
        self.action_embed_projector = nn.Sequential(
            nn.Linear(self.vlm_config.hidden_size, self.action_config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.action_config.hidden_size, self.action_config.hidden_size),
        )

        # 共享层封装，�?forward_with_cache 使用（保持你现有实现�?
        self.shared_layers = [
            Emu3Pi0SharedLayer(vlm_layer, action_layer)
            for vlm_layer, action_layer in zip(self.vlm.model.layers, self.action_expert.layers)
        ]

        # 训练配置 / 特殊 token
        self.action_loss_weight = getattr(config, "action_loss_weight", 1.0)
        self.boa_token_id = getattr(config, "boa_token_id", 151844)
        self.eoa_token_id = getattr(config, "eoa_token_id", 151845)

        # VLM loss weighting (mimic Emu3MoE): if config provides vision_loss_weight, enable weighting
        self.vlm_loss_weight = 0.0
        self.use_weight = True
        self.vision_loss_weight = 0.5

        # Gradient Checkpointing 开关（默认关闭�?
        self.gradient_checkpointing = False

        self.post_init()

    # ---------------- HF generate 需要的接口 ----------------
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[dict] = None, **kwargs):
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        super().gradient_checkpointing_disable()
        self.gradient_checkpointing = False

    @add_start_docstrings_to_model_forward("Emu3AutoRegressive forward")
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="Emu3Config")
    def forward(
            self,
            input_ids: torch.LongTensor = None,  # 兼容旧参名：Action �?tokens（从 <boa> 开始）
            action_input_ids: torch.LongTensor = None,  # 新参名：Action �?tokens（从 <boa> 开始）
            attention_mask: Optional[torch.Tensor] = None,  # (unused)
            position_ids: Optional[torch.LongTensor] = None,  # (unused)
            inputs_embeds: Optional[torch.FloatTensor] = None,  # (unused)
            labels: Optional[torch.LongTensor] = None,  # 兼容旧参名：teacher-forcing 训练标签
            action_labels: Optional[torch.LongTensor] = None,  # 新参名：teacher-forcing 训练标签
            cache_position: Optional[torch.LongTensor] = None,  # 兼容 HF generate 传入
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # 生成/训练都需要的 VLM 上下�?
            vlm_input_ids: Optional[torch.LongTensor] = None,
            vlm_attention_mask: Optional[torch.Tensor] = None,  # 2D padding mask
            vlm_position_ids: Optional[torch.LongTensor] = None,
            vlm_labels: Optional[torch.LongTensor] = None,
            pre_action: Optional[torch.Tensor] = None,
            cmd: Optional[torch.Tensor] = None,
            # 可选：外部已算好的缓存
            vlm_k_rope_cache: Optional[List[torch.Tensor]] = None,
            vlm_v_cache: Optional[List[torch.Tensor]] = None,
            vlm_seq_len: Optional[int] = None,
            token=None,
            grpo_sample=False,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Forward pass combining Emu3MoE (VLM) and Action Expert using pi0-style shared attention.
        This method is primarily designed for training.

        Args:
            action (`torch.Tensor` of shape `(batch_size, action_frames, action_dim)`):
                Ground truth action sequence for training. The model will compute L1 loss against the predicted action.

        Returns:
            CausalLMOutputWithPast or tuple: The model's output.
        """
        # 兼容参数别名
        if action_input_ids is not None:
            input_ids = action_input_ids
        if action_labels is not None:
            labels = action_labels

        return_dict = True if return_dict is None else return_dict

        # Training path: do NOT use VLM KV precompute; run joint forward via shared layers
        if self.training and not grpo_sample:
            if input_ids is None:
                raise ValueError("action_input_ids is required (starts with <boa>) in training")
            if (pre_action is None) or (cmd is None):
                raise ValueError("pre_action and cmd must be provided to build the state token in training")
            if vlm_input_ids is None:
                raise ValueError("Training requires vlm_input_ids / vlm_attention_mask / pre_action / cmd")

            device = input_ids.device
            B = input_ids.shape[0]

            # Build state + action embeddings
            state_in = torch.cat([pre_action.view(B, -1), cmd], dim=1)
            state_tok = self.state_projector(state_in).unsqueeze(1)  # [B,1,H_a]

            action_emb = self.action_expert.embed_tokens(input_ids)  # [B,T,H_vlm]
            action_emb = self.action_embed_projector(action_emb)  # [B,T,H_a]
            current_action_h = torch.cat([state_tok, action_emb], dim=1)  # [B,1+T,H_a]
            action_len_with_state = current_action_h.size(1)

            # VLM initial hidden states
            current_vlm_h = self.vlm.model.embed_tokens(vlm_input_ids)  # [B,S,H_vlm]
            if vlm_position_ids is None:
                S = vlm_input_ids.shape[1]
                vlm_position_ids = torch.arange(S, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            else:
                S = vlm_position_ids.shape[1]

            # Valid action mask with state token
            pad_id = getattr(self.vlm_config, "pad_token_id", getattr(self.config, "pad_token_id", 0))
            action_valid_wo_state = (input_ids != pad_id)
            action_valid_with_state = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=device), action_valid_wo_state], dim=1)

            combined_attention_mask_4d = Emu3Pi0.create_causal_style_attention_mask(
                S,
                action_len_with_state,
                vlm_attention_mask,
                vlm_input_ids,
                B,
                device,
                current_action_h.dtype,
                action_is_causal=True,
                action_attention_mask_1d=action_valid_with_state,
            )

            # Layer-wise joint forward (QFormer-style bound method for ckpt)
            for shared_layer in self.shared_layers:
                if self.gradient_checkpointing and self.training:
                    current_vlm_h, current_action_h = self._gradient_checkpointing_func(
                        shared_layer.__call__,
                        current_vlm_h,
                        current_action_h,
                        vlm_position_ids,
                        combined_attention_mask_4d,
                        S,
                        action_len_with_state,
                        B,
                    )
                else:
                    current_vlm_h, current_action_h = shared_layer(
                        current_vlm_h,
                        current_action_h,
                        vlm_position_ids,
                        combined_attention_mask_4d,
                        S,
                        action_len_with_state,
                        B,
                    )

            # logits & loss (ignore state token)
            h_final = self.action_expert.norm(current_action_h)
            logits = self.action_lm_head(h_final)
            loss = None
            if labels is not None:
                # Next-token objective via shifting logits; include state-><boa>
                logits_for_loss = logits[:, :-1, :].contiguous()  # [B, T, V]
                targets = labels.contiguous()  # [B, T]
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits_for_loss.view(-1, logits_for_loss.size(-1)),
                    targets.view(-1)
                )
                loss = loss * getattr(self, "action_loss_weight", 1.0)

            if self.vlm_loss_weight > 0.0:
                # --- LM Head and Loss Calculation (mimic Emu3MoE) ---
                if self.vlm_config.pretraining_tp > 1:
                    lm_head_slices = self.vlm.lm_head.weight.split(
                        self.vlm_config.vocab_size // self.vlm_config.pretraining_tp, dim=0)
                    logits = [F.linear(current_vlm_h, lm_head_slices[i]) for i in range(self.vlm_config.pretraining_tp)]
                    logits = torch.cat(logits, dim=-1)
                else:
                    logits = self.vlm.lm_head(current_vlm_h)
                logits = logits.float()

                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = vlm_labels[..., 1:].contiguous()
                if self.use_weight:
                    weights = torch.ones(self.vlm_config.vocab_size, device=logits.device)
                    vision_token_range = range(self.vlm.bov_token_id, self.vlm.eov_token_id + 1)
                    weights[vision_token_range] = self.vision_loss_weight
                    loss_fct = CrossEntropyLoss(weight=weights)
                else:
                    loss_fct = CrossEntropyLoss()

                shift_logits = shift_logits.view(-1, self.vlm_config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                vlm_loss = loss_fct(shift_logits, shift_labels)
                loss = loss + self.vlm_loss_weight * vlm_loss

            if not return_dict:
                return logits, None, None, loss
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

        # 如果不是training阶段的话走下�?
        else:
            if input_ids is None:
                raise ValueError("action_input_ids（Action tokens，从 <boa> 开始）不能为空")
            if pre_action is None or cmd is None:
                raise ValueError("需要提�?pre_action �?cmd 以构�?state token")
            if (vlm_k_rope_cache is None) or (vlm_v_cache is None) or (vlm_seq_len is None):
                if vlm_input_ids is None:
                    raise ValueError("缺少 VLM 上下文：请提�?vlm_input_ids 及其 mask")
                # 训练态：带梯度；评估/推理：无梯度
                no_grad = not self.training
                vlm_k_rope_cache, vlm_v_cache, vlm_seq_len = self._precompute_vlm_kv(
                    vlm_input_ids=vlm_input_ids,
                    vlm_attention_mask=vlm_attention_mask,
                    vlm_position_ids=vlm_position_ids,
                    no_grad=no_grad,
                )

            device = input_ids.device
            B = input_ids.shape[0]

            # --- 构�?state + action 嵌入 ---
            state_in = torch.cat([pre_action.view(B, -1), cmd], dim=1)
            state_tok = self.state_projector(state_in).unsqueeze(1)  # [B,1,H_a]

            action_emb = self.action_expert.embed_tokens(input_ids)  # [B,T,H_vlm]
            action_emb = self.action_embed_projector(action_emb)  # [B,T,H_a]
            h = torch.cat([state_tok, action_emb], dim=1)  # [B,1+T,H_a]
            action_len_with_state = h.size(1)

            # --- 构造联合注意力掩码 ---
            # �?Action 段构�?1D 有效位掩码（包含 state=1 �?�?PAD �?action token�?
            pad_id = getattr(self.vlm_config, "pad_token_id", getattr(self.config, "pad_token_id", None))
            if pad_id is None:
                pad_id = 0
            action_valid_wo_state = (input_ids != pad_id)
            action_valid_with_state = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=device), action_valid_wo_state], dim=1)

            combined_attention_mask_4d = Emu3Pi0.create_causal_style_attention_mask(
                vlm_seq_len,
                action_len_with_state,
                vlm_attention_mask,
                vlm_input_ids,
                B,
                device,
                h.dtype,
                action_is_causal=True,
                action_attention_mask_1d=action_valid_with_state,
            )
            action_attention_mask = combined_attention_mask_4d[:, :, vlm_seq_len:, :]

            # --- 过共享层：Action-Q attends [VLM-KV(cache), Action-KV] ---
            for li, layer in enumerate(self.shared_layers):
                h = layer.forward_with_cache(
                    h,
                    action_attention_mask=action_attention_mask,
                    vlm_seq_len=vlm_seq_len,
                    action_seq_len=action_len_with_state,
                    batch_size=B,
                    vlm_k_rope_cached=vlm_k_rope_cache[li],
                    vlm_v_cached=vlm_v_cache[li],
                    action_query_valid_1d=action_valid_with_state,
                )

            # --- logits & loss ---
            h = self.action_expert.norm(h)
            logits = self.action_lm_head(h)  # [B,1+T,V]

            loss = None
            if labels is not None:
                # 典型 CausalLM：丢�?state 位置�?logits，与 labels 对齐
                # 要求 labels 形状�?[B, T]（与 input_ids 对齐），可在需要忽略的位置�?-100
                logits_for_loss = logits[:, :-1, :].contiguous()  # [B,T,V]
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits_for_loss.view(-1, logits_for_loss.size(-1)), labels.view(-1)) * getattr(self,
                                                                                                               "action_loss_weight",
                                                                                                               1.0)

            if not return_dict:
                return logits, None, None, loss
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    # ---- HF .generate() 会先调这个方法打包额外参数（只在首步构建 VLM 缓存�?---
    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        # 兼容 action_input_ids 作为 generate() 的输入名
        if kwargs.get("action_input_ids", None) is not None and input_ids is None:
            input_ids = kwargs.pop("action_input_ids")
        # 首步：若没附带缓存，则用 no_grad �?fast-path 预计算（推理专用�?
        if kwargs.get("vlm_k_rope_cache", None) is None or kwargs.get("vlm_v_cache", None) is None:
            vlm_input_ids = kwargs.get("vlm_input_ids", None)
            if vlm_input_ids is None:
                raise ValueError("generate() 需要传�?vlm_input_ids / vlm_attention_mask / pre_action / cmd")
            k_cache, v_cache, vlm_seq_len = self._precompute_vlm_kv(
                vlm_input_ids=vlm_input_ids,
                vlm_attention_mask=kwargs.get("vlm_attention_mask", None),
                vlm_position_ids=kwargs.get("vlm_position_ids", None),
                no_grad=True,
            )
            kwargs["vlm_k_rope_cache"] = k_cache
            kwargs["vlm_v_cache"] = v_cache
            kwargs["vlm_seq_len"] = vlm_seq_len
        # 我们不使�?past_key_values（Action 侧每步重算，已足够快�?
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            **kwargs,
        }

    def freeze_vlm(self):
        """Freeze VLM parameters; only train Action Expert and related heads."""
        for param in self.vlm.parameters():
            param.requires_grad = False

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 未使�?past；为兼容 BeamSearch 接口，返回原�?
        return past_key_values

    # ---- 便捷封装：直接生成动作序列（逐样�?EOS 停止�?---
    @torch.no_grad()
    def generate_actions(
            self,
            vlm_input_ids: torch.LongTensor,
            vlm_attention_mask: Optional[torch.Tensor] = None,
            vlm_position_ids: Optional[torch.LongTensor] = None,
            pre_action: Optional[torch.Tensor] = None,
            cmd: Optional[torch.Tensor] = None,
            max_new_tokens: int = 10,
            do_sample: bool = True,
            top_p: float = 1.0,
            temperature: float = 1.0,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            logits_processor: Optional[list] = None,
            return_scores: bool = False,
            vlm_k_rope_cache: Optional[List[torch.Tensor]] = None,
            vlm_v_cache: Optional[List[torch.Tensor]] = None,
            vlm_seq_len: Optional[int] = None,
    ) -> Union[torch.LongTensor, "GenerateDecoderOnlyOutput"]:
        device = vlm_input_ids.device
        B = vlm_input_ids.size(0)
        bos = torch.full((B, 1), self.boa_token_id, dtype=torch.long, device=device)

        # Fast-path：无梯度缓存（可复用传入缓存�?
        if vlm_k_rope_cache is None or vlm_v_cache is None or vlm_seq_len is None:
            k_cache, v_cache, vlm_seq_len = self._precompute_vlm_kv(
                vlm_input_ids=vlm_input_ids,
                vlm_attention_mask=vlm_attention_mask,
                vlm_position_ids=vlm_position_ids,
                no_grad=True,
            )
        else:
            k_cache, v_cache = vlm_k_rope_cache, vlm_v_cache

        bos_attention_mask = torch.ones_like(bos, dtype=torch.long, device=device)

        gen_res = self.generate(
            input_ids=bos,
            attention_mask=bos_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=max(temperature, 1e-5) if do_sample else 1.0,
            eos_token_id=self.eoa_token_id,
            pad_token_id=self.config.pad_token_id,
            logits_processor=logits_processor,
            return_dict_in_generate=return_scores,
            output_scores=return_scores,
            vlm_input_ids=vlm_input_ids,
            vlm_attention_mask=vlm_attention_mask,
            vlm_position_ids=vlm_position_ids,
            pre_action=pre_action,
            cmd=cmd,
            vlm_k_rope_cache=k_cache,
            vlm_v_cache=v_cache,
            vlm_seq_len=vlm_seq_len,
            grpo_sample=True
        )

        return gen_res

    # ---------------- 内部：VLM KV 预计算（可控是否带梯�?& 可�?checkpoint�?----------------
    def _precompute_vlm_kv(
            self,
            vlm_input_ids: torch.LongTensor,
            vlm_attention_mask: Optional[torch.Tensor] = None,
            vlm_position_ids: Optional[torch.LongTensor] = None,
            no_grad: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """
        返回�?
            vlm_k_rope_cache, vlm_v_cache, vlm_seq_len
            形状：list[num_layers]，每个元�?[B, n_kv_heads, S_vlm, head_dim]
        """
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            device = vlm_input_ids.device
            B, S = vlm_input_ids.shape
            # 词嵌�?
            h = self.vlm.model.embed_tokens(vlm_input_ids)  # [B,S,E]

            if vlm_position_ids is None:
                vlm_position_ids = torch.arange(S, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

            attn_4d = _prepare_4d_causal_attention_mask(vlm_attention_mask, (B, S), h, 0)

            vlm_k_list: List[torch.Tensor] = []
            vlm_v_list: List[torch.Tensor] = []

            # 逐层构建缓存；如在训练并开启了 GC，对 layer(h) 前向进行 checkpoint
            for layer in self.vlm.model.layers:
                normed = layer.input_layernorm(h)
                n_heads = layer.self_attn.num_heads
                kv_heads = layer.self_attn.num_key_value_heads
                head_dim = layer.self_attn.head_dim

                q = layer.self_attn.q_proj(normed).view(B, S, n_heads, head_dim).transpose(1, 2)
                k = layer.self_attn.k_proj(normed).view(B, S, kv_heads, head_dim).transpose(1, 2)
                v = layer.self_attn.v_proj(normed).view(B, S, kv_heads, head_dim).transpose(1, 2)

                cos, sin = layer.self_attn.rotary_emb(v, seq_len=S)
                from .modeling_emu3 import apply_rotary_pos_emb  # 与工程一�?
                _, k_rope = apply_rotary_pos_emb(q, k, cos, sin, vlm_position_ids, unsqueeze_dim=1)

                vlm_k_list.append(k_rope)
                vlm_v_list.append(v)

                # 前推到下一层输入；可�?checkpoint 仅作用于 layer(h) 以节约激�?
                if self.training and self.gradient_checkpointing and (not no_grad):
                    layer_outputs = self._gradient_checkpointing_func(layer.__call__, h, attn_4d, vlm_position_ids)
                    h = layer_outputs[0]
                else:
                    h = layer(h, attention_mask=attn_4d, position_ids=vlm_position_ids)[0]

        return vlm_k_list, vlm_v_list, S

    # ---- Embedding/Decoder API（与 HF 工具链兼容）----
    def get_input_embeddings(self):
        return self.vlm.model.embed_tokens

    def set_input_embeddings(self, value):
        self.vlm.model.embed_tokens = value
        self.action_expert.embed_tokens = value

    def get_output_embeddings(self):
        return self.action_lm_head

    def set_output_embeddings(self, new_embeddings):
        self.action_lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.vlm.model = decoder

    def get_decoder(self):
        return self.vlm.model
