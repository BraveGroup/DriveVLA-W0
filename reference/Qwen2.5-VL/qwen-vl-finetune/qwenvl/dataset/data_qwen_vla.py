# VLA adaptation of data_qwen.py - adds action processing capability
# Based on data_qwen.py with action tokenization support

import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence
from collections.abc import Sequence as SequenceType

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
# from torchcodec.decoders import VideoDecoder  # Commented out due to compatibility issues
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

# Import for action tokenizer
from transformers import AutoProcessor
import pickle
from qwenvl.utils.token_utils import prepare_action_tokenizer_mapping

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

# Import token utilities instead of duplicating logic
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))


def preprocess_qwen_2_visual_vla_sources(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List[int] = [],
    action_segments: Optional[List[torch.Tensor]] = None,
    action_roles: Optional[List[str]] = None,
    return_masks: bool = True,
) -> Dict:
    """Preprocess multiple sources (each a dialogue) with a single action segment list.

    - grid_thw_image: flattened list[int], one entry per <image> occurrence across all sources/turns
    - action_segments: list of LongTensor; consumed sequentially when a turn requires action insertion
      (user turns: historical actions → no loss; assistant turns: future actions → loss on action tokens)
    """
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_image_idx = 0
    action_idx = 0
    action_list: List[torch.Tensor] = [] if action_segments is None else (
        action_segments if isinstance(action_segments, list) else [action_segments]
    )
    action_role_list: List[str] = [] if action_roles is None else (
        action_roles if isinstance(action_roles, list) else [action_roles]
    )

    input_ids, targets = [], []
    
    img_token_id = vs_id = ve_id = None
    if return_masks:
        try:
            img_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            vs_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        except Exception:
            img_token_id = vs_id = ve_id = None

    def expand_images(content: str) -> str:
        nonlocal visual_image_idx
        if "<image>" not in content:
            return content
        parts = content.split("<image>")
        new_parts = []
        for j in range(len(parts) - 1):
            new_parts.append(parts[j])
            replicate = grid_thw_image[visual_image_idx] if visual_image_idx < len(grid_thw_image) else 0
            replacement = (
                "<|vision_start|>" + ("<|image_pad|>" * replicate) + "<|vision_end|>"
            )
            new_parts.append(replacement)
            visual_image_idx += 1
        new_parts.append(parts[-1])
        return "".join(new_parts)
    
    for source in sources:
        cur_input, cur_target = [], []
        
        sys_tokens = tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant."}
        ])
        cur_input += sys_tokens
        cur_target += [IGNORE_INDEX] * len(sys_tokens)

        norm_source = []
        for turn in source:
            if "role" in turn and "content" in turn:
                role = turn["role"]
                content = turn["content"]
            else:
                role = turn.get("from", turn.get("role"))
                content = turn.get("value", turn.get("content", ""))
            role = roles.get(role, role)
            norm_source.append({"role": role, "content": content})

        for turn in norm_source:
            role = turn["role"]
            content = expand_images(turn["content"])

            templ = tokenizer.apply_chat_template([{"role": role, "content": content}])

            inserted_action: List[int] = []
            should_insert = (
                action_idx < len(action_list)
                and (len(action_role_list) == 0 or (action_idx < len(action_role_list) and action_role_list[action_idx] == role))
            )
            if should_insert:
                inserted_action = action_list[action_idx].tolist()
                action_idx += 1

                insert_pos = len(templ) - 2
                enc = templ[:insert_pos] + inserted_action + templ[insert_pos:]
                lbl = [IGNORE_INDEX] * len(enc)
                if role == "assistant":
                    for k in range(3, insert_pos):
                        lbl[k] = enc[k]
                    for k in range(insert_pos, insert_pos + len(inserted_action)):
                        lbl[k] = enc[k]
                cur_input += enc
                cur_target += lbl
            else:
                enc = templ
                if role in ["user", "system"]:
                    lbl = [IGNORE_INDEX] * len(enc)
                else:
                    lbl = enc.copy()
                    lbl[:3] = [IGNORE_INDEX] * 3
                cur_input += enc
                cur_target += lbl

        input_ids.append(cur_input)
        targets.append(cur_target)

    out = dict(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        labels=torch.tensor(targets, dtype=torch.long),
    )
    
    if return_masks:
        input_tensor = torch.tensor(input_ids[0], dtype=torch.long) if input_ids else None
        
        if input_tensor is not None:
            seq_len = len(input_tensor)
            
            image_masks_list = []
            action_masks_list = []
            
            if img_token_id is not None and vs_id is not None and ve_id is not None:
                vs_positions = torch.where(input_tensor == vs_id)[0]
                ve_positions = torch.where(input_tensor == ve_id)[0]
                
                for vs_pos in vs_positions:
                    ve_candidates = ve_positions[ve_positions > vs_pos]
                    if len(ve_candidates) > 0:
                        ve_pos = ve_candidates[0]
                        
                        img_mask = torch.zeros(seq_len, dtype=torch.bool)
                        img_mask[vs_pos+1:ve_pos] = True
                        
                        image_masks_list.append([img_mask])
            
            boa_id = tokenizer.encode(tokenizer.boa_token, add_special_tokens=False)[0]
            eoa_id = tokenizer.encode(tokenizer.eoa_token, add_special_tokens=False)[0]

            
            if boa_id is not None and eoa_id is not None:
                boa_positions = torch.where(input_tensor == boa_id)[0]
                eoa_positions = torch.where(input_tensor == eoa_id)[0]
                
                for boa_pos in boa_positions:
                    eoa_candidates = eoa_positions[eoa_positions > boa_pos]
                    if len(eoa_candidates) > 0:
                        eoa_pos = eoa_candidates[0]
                        
                        act_mask = torch.zeros(seq_len, dtype=torch.bool)
                        if eoa_pos > boa_pos + 1:
                            act_mask[boa_pos+1:eoa_pos] = True
                        action_masks_list.append(act_mask)
            
            if image_masks_list:
                T = len(image_masks_list)
                image_masks_tensor = torch.zeros((T, 1, seq_len), dtype=torch.bool)
                for t, masks in enumerate(image_masks_list):
                    if masks:
                        image_masks_tensor[t, 0] = masks[0]
                out["image_token_masks"] = image_masks_tensor.unsqueeze(0)
            
            if action_masks_list:
                action_masks_tensor = torch.stack(action_masks_list, dim=0)
                out["action_future_masks"] = action_masks_tensor.unsqueeze(0)
    
    return out


class LazySupervisedNavsim2VAROSSDataset(Dataset):
    """VLA Dataset for NavSim2 with pre 1s sequence support."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedNavsim2VAROSSDataset, self).__init__()

        # Check VLA tokens
        if not (hasattr(tokenizer, 'boa_token') and hasattr(tokenizer, 'eoa_token')):
            raise ValueError("Tokenizer missing BOA/EOA tokens. Call check_and_add_vla_tokens first.")

        # Initialize Action Tokenizer
        if getattr(data_args, "actions_format", "fast") == "fast":
            self.fast_path = getattr(data_args, "action_tokenizer_path", None)
            if self.fast_path:
                self.action_tokenizer = AutoProcessor.from_pretrained(self.fast_path, trust_remote_code=True)
            else:
                raise ValueError("action_tokenizer_path is required for fast actions format")
        else:
            raise ValueError(f"Unsupported actions_format: {getattr(data_args, 'actions_format', 'fast')}")
        
        # Action processing config
        self.use_actions = getattr(data_args, "use_actions", True)
        self.actions_format = getattr(data_args, "actions_format", "fast")
        
        # Get special token IDs
        mapping = prepare_action_tokenizer_mapping(tokenizer)
        self.boa_token_id = mapping["boa_token_id"]
        self.eoa_token_id = mapping["eoa_token_id"]
        self.last_vocab_idx = mapping["last_vocab_idx"]
        self.last_text_token_idx = mapping["last_vocab_idx"] - self.action_tokenizer.vocab_size
        
        rank0_print(f"VLA tokens: BOA={self.boa_token_id}, EOA={self.eoa_token_id}, last_vocab_idx={self.last_vocab_idx}")

        # Load data from pickle file
        data_path = getattr(data_args, "data_path", None)
        if data_path and data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
            rank0_print(f"Loaded {len(self.data)} samples from pickle file: {data_path}")
        else:
            raise ValueError("NavSim2 dataset requires pickle file")
        
        # Store additional config
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2
        
        # Driving specific config
        self.use_previous_actions = getattr(data_args, "use_previous_actions", False)
        self.cur_idx = getattr(data_args, "cur_frame_idx", 0)
        self.future_nums = getattr(data_args, "future_nums", 8)
        self.pre_future_nums = getattr(data_args, "pre_future_nums", 2)
        
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        # 不覆盖 size.longest_edge/shortest_edge，交由 min/max_pixels 控制自适应缩放

        # 新增可选功能开关
        self.return_raw_vae = getattr(data_args, "return_raw_vae", True)
        self.return_masks = getattr(data_args, "return_masks", True)  # 始终返回 masks

        if not self.data_args.data_root:
            raise ValueError("data_root is required for NavSim2 VLA dataset")
        
        
    def __len__(self):
        return len(self.data)

    @property
    def lengths(self):
        length_list = []
        for sample in self.data:
            img_tokens = 256 if "image" in sample and "pre_1s_image" in sample else 128  # Double for pre 1s
            action_tokens = 100 if self.use_actions else 0  # Estimate for pre + current
            try:
                text_tokens = 100  # Default estimate
                if "text" in sample:
                    if isinstance(sample["text"], list):
                        text_tokens = sum(len(str(t).split()) for t in sample["text"])
                    else:
                        text_tokens = len(str(sample["text"]).split())
                if "pre_1s_text" in sample:
                    if isinstance(sample["pre_1s_text"], list):
                        text_tokens += sum(len(str(t).split()) for t in sample["pre_1s_text"])
                    else:
                        text_tokens += len(str(sample["pre_1s_text"]).split())
                length_list.append(text_tokens + img_tokens + action_tokens)
            except:
                length_list.append(400)  # fallback
        return length_list

    def wrap_action_sequence(self, action_ids: List[int]) -> torch.Tensor:
        """Wrap action tokens with BOA/EOA"""
        wrapped = [self.boa_token_id] + action_ids + [self.eoa_token_id]
        return torch.tensor(wrapped, dtype=torch.long)

    def process_actions(self, actions: np.ndarray) -> List[int]:
        """Process continuous actions to token IDs"""
        if self.actions_format == "fast":
            if isinstance(actions, list):
                tensor_list = [torch.tensor(item).unsqueeze(0) for item in actions]
                action_tokens = torch.cat(tensor_list, dim=0)
            else:
                action_tokens = torch.tensor(actions) if isinstance(actions, np.ndarray) else actions
            
            action_ids = self.action_tokenizer(action_tokens)[0]
            mapped_action_ids = [self.last_vocab_idx - id for id in action_ids]
            
            return mapped_action_ids
        else:
            raise ValueError(f"Unsupported actions_format: {self.actions_format}")

    def load_image(self, scene, image_key, fallback_key=None):
        if image_key not in scene:
            return None, None
            
        image_paths = scene[image_key]
        image_file = None
        
        # 提取图像文件路径
        if isinstance(image_paths, str):
            image_file = image_paths
        elif isinstance(image_paths, list) and len(image_paths) > 0:
            image_file = image_paths[self.cur_idx] if self.cur_idx < len(image_paths) else image_paths[0]
        else:
            return None, None
            
        if not image_file:
            return None, None
            
        # 路径处理和图像加载
        image_file = self._resolve_image_path(image_file)
        
        # 调用原有的图像处理逻辑
        return self.process_image_unified(image_file)
    
    def _resolve_image_path(self, image_path):

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        # 检查是否为标准格式且文件存在
        if (image_path.lower().endswith(image_extensions) and 
            os.path.exists(image_path)):
            return image_path
        
        # 使用硬编码逻辑构造路径
        return self._construct_image_path(image_path)
    
    def _construct_image_path(self, original_path):
        # 分割路径获取后两位
        path_parts = original_path.split('/')
        
        if len(path_parts) >= 2:
            # 获取最后一个文件名并去掉.npy扩展名（如果存在）
            filename = path_parts[-1]
            if filename.endswith('.npy'):
                filename = filename[:-4]  # 去掉.npy
            
            # 获取最后两个路径组件
            last_two = os.path.join(path_parts[-2], "CAM_F0", filename)
        else:
            filename = path_parts[-1]
            if filename.endswith('.npy'):
                filename = filename[:-4]  # 去掉.npy
            last_two = filename
        
        # 构造实际路径
        data_root = self.data_args.data_root
        actual_path = os.path.join(data_root, last_two + '.jpg')
        
        if os.path.exists(actual_path):
            return actual_path
        
        # 尝试其他扩展名
        for ext in ['.jpeg', '.png']:
            alt_path = os.path.join(data_root, last_two + ext)
            if os.path.exists(alt_path):
                return alt_path
        
        # 如果都找不到，返回原始路径（让后续处理报错）
        print(f"Warning: Cannot resolve image path: {original_path}, trying original path")
        return original_path

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"] 
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        if self.return_raw_vae: 
            try:
                raw_vae_img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0 * 2 - 1  # [3, H, W], [-1, 1]
            except Exception:
                raw_vae_img_tensor = None
            return image_tensor, grid_thw, raw_vae_img_tensor
        return image_tensor, grid_thw

    # ---------- Reusable builders ----------
    def build_sources(self, pre_prompt: str, cur_prompt: str) -> List[List[Dict[str, str]]]:
        return [[
            {"from": "user", "value": f"<image>{pre_prompt}"},
            {"from": "assistant", "value": ""},
            {"from": "user", "value": f"<image>{cur_prompt}"},
            {"from": "assistant", "value": ""},
        ]]

    def flatten_grid_thw(self, pre_grid_thw_merged: Optional[List[int]], grid_thw_merged: Optional[List[int]]) -> List[int]:
        flat: List[int] = []
        if pre_grid_thw_merged:
            flat.extend(pre_grid_thw_merged)
        if grid_thw_merged:
            flat.extend(grid_thw_merged)
        return flat

    def build_action_segments(
        self,
        pre_user_action_tokens: Optional[torch.Tensor],
        pre_wrapped_action_tokens: Optional[torch.Tensor],
        user_action_tokens: Optional[torch.Tensor],
        wrapped_action_tokens: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[str]]:
        segs: List[torch.Tensor] = []
        roles: List[str] = []
        if self.use_previous_actions and pre_user_action_tokens is not None:
            segs.append(pre_user_action_tokens)
            roles.append("user")
        if pre_wrapped_action_tokens is not None:
            segs.append(pre_wrapped_action_tokens)
            roles.append("assistant")
        if self.use_previous_actions and user_action_tokens is not None:
            segs.append(user_action_tokens)
            roles.append("user")
        if wrapped_action_tokens is not None:
            segs.append(wrapped_action_tokens)
            roles.append("assistant")
        return segs, roles

    def compute_action_tokens(
        self,
        actions: Optional[np.ndarray],
        use_previous: bool,
        cur_idx: int,
        future_len: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (historical_tokens, future_wrapped_tokens) for a single stream.

        - historical_tokens: None if use_previous is False or no history
        - future_wrapped_tokens: None if no future available
        """
        if actions is None:
            return None, None
        hist_tokens = None
        fut_wrapped = None

        if len(actions) > cur_idx:
            if cur_idx > 0 and use_previous:
                hist_ids = self.process_actions(actions[:cur_idx])
                hist_tokens = torch.tensor(hist_ids, dtype=torch.long)

            fut_end = min(cur_idx + future_len, len(actions))
            fut_ids = self.process_actions(actions[cur_idx:fut_end])
            fut_wrapped = self.wrap_action_sequence(fut_ids)
        return hist_tokens, fut_wrapped

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # Try other samples
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.data) - 1)
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        scene = self.data[i]
        
        # Get text prompts
        if "text" in scene:
            if isinstance(scene["text"], list):
                prompt = scene["text"][self.cur_idx] if self.cur_idx < len(scene["text"]) else scene["text"][0]
            else:
                prompt = scene["text"]
        else:
            prompt = "Describe the driving scene."
            
        if "pre_1s_text" in scene:
            if isinstance(scene["pre_1s_text"], list):
                pre_prompt = scene["pre_1s_text"][self.cur_idx] if self.cur_idx < len(scene["pre_1s_text"]) else scene["pre_1s_text"][0]
            else:
                pre_prompt = scene["pre_1s_text"]
        else:
            pre_prompt = prompt
        
        # Process image
        # 当前帧/前1s图像与可选 raw_vae
        image_tensor = grid_thw = pre_image_tensor = pre_grid_thw = None
        image_raw = pre_image_raw = None
        out_cur = self.load_image(scene, "image")
        out_pre = self.load_image(scene, "pre_1s_image")
        if self.return_raw_vae:
            if out_cur is not None and isinstance(out_cur, tuple) and len(out_cur) == 3:
                image_tensor, grid_thw, image_raw = out_cur
            else:
                image_tensor, grid_thw = out_cur if isinstance(out_cur, tuple) else (None, None)
            if out_pre is not None and isinstance(out_pre, tuple) and len(out_pre) == 3:
                pre_image_tensor, pre_grid_thw, pre_image_raw = out_pre
            else:
                pre_image_tensor, pre_grid_thw = out_pre if isinstance(out_pre, tuple) else (None, None)
        else:
            image_tensor, grid_thw = out_cur if isinstance(out_cur, tuple) else (None, None)
            pre_image_tensor, pre_grid_thw = out_pre if isinstance(out_pre, tuple) else (None, None)

        image = [image_tensor] if image_tensor is not None else None
        pre_image = [pre_image_tensor] if pre_image_tensor is not None else None

        # Process actions via unified helper
        user_action_tokens = None
        wrapped_action_tokens = None
        pre_user_action_tokens = None
        pre_wrapped_action_tokens = None
        if self.use_actions:
            if "action" in scene:
                user_action_tokens, wrapped_action_tokens = self.compute_action_tokens(
                    actions=scene["action"],
                    use_previous=self.use_previous_actions,
                    cur_idx=self.cur_idx,
                    future_len=self.future_nums,
                )
            if "pre_1s_action" in scene:
                pre_user_action_tokens, pre_wrapped_action_tokens = self.compute_action_tokens(
                    actions=scene["pre_1s_action"],
                    use_previous=self.use_previous_actions,
                    cur_idx=self.cur_idx,
                    future_len=self.pre_future_nums,
                )

        # (legacy chat_sources/pre_chat_sources removed)

        # Process grid_thw for tokenization
        grid_thw_merged = None
        pre_grid_thw_merged = None
        
        if grid_thw is not None:
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, SequenceType):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            
        if pre_grid_thw is not None:
            pre_grid_thw_merged = copy.deepcopy(pre_grid_thw)
            if not isinstance(pre_grid_thw, SequenceType):
                pre_grid_thw_merged = [pre_grid_thw_merged]
                pre_grid_thw = [pre_grid_thw]
            pre_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in pre_grid_thw_merged
            ]

        # Build a single source: pre (user→assistant) then cur (user→assistant)
        sources = self.build_sources(pre_prompt, prompt)

        # Flattened grid list in the order of <image> appearances across sources
        grid_thw_image_flat: List[int] = self.flatten_grid_thw(pre_grid_thw_merged, grid_thw_merged)

        # Assemble action segments list and their target roles, in traversal order
        action_segments, action_roles = self.build_action_segments(
            pre_user_action_tokens, pre_wrapped_action_tokens, user_action_tokens, wrapped_action_tokens
        )

        data_dict = preprocess_qwen_2_visual_vla_sources(
            sources=sources,
            tokenizer=self.tokenizer,
            grid_thw_image=grid_thw_image_flat,
            action_segments=action_segments if len(action_segments) > 0 else None,
            action_roles=action_roles if len(action_roles) > 0 else None,
            # return_masks=self.return_masks,
            return_masks=True,
        )

        # Get position IDs
        all_grid_thw = []
        if pre_grid_thw:
            all_grid_thw.extend(pre_grid_thw)
        if grid_thw:
            all_grid_thw.extend(grid_thw)
            
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(all_grid_thw, dim=0) if all_grid_thw else None,
            video_grid_thw=None,
            second_per_grid_ts=None,
        )
        
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        # Add image tensors
        all_images = []
        all_image_grid_thw = []
        
        if pre_image and pre_grid_thw:
            all_images.extend(pre_image)
            all_image_grid_thw.extend([thw.unsqueeze(0) for thw in pre_grid_thw])
            
        if image and grid_thw:
            all_images.extend(image)
            all_image_grid_thw.extend([thw.unsqueeze(0) for thw in grid_thw])
            
        if all_images:
            data_dict["pixel_values"] = torch.cat(all_images, dim=0)
            data_dict["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
            if self.return_raw_vae:
                T, N = 2, 1
                
                raw_shape = image_raw.shape
                raw_tensor = torch.zeros((T, N) + raw_shape, dtype=data_dict["pixel_values"].dtype)
                if pre_image_raw is not None:
                    raw_tensor[0, 0] = pre_image_raw
                raw_tensor[1, 0] = image_raw

                data_dict["raw_pixel_values_vae"] = raw_tensor
                data_dict["frame_image_counts"] = torch.tensor([1, 1], dtype=torch.long)
        return data_dict


# ----------------------------- NuPlan 2VA Dataset -----------------------------
class LazySupervisedNuplan2VAROSSDataset(Dataset):
    """NuPlan VLA Dataset building a single sequence from pre(1s) and cur.

    Differences from NavSim2:
    - No explicit pre_1s fields; compute pre image/action via 1s offset on the same sequence
    - Exactly one pre+cur sequence per sample (not two repeated cur anchors)
    - Time windows:
        pre action: 1.5s history, 1s future
        cur action: 1.5s history, 4s future
    - Index sampling controlled by rng(seed)
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedNuplan2VAROSSDataset, self).__init__()

        # Check VLA tokens
        if not (hasattr(tokenizer, 'boa_token') and hasattr(tokenizer, 'eoa_token')):
            raise ValueError("Tokenizer missing BOA/EOA tokens. Call check_and_add_vla_tokens first.")

        # Initialize Action Tokenizer
        if getattr(data_args, "actions_format", "fast") == "fast":
            self.fast_path = getattr(data_args, "action_tokenizer_path", None)
            if self.fast_path:
                self.action_tokenizer = AutoProcessor.from_pretrained(self.fast_path, trust_remote_code=True)
            else:
                raise ValueError("action_tokenizer_path is required for fast actions format")
        else:
            raise ValueError(f"Unsupported actions_format: {getattr(data_args, 'actions_format', 'fast')}")

        # Action processing config
        self.use_actions = getattr(data_args, "use_actions", True)
        self.actions_format = getattr(data_args, "actions_format", "fast")

        # Get special token IDs
        mapping = prepare_action_tokenizer_mapping(tokenizer)
        self.boa_token_id = mapping["boa_token_id"]
        self.eoa_token_id = mapping["eoa_token_id"]
        self.last_vocab_idx = mapping["last_vocab_idx"]
        self.last_text_token_idx = mapping["last_vocab_idx"] - self.action_tokenizer.vocab_size

        rank0_print(f"VLA tokens: BOA={self.boa_token_id}, EOA={self.eoa_token_id}, last_vocab_idx={self.last_vocab_idx}")

        # Load data from pickle file
        data_path = getattr(data_args, "data_path", None)
        if data_path and data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
            rank0_print(f"Loaded {len(self.data)} samples from pickle file: {data_path}")
        else:
            raise ValueError("NuPlan dataset requires pickle file")

        # Config
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        # Driving-specific config
        self.use_previous_actions = getattr(data_args, "use_previous_actions", False)
        self.action_hz = getattr(data_args, "action_hz", 2)
        self.frames_per_second = getattr(data_args, "frames_per_second", 10)  # NuPlan/Emu3 convention
        self.va_pair_num = 2  # fixed per requirement
        self.gap_frames = getattr(data_args, "va_gap_frames", self.frames_per_second)  # 1s gap default
        self.rng = random.Random(data_args.seed if hasattr(data_args, 'seed') else 42)

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        if not getattr(self.data_args, 'data_root', None):
            raise ValueError("data_root is required for NuPlan VLA dataset")

        # ROSS features
        self.return_raw_vae = getattr(data_args, "return_raw_vae", True)
        self.return_masks = getattr(data_args, "return_masks", True)

        rank0_print("NuPlan2VA: frames_per_second=%d, action_hz=%d, gap_frames=%d" % (self.frames_per_second, self.action_hz, self.gap_frames))

    def __len__(self):
        return len(self.data)

    @property
    def lengths(self):
        # Rough estimate: 2 images + two action segments (pre 1s fut + cur 4s fut)
        length_list = []
        for sample in self.data:
            img_tokens = 2 * 128 if "image" in sample else 0
            action_tokens = (int(1.0 * self.action_hz) + int(4.0 * self.action_hz))
            try:
                text_tokens = 200 if "text" in sample else 50
                length_list.append(text_tokens + img_tokens + action_tokens)
            except:
                length_list.append(400)
        return length_list

    # ---------- Image helpers ----------
    def _resolve_image_path(self, image_path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        if (isinstance(image_path, str)
            and image_path.lower().endswith(image_extensions)
            and os.path.exists(image_path)):
            return image_path
        return self._construct_image_path(image_path)

    def _construct_image_path(self, original_path):
        path_parts = original_path.split('/') if isinstance(original_path, str) else []
        if len(path_parts) >= 2:
            filename = path_parts[-1]
            if filename.endswith('.npy'):
                filename = filename[:-4]
            last_two = os.path.join(path_parts[-2], "CAM_F0", filename)
        else:
            filename = path_parts[-1] if path_parts else str(original_path)
            if isinstance(filename, str) and filename.endswith('.npy'):
                filename = filename[:-4]
            last_two = filename
        data_root = self.data_args.data_root
        actual_path = os.path.join(data_root, last_two + '.jpg')
        if os.path.exists(actual_path):
            return actual_path
        for ext in ['.jpeg', '.png']:
            alt_path = os.path.join(data_root, last_two + ext)
            if os.path.exists(alt_path):
                return alt_path
        print(f"Warning: Cannot resolve image path: {original_path}, trying original path")
        return original_path

    def load_image_by_index(self, scene, frame_idx):
        if "image" not in scene:
            return None, None
        image_paths = scene["image"]
        image_file = None
        if isinstance(image_paths, str):
            image_file = image_paths
        elif isinstance(image_paths, list) and len(image_paths) > 0:
            if frame_idx < 0:
                frame_idx = 0
            if frame_idx >= len(image_paths):
                frame_idx = len(image_paths) - 1
            image_file = image_paths[frame_idx]
        else:
            return None, None
        if not image_file:
            return None, None
        image_file = self._resolve_image_path(image_file)
        result = self.process_image_unified(image_file)
        if self.return_raw_vae and len(result) == 3:
            return result  # (image_tensor, grid_thw, raw_vae_img_tensor)
        else:
            return result[:2] if len(result) >= 2 else result  # (image_tensor, grid_thw)

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        if self.return_raw_vae:
            try:
                raw_vae_img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0 * 2 - 1  # [3, H, W], [-1, 1]
            except Exception:
                raw_vae_img_tensor = None
            return image_tensor, grid_thw, raw_vae_img_tensor
        return image_tensor, grid_thw

    # ---------- Action helpers ----------
    def wrap_action_sequence(self, action_ids: List[int]) -> torch.Tensor:
        return torch.tensor([self.boa_token_id] + action_ids + [self.eoa_token_id], dtype=torch.long)

    def process_actions(self, actions: np.ndarray) -> List[int]:
        if self.actions_format == "fast":
            if isinstance(actions, list):
                tensor_list = [torch.tensor(item).unsqueeze(0) for item in actions]
                action_tokens = torch.cat(tensor_list, dim=0)
            else:
                action_tokens = torch.tensor(actions) if isinstance(actions, np.ndarray) else actions
            action_ids = self.action_tokenizer(action_tokens)[0]
            mapped_action_ids = [self.last_vocab_idx - id for id in action_ids]
            return mapped_action_ids
        else:
            raise ValueError(f"Unsupported actions_format: {self.actions_format}")

    def _sample_actions_window(self, actions: np.ndarray, anchor_idx: int, hist_secs: float, fut_secs: float) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample actions around an anchor using action_hz within a frames timeline.

        Supports two layouts:
        - [T, H, D]: per-frame future horizon (H steps at action_hz, e.g., H=8 for 4s@2Hz)
          - history: actions[anchor_idx - step*hist_count, :hist_count, :]
          - future:  actions[anchor_idx, :fut_count, :]
        - [T, D]: per-frame single-step actions (fallback)
          - history: concat actions at indices [anchor- step*k]
          - future:  concat actions at indices [anchor + step*k]
        """
        if actions is None:
            return None, None

        step = max(1, int(round(self.frames_per_second / max(1, self.action_hz))))
        hist_count = int(round(hist_secs * self.action_hz))
        fut_count = int(round(fut_secs * self.action_hz))

        hist_tensor = None
        fut_wrapped = None

        # Case A: actions with future horizon [T, H, D]
        if actions.ndim == 3:
            T, H, D = actions.shape
            eff_hist = max(0, min(hist_count, H))
            eff_fut = max(0, min(fut_count, H))

            # history from earliest start so that the last step is just before anchor
            if self.use_previous_actions and eff_hist > 0:
                hist_start = anchor_idx - step * eff_hist
                if 0 <= hist_start < T:
                    hist_seq = actions[hist_start, :eff_hist, :]
                    hist_ids = self.process_actions(hist_seq)
                    hist_tensor = torch.tensor(hist_ids, dtype=torch.long)

            # future from anchor
            if eff_fut > 0 and 0 <= anchor_idx < T:
                fut_seq = actions[anchor_idx, :eff_fut, :]
                fut_ids = self.process_actions(fut_seq)
                fut_wrapped = self.wrap_action_sequence(fut_ids)

            return hist_tensor, fut_wrapped

        # Case B: fallback [T, D] single-step per frame
        # History indices (oldest -> most recent), exclude anchor
        hist_indices = [anchor_idx - step * k for k in range(hist_count, 0, -1)]
        hist_indices = [idx for idx in hist_indices if 0 <= idx < len(actions)]
        if self.use_previous_actions and len(hist_indices) > 0:
            hist = actions[hist_indices]
            hist_ids = self.process_actions(hist)
            hist_tensor = torch.tensor(hist_ids, dtype=torch.long)

        # Future indices (include anchor)
        fut_indices = [anchor_idx + step * k for k in range(fut_count)]
        fut_indices = [idx for idx in fut_indices if 0 <= idx < len(actions)]
        if len(fut_indices) > 0:
            fut = actions[fut_indices]
            fut_ids = self.process_actions(fut)
            fut_wrapped = self.wrap_action_sequence(fut_ids)

        return hist_tensor, fut_wrapped

    # ---------- Builders ----------
    def _grid_merge_count(self, grid_thw):
        merged = copy.deepcopy(grid_thw)
        if not isinstance(merged, SequenceType):
            merged = [merged]
        return [m.prod() // self.data_args.image_processor.merge_size**2 for m in merged]

    def _build_conversation_for_pairs(self, pairs_prompts: List[Tuple[str, str]]):
        # Each pair: (pre_prompt, cur_prompt)
        conv = []
        for pre_p, cur_p in pairs_prompts:
            conv.extend([
                {"from": "user", "value": f"<image>{pre_p}"},
                {"from": "assistant", "value": ""},
                {"from": "user", "value": f"<image>{cur_p}"},
                {"from": "assistant", "value": ""},
            ])
        return [conv]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item(i)
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)
        # try next sample
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.data) - 1)
                return self._get_item(next_index)
            except Exception as e:
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
        # final raise
        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        scene = self.data[i]

        # Determine scene length: prefer action length if available, else fallback
        scene_len = len(scene["action"])

        # Compute index constraints
        one_sec = int(round(self.frames_per_second * 1.0))
        step = max(1, int(round(self.frames_per_second / max(1, self.action_hz))))
        hist_pre = int(round(1.5 * self.action_hz))
        fut_pre = int(round(1.0 * self.action_hz))
        hist_cur = int(round(1.5 * self.action_hz))
        fut_cur = int(round(4.0 * self.action_hz))

        start_idx = one_sec + step * hist_pre
        end_idx = (scene_len - 1) - (step * (fut_cur - 1))

        low = max(0, start_idx)
        high = max(low, end_idx)
        if high < low:
            # degenerate window; fallback to center
            low = max(0, scene_len // 4)
            high = max(low, scene_len // 2)
        cur_idx = self.rng.randint(low, high)
        pre_idx = cur_idx - one_sec

        # Prepare containers
        all_images: List[torch.Tensor] = []
        all_image_thw: List[torch.Tensor] = []
        grid_thw_image_flat: List[int] = []
        action_segments: List[torch.Tensor] = []
        action_roles: List[str] = []
        actions_array = scene.get("action", None)


        if isinstance(scene["text"], list) and len(scene["text"]) > 0:
            cur_prompt = scene["text"][cur_idx if cur_idx < len(scene["text"]) else -1]
            pre_prompt = scene["text"][pre_idx if 0 <= pre_idx < len(scene["text"]) else max(0, min(cur_idx, len(scene["text"]) - 1))]
        else:
            cur_prompt = scene["text"]
            pre_prompt = scene["text"]


        # Images (pre, cur) with optional raw VAE support
        pre_image_raw = cur_image_raw = None
        pre_result = self.load_image_by_index(scene, pre_idx)
        cur_result = self.load_image_by_index(scene, cur_idx)

        if self.return_raw_vae:
            if pre_result is not None and len(pre_result) == 3:
                pre_img, pre_thw, pre_image_raw = pre_result
            else:
                pre_img, pre_thw = pre_result if pre_result else (None, None)
            if cur_result is not None and len(cur_result) == 3:
                cur_img, cur_thw, cur_image_raw = cur_result
            else:
                cur_img, cur_thw = cur_result if cur_result else (None, None)
        else:
            pre_img, pre_thw = pre_result if pre_result else (None, None)
            cur_img, cur_thw = cur_result if cur_result else (None, None)

        if pre_img is not None and pre_thw is not None:
            all_images.append(pre_img)
            all_image_thw.append(pre_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(pre_thw))
        if cur_img is not None and cur_thw is not None:
            all_images.append(cur_img)
            all_image_thw.append(cur_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(cur_thw))

        # Actions: pre then cur
        if self.use_actions and actions_array is not None:
            pre_hist, pre_fut = self._sample_actions_window(actions_array, pre_idx, 1.5, 1.0)
            if self.use_previous_actions and pre_hist is not None:
                action_segments.append(pre_hist)
                action_roles.append("user")
            if pre_fut is not None:
                action_segments.append(pre_fut)
                action_roles.append("assistant")

            cur_hist, cur_fut = self._sample_actions_window(actions_array, cur_idx, 1.5, 4.0)
            if self.use_previous_actions and cur_hist is not None:
                action_segments.append(cur_hist)
                action_roles.append("user")
            if cur_fut is not None:
                action_segments.append(cur_fut)
                action_roles.append("assistant")

        # Build a single conversation with one pre+cur pair
        sources = self._build_conversation_for_pairs([(pre_prompt, cur_prompt)])

        data_dict = preprocess_qwen_2_visual_vla_sources(
            sources=sources,
            tokenizer=self.tokenizer,
            grid_thw_image=grid_thw_image_flat,
            action_segments=action_segments if len(action_segments) > 0 else None,
            action_roles=action_roles if len(action_roles) > 0 else None,
            return_masks=self.return_masks,
        )

        # Position ids
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(all_image_thw, dim=0) if len(all_image_thw) > 0 else None,
            video_grid_thw=None,
            second_per_grid_ts=None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if len(all_images) > 0:
            data_dict["pixel_values"] = torch.cat(all_images, dim=0)
            data_dict["image_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in all_image_thw], dim=0)

            if self.return_raw_vae:
                T, N = 2, 1
                if cur_image_raw is not None:
                    raw_shape = cur_image_raw.shape
                    raw_tensor = torch.zeros((T, N) + raw_shape, dtype=data_dict["pixel_values"].dtype)
                    if pre_image_raw is not None:
                        raw_tensor[0, 0] = pre_image_raw
                    raw_tensor[1, 0] = cur_image_raw
                    data_dict["raw_pixel_values_vae"] = raw_tensor
                    data_dict["frame_image_counts"] = torch.tensor([1, 1], dtype=torch.long)

        return data_dict


# Copy other classes from original file with VLA support
def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    stacked_tensor = torch.cat(padded_tensors, dim=1)
    return stacked_tensor

@dataclass
class DataCollatorForSupervisedVLADataset(object):
    """Collate examples for supervised VLA fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        # Handle images/videos (same as original)
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        
        # Handle raw VAE pixel values and frame counts
        raw_pixel_values_vae = [instance["raw_pixel_values_vae"] for instance in instances if "raw_pixel_values_vae" in instance]
        frame_image_counts = [instance["frame_image_counts"] for instance in instances if "frame_image_counts" in instance]
        
        if len(raw_pixel_values_vae) != 0:
            # raw_pixel_values_vae shape: [T, N, ...] -> concat along N (batch) dimension
            # Since each instance has N=1, we concat them to create actual batch dimension
            batch["raw_pixel_values_vae"] = torch.cat(raw_pixel_values_vae, dim=1)  # concat along N dimension
        else:
            batch["raw_pixel_values_vae"] = None
            
        if len(frame_image_counts) != 0:
            # frame_image_counts shape: [T] -> stack to create batch dimension
            batch["frame_image_counts"] = torch.stack(frame_image_counts, dim=0)  # create batch dimension
        else:
            batch["frame_image_counts"] = None
        
        # Handle image token masks and action future masks
        image_token_masks = [instance["image_token_masks"] for instance in instances if "image_token_masks" in instance]
        action_future_masks = [instance["action_future_masks"] for instance in instances if "action_future_masks" in instance]
        
        if len(image_token_masks) != 0 and len(action_future_masks) != 0:
            # 将变长的掩码在时间维(最后一维)对齐到同一长度，并在 batch 维度拼接
            max_img_len = max(mask.shape[-1] for mask in image_token_masks)
            max_act_len = max(mask.shape[-1] for mask in action_future_masks)
            target_length = max(max_img_len, max_act_len)

            padded_image_masks = []
            padded_action_masks = []
            for img_mask, act_mask in zip(image_token_masks, action_future_masks):
                pad_img = target_length - img_mask.shape[-1]
                pad_act = target_length - act_mask.shape[-1]

                if pad_img > 0:
                    img_mask = torch.nn.functional.pad(img_mask, (0, pad_img), mode='constant', value=0)
                if pad_act > 0:
                    act_mask = torch.nn.functional.pad(act_mask, (0, pad_act), mode='constant', value=0)

                padded_image_masks.append(img_mask)
                padded_action_masks.append(act_mask)

            batch["image_token_masks"] = torch.cat(padded_image_masks, dim=0)
            batch["action_future_masks"] = torch.cat(padded_action_masks, dim=0)
        else:
            batch["image_token_masks"] = None if len(image_token_masks) == 0 else image_token_masks
            batch["action_future_masks"] = None if len(action_future_masks) == 0 else action_future_masks
               
        return batch


def make_supervised_data_module_navsim2_vla_ross(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make NavSim2 VLA dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedNavsim2VAROSSDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedVLADataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def make_supervised_data_module_nuplan2_vla_ross(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make NuPlan 2VA dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedNuplan2VAROSSDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedVLADataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)