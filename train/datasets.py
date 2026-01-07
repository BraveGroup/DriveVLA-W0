# -*- coding: utf-8 -*-

import json
import os
import os.path as osp
import random
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union
from torch.utils.data import Dataset
from PIL import Image
import sys
sys.path.append("/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu")
from models.tokenizer.action_tokenizer import ActionTokenizer
from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
from torch.utils.data import Subset

    
class Emu3SFTDataset(Dataset):

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__()

        self.args = args
        # data args
        self.random_frame_sampling = args.random_frame_sampling
        self.raw_image = args.raw_image
        
        with open(args.data_path,'rb') as f:
            self.data = pickle.load(f)
        
        if not self.random_frame_sampling:
            self.data = list(self.sliding_window_sampling(self.data, interval=args.action_frames*args.frames))
        
        self.tokenizer = tokenizer
        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]
        self.chat_template="You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:"
        self.gen_template="You are a powerful painter. USER: {text} ASSISTANT:{image}"
        self.act_template="Action: {action_prompt}"
        self.VL = args.VL
        self.cfg = False
        self.post_training = args.post_training

        # pretrain use
        if self.post_training:
            # self.dataset_fps = {'rt1':3, 'bridgev2':5, 'droid':15, '1x':1, 'calvin':5, 'libero':5}
            self.dataset_fps = {'rt1':3, 'bridgev2':5, 'droid':15, '1x':1, 'kuka':3, 'calvin':5, 'libero':10}
        else:
            self.dataset_fps = {}
        self.T = args.frames
        self.action_frames = args.action_frames
        
        self.actions = args.actions
        self.actions_format = args.actions_format

        self.use_gripper = args.use_gripper  

        self.video_format = args.video_format

        self.driving = args.driving if hasattr(args, "driving") else False

        if self.raw_image:
            self.vision_hub = "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/pretrained_models/Emu3-VisionTokenizer"
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
            self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True)
            self.image_processor.min_pixels = 80 * 80

        self.fast_path = args.action_tokenizer_path
        self.action_tokenizer = AutoProcessor.from_pretrained(self.fast_path, trust_remote_code=True)

    def __len__(self):
        return len(self.data)
    
    def sliding_window_sampling(self, data, interval=5):
        """
        Implement sliding window sampling using a generator.
        """
        for item in data:
            T = len(item['image'])
            if T <= interval:
                raise ValueError("Length of 'image', 'action', and 'gripper' must be greater than 'interval'.")
            for start_idx in range(0, T - interval + 1, 1):
                yield {
                    'text': item['text'],
                    'image': item['image'][start_idx:start_idx+interval],
                    'action': item['action'][start_idx:start_idx+interval],
                    'gripper_image': item['gripper_image'][start_idx:start_idx+interval],
                }

    def random_frames_to_tensor(self, img_list, T, action_prompt=None, gripper=None):
        start_idx = random.randint(0, len(img_list) - T)

        if hasattr(self, 'raw_image') and self.raw_image:
            self.image_tokenizer.eval()
            # Process raw images with VQ encoding
            selected_frames = [Image.open(img_path) for img_path in img_list[start_idx:start_idx + T]]
            selected_frames = [self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0) for img in selected_frames]

            tensor_frames = torch.stack(selected_frames, dim=0)
            with torch.no_grad():
                image_code = self.image_tokenizer.encode(tensor_frames)
            
            if gripper is not None and action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                selected_gripper = [Image.open(img_path) for img_path in gripper[start_idx:start_idx + T]]
                selected_gripper = [self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0) for img in selected_gripper]
                tensor_gripper = torch.stack(selected_gripper, dim=0)
                with torch.no_grad():
                    gripper_code = self.image_tokenizer.encode(tensor_gripper)
                return image_code, selected_actions, gripper_code
            elif action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                return image_code, selected_actions
        else:
            selected_frames = [np.load(img_path) for img_path in img_list[start_idx:start_idx + T]]
            tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
            tensor = torch.stack(tensor_frames, dim=1)

            if gripper is not None and action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                selected_gripper = [np.load(img_path) for img_path in gripper[start_idx:start_idx + T]]
                tensor_gripper = [torch.from_numpy(frame) for frame in selected_gripper]
                return tensor.squeeze(0), selected_actions, torch.stack(tensor_gripper, dim=1).squeeze(0)
            elif action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                return tensor.squeeze(0), selected_actions
            elif gripper is not None:
                selected_gripper = [np.load(img_path) for img_path in gripper[start_idx:start_idx + T]]
                tensor_gripper = [torch.from_numpy(frame) for frame in selected_gripper]
                return tensor.squeeze(0), torch.stack(tensor_gripper, dim=1).squeeze(0)
        return tensor.squeeze(0)
    
    def get_fps_for_path(self, image_tokens_path):
        for key in self.dataset_fps.keys():
            if key in image_tokens_path[0]:
                return self.dataset_fps[key]
        # Default return value if no key matches
        return None  # or some default FPS value
    
    def pad_tensor(self, tensor, max_length, pad_value):
        """Pads a tensor to a specified maximum length."""
        current_length = tensor.shape[-1]
        if current_length < max_length:
            pad_length = max_length - current_length
            padding = torch.full((pad_length,), fill_value=pad_value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=-1)
        return tensor

    def __getitem__(self, index: int):

        scene = self.data[index]

        if self.cfg:
            p_prob = random.random()
            if p_prob < self.args.null_prompt_prob:
                prompt = ""
            else:
                prompt = scene["text"]
        else:
            prompt = scene["text"]

        image_tokens_path = scene["image"]

        # handle different dataset fps for post training
        fps = self.get_fps_for_path(image_tokens_path)
        if fps is not None:
            self.action_frames = fps
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        # use action information
        if self.actions:
            action = scene["action"] 
            if self.use_gripper:
                gripper = scene["gripper_image"]
                image_tokens, action_tokens, gripper_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper)
            else:
                image_tokens, action_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action)
            
            if self.video_format == "interleave":
                if self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                    action_ids = self.action_tokenizer(action_tokens)
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                    
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
            else:
                if self.actions_format == "openvla":
                    action_tokens = action_tokens.flatten()
                    action_ids = self.action_tokenizer(action_tokens)

                    # Debugging
                    # action_debug = self.action_tokenizer.decode_token_ids_to_actions(action_ids)
                    # error = action_tokens - action_debug
                elif self.actions_format == "text":
                    action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                    action_prompt = self.act_template.format(action_prompt=action_str)
                elif self.actions_format == "continuous":
                    action_continuous = action_tokens
                elif self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_ids = self.action_tokenizer(action_tokens)[0]
                    # action_decode = self.action_tokenizer.decode([action_ids])
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - id for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
        else:
            if self.use_gripper:
                gripper = scene["gripper_image"]
                image_tokens, gripper_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, gripper=gripper)
            else:
                image_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num) 
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
        # VLA Baseline (Img)
        else:
            image_tokens = image_tokens[0:self.T,...]
            image_prompt = self.format_video_prompt(image_tokens)

            if self.use_gripper:
                gripper_tokens = gripper_tokens[0:self.T,...]
                gripper_prompt = self.format_video_prompt(gripper_tokens)
                image_prompt = image_prompt + gripper_prompt  

            if self.VL:
                p_prob_order = random.random()
                if p_prob_order < 0.5:
                    input = self.tokenizer.bos_token + prompt + image_prompt + self.tokenizer.eos_token
                else:
                    # input = self.tokenizer.bos_token + image_prompt + prompt
                    input = self.tokenizer.bos_token + self.chat_template.format(image_prompt=image_prompt, text_prompt=prompt) + self.tokenizer.eos_token
            else:
                input = self.tokenizer.bos_token + prompt + image_prompt 
            # 先不进行padding，后面统一padding
            sample = self.tokenizer(
                input,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            labels = sample["input_ids"]

            # only use vision loss
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            # based on the actions_format, append the action information to the sample
            if self.actions:
                if self.actions_format == "openvla":
                    action_sample = self.wrap_action_sequence(action_ids)
                    sample["input_ids"] = torch.cat([sample["input_ids"], action_sample], dim=-1)

                    # Update attention_mask
                    action_mask = torch.ones_like(action_sample, dtype=torch.long)
                    sample["attention_mask"] = torch.cat([sample["attention_mask"], action_mask], dim=-1)

                    action_labels = action_sample.clone()  # Clone action_sample for labels
                    sample["labels"] = torch.cat([sample["labels"], action_labels], dim=-1)
                
                # FAST
                elif self.actions_format == "fast":
                    if self.args.apply_loss_on_only_action:
                        sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                    sample = self.append_action_to_sample(sample, action_ids)
                
                # Flow Matching
                elif self.actions_format == "continuous":
                    boa_token_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
                    sample = self.append_boa_to_sample(sample, [boa_token_id])
                    sample["action"] = action_continuous
            
            # finally, do padding
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        return sample

    def append_action_to_sample(self, sample, action_ids):
        """
        将 action_ids 处理后，追加到 sample 中，包括 input_ids, attention_mask 和 labels。
        """
        # action_sample = self.wrap_action_sequence(action_ids)[1:]  # Exclude the first BOA token
        action_sample = self.wrap_action_sequence(action_ids)  # Include the BOA token
        action_mask = torch.ones_like(action_sample, dtype=torch.long)

        for key, value in zip(["input_ids", "attention_mask", "labels"], [action_sample, action_mask, action_sample.clone()]):
            sample[key] = torch.cat([sample[key], value], dim=-1)

        return sample
    
    def append_boa_to_sample(self, sample, action_ids):

        action_sample = torch.tensor(action_ids, dtype=torch.long)
        action_mask = torch.ones_like(action_sample, dtype=torch.long)

        for key, value in zip(["input_ids", "attention_mask", "labels"], [action_sample, action_mask, action_sample.clone()]):
            sample[key] = torch.cat([sample[key], value], dim=-1)

        return sample

    def wrap_action_sequence(self, action_ids: List[int]) -> torch.Tensor:
        """
        Wraps a sequence of action token IDs with special tokens (beginning and end).

        Args:
            action_ids (List[int]): The sequence of action token IDs.

        Returns:
            torch.Tensor: A tensor containing the wrapped sequence.
        """
        # Encode the beginning and end action tokens
        action_begin = self.tokenizer.encode(self.tokenizer.boa_token)[0]
        action_end = self.tokenizer.encode(self.tokenizer.eoa_token)[0]
        eos = self.tokenizer.encode(self.tokenizer.eos_token)[0]

        # Wrap the action sequence
        # wrapped_action = [action_begin] + action_ids + [action_end] + [eos]
        wrapped_action = [action_begin] + action_ids + [action_end]
        
        # Convert to a PyTorch tensor
        return torch.tensor(wrapped_action, dtype=torch.long)

    def format_video_prompt(self, video_tokens):
        # 假设video_tokens是一个形状为[frames, height, width]的张量
        frames, h, w = video_tokens.shape
        videostr = self.to_videostr(video_tokens)

        video_prompt = (
            self.tokenizer.boi_token +
            f"{frames}*{h}*{w}" +  # 视频的帧数、高度和宽度
            self.tokenizer.img_token +  # 视频开始标记
            videostr +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return video_prompt

    def to_videostr(self, video_tokens):
        frame_str_list = []
        for frame in video_tokens:
            frame_token_str = [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in frame.flatten()
            ]
            frame_str = "".join(frame_token_str)
            frame_str_list.append(frame_str)
        videostr = self.tokenizer.eof_token.join(frame_str_list)
        return videostr


    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token +
            f"{h}*{w}" +
            self.tokenizer.img_token +
            imgstr +
            self.tokenizer.eol_token +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return image_prompt

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr


class Emu3DrivingDataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__(args, tokenizer=tokenizer)
        self.use_previous_actions = args.use_previous_actions
        self.cur_idx = args.cur_frame_idx
        self.use_flip = args.use_flip
        self.rng = random.Random(args.seed if hasattr(args, 'seed') else 42)

    def random_frames_to_tensor(self, img_list, T, num_frames, action_prompt=None, gripper=None, do_flip=False):
        start_idx = self.cur_idx
        selected_frames = []

        for image_path in img_list[start_idx-2*(num_frames-1):start_idx + 1:2]:  
            if do_flip:
                image_path = image_path.replace("/trainval_vq_codes/", "/trainval_vq_codes_flip/")
            # image_path = image_path.replace("/mnt/nvme0n1p1/yingyan.li/repo/OmniSim/", "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/")
            selected_frames.append(np.load(image_path))

        tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
        img_tensor = torch.stack(tensor_frames, dim=1)

        selected_actions = action_prompt[start_idx:start_idx + T]
        return img_tensor.squeeze(0), selected_actions

    def train_test_split(self, test_size=0.05, seed=42):
        total_size = len(self.data)
        indices = np.arange(total_size)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        split = int(total_size * (1 - test_size))
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_set = Subset(self, train_indices)
        val_set = Subset(self, val_indices)
        return {"train": train_set, "test": val_set}

    def __getitem__(self, index: int):
        scene = self.data[index]
        prompt = scene["text"][self.cur_idx]
        image_tokens_path = scene["image"]

        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)

        do_flip = self.use_flip and self.rng.random() < 0.5
        
        if self.actions:
            action = scene["action"]
            if do_flip:
                action = action.copy()
                action[:, 1:] *= -1
            image_tokens, action_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, self.T, action_prompt=action, do_flip=do_flip)

        if self.actions_format == "fast":
            if isinstance(action_tokens, list):
                tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                action_tokens = torch.cat(tensor_list, dim=0)
            action_ids = self.action_tokenizer(action_tokens)[0]
            self.last_vocab_idx = self.tokenizer.pad_token_id - 1
            action_ids = [self.last_vocab_idx - id for id in action_ids]

        image_prompt = self.format_video_prompt(image_tokens)
        input = self.tokenizer.bos_token + prompt + image_prompt

        sample = self.tokenizer(
            input,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        labels = sample["input_ids"]

        if self.args.apply_loss_on_only_vision:
            labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

        sample["labels"] = labels
        for k, v in sample.items():
            sample[k] = v.squeeze(0)

        if self.actions:
            if self.use_previous_actions:
                previous_actions = action[0:self.cur_idx]
                previous_action_ids = self.action_tokenizer(previous_actions)[0]
                previous_action_ids = [self.last_vocab_idx - id for id in previous_action_ids]
                # previous_action_sample = self.wrap_action_sequence(previous_action_ids)
                previous_action_sample = torch.tensor(previous_action_ids, dtype=torch.long)
                previous_action_mask = torch.ones_like(previous_action_sample, dtype=torch.long)
                previous_action_label = torch.full_like(previous_action_sample, fill_value=self.args.ignore_index, dtype=torch.long)

                for key, value in zip(["input_ids", "attention_mask", "labels"], [previous_action_sample, previous_action_mask, previous_action_label]):
                    # sample[key] = torch.cat([value, sample[key]], dim=-1)
                    sample[key] = torch.cat([sample[key], value], dim=-1)

            if self.actions_format == "fast":
                if self.args.apply_loss_on_only_action:
                    sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                sample = self.append_action_to_sample(sample, action_ids)

            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)

        return sample


class Emu3DrivingVAVADataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__(args, tokenizer=tokenizer)
        self.use_previous_actions = args.use_previous_actions
        self.cur_idx = args.cur_frame_idx
        self.use_flip = args.use_flip
        self.rng = random.Random(args.seed if hasattr(args, 'seed') else 42)

        # 原始类别列表
        self.text_name_list = [
            "go left",
            "go straight",
            "go right",
            "unknown",
        ]

        # 生成 {str: one-hot tensor} 映射
        self.prompt2vec = {
            name: F.one_hot(torch.tensor(i), num_classes=len(self.text_name_list)).float()
            for i, name in enumerate(self.text_name_list)
        }

    def random_frames_to_tensor(self, img_list, T, num_frames, action_prompt=None, gripper=None, do_flip=False):
        start_idx = self.cur_idx
        selected_frames = []

        for image_path in img_list[start_idx-2*(num_frames-1):start_idx + 1:2]:  #✅修改
            if do_flip:
                image_path = image_path.replace("/trainval_vq_codes/", "/trainval_vq_codes_flip/")
            image_path = image_path.replace("/mnt/vdb1/yingyan.li/repo/VLA/", "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/")
            selected_frames.append(np.load(image_path).reshape(1,18,32))

        tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
        img_tensor = torch.stack(tensor_frames, dim=1)

        selected_actions = action_prompt[start_idx:start_idx + T]
        return img_tensor.squeeze(0), selected_actions

    def train_test_split(self, test_size=0.05, seed=42):
        total_size = len(self.data)
        indices = np.arange(total_size)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        split = int(total_size * (1 - test_size))
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_set = Subset(self, train_indices)
        val_set = Subset(self, val_indices)
        return {"train": train_set, "test": val_set}

    def __getitem__(self, index: int):
        # pre_index = index-8 if index-8 >= 0 else index

        scene = self.data[index]
        prompt = scene["text"][self.cur_idx]
        image_tokens_path = scene["image"]

        # pre_scene = self.data[pre_index]
        pre_prompt = scene["pre_1s_text"][self.cur_idx]
        pre_image_tokens_path = scene["pre_1s_image"]

        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)

        do_flip = self.use_flip and self.rng.random() < 0.5
        
        if self.actions:
            action = scene["action"]
            pre_action = scene["pre_1s_action"]
            if do_flip:
                action = action.copy()
                action[:, 1:] *= -1
            image_tokens, action_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, self.T, action_prompt=action, do_flip=do_flip)
            pre_image_tokens, pre_action_tokens = self.random_frames_to_tensor(pre_image_tokens_path, 2, self.T, action_prompt=pre_action, do_flip=do_flip)
            pre_action_tokens = np.array(pre_action_tokens)

        if self.actions_format == "fast":
            if isinstance(action_tokens, list):
                tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                action_tokens = torch.cat(tensor_list, dim=0)

                pre_tensor_list = [torch.tensor(item).unsqueeze(0) for item in pre_action_tokens]
                pre_action_tokens = torch.cat(pre_tensor_list, dim=0)
            self.last_vocab_idx = self.tokenizer.pad_token_id - 1
            action_ids = self.action_tokenizer(action_tokens)[0]
            action_ids = [self.last_vocab_idx - id for id in action_ids]

            pre_action_ids = self.action_tokenizer(pre_action_tokens)[0]
            pre_action_ids = [self.last_vocab_idx - id for id in pre_action_ids]

        image_prompt = self.format_video_prompt(image_tokens)
        #current image prompt don`t need bos_token`
        input = prompt + image_prompt

        pre_image_prompt = self.format_video_prompt(pre_image_tokens)
        pre_input = self.tokenizer.bos_token + pre_prompt + pre_image_prompt

        sample = self.tokenizer(
            input,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        labels = sample["input_ids"]

        if self.args.apply_loss_on_only_vision:
            labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)
        else:
            labels = torch.full_like(labels, fill_value=self.args.ignore_index)

        sample["labels"] = labels
        for k, v in sample.items():
            sample[k] = v.squeeze(0)

        pre_sample = self.tokenizer(
            pre_input,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        pre_labels = pre_sample["input_ids"]

        if self.args.apply_loss_on_only_vision:
            pre_labels = torch.where(torch.logical_and(pre_labels >= self.bov, pre_labels <= self.eov), pre_labels, self.args.ignore_index)

        pre_sample["labels"] = pre_labels
        for k, v in pre_sample.items():
            pre_sample[k] = v.squeeze(0)
    
    
        if self.actions:
            if self.use_previous_actions:
                previous_actions = action[0:self.cur_idx]
                previous_action_ids = self.action_tokenizer(previous_actions)[0]
                previous_action_ids = [self.last_vocab_idx - id for id in previous_action_ids]
                # previous_action_sample = self.wrap_action_sequence(previous_action_ids)
                previous_action_sample = torch.tensor(previous_action_ids, dtype=torch.long)
                previous_action_mask = torch.ones_like(previous_action_sample, dtype=torch.long)
                previous_action_label = torch.full_like(previous_action_sample, fill_value=self.args.ignore_index, dtype=torch.long)

                pre_previous_actions = np.array(pre_action[0:self.cur_idx])
                pre_previous_action_ids = self.action_tokenizer(pre_previous_actions)[0]
                pre_previous_action_ids = [self.last_vocab_idx - id for id in pre_previous_action_ids]
                # pre_previous_action_sample = self.wrap_action_sequence(pre_previous_action_ids)
                pre_previous_action_sample = torch.tensor(pre_previous_action_ids, dtype=torch.long)
                pre_previous_action_mask = torch.ones_like(pre_previous_action_sample, dtype=torch.long)
                pre_previous_action_label = torch.full_like(pre_previous_action_sample, fill_value=self.args.ignore_index, dtype=torch.long)
                

                for key, value in zip(["input_ids", "attention_mask", "labels"], [previous_action_sample, previous_action_mask, previous_action_label]):
                    sample[key] = torch.cat([sample[key], value], dim=-1)

                for key, value in zip(["input_ids", "attention_mask", "labels"], [pre_previous_action_sample, pre_previous_action_mask, pre_previous_action_label]):
                    # pre_sample[key] = torch.cat([value, pre_sample[key]], dim=-1)
                    pre_sample[key] = torch.cat([pre_sample[key], value], dim=-1)

            if self.actions_format == "fast":
                if self.args.apply_loss_on_only_action:
                    sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                    pre_sample['labels'] = torch.full_like(pre_sample['labels'], self.args.ignore_index)
                
                sample = self.append_action_to_sample(sample, action_ids)
                pre_sample = self.append_action_to_sample(pre_sample, pre_action_ids)

            for k, _ in sample.items():
                sample[k] = torch.cat([pre_sample[k], sample[k]], dim=-1)

            # action expert 需要的continous action
            sample["action"] = torch.tensor(action_tokens, dtype=torch.float)
            sample["pre_action"] = torch.tensor(np.array(action[0:self.cur_idx]), dtype=torch.float)
            sample["cmd"] = self.prompt2vec[prompt]

            
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)

        # "pre_action": torch.tensor(np.array(action[0:self.cur_idx]), dtype=torch.float),
        # "cmd": self.prompt2vec[prompt],

        return sample


class Emu3DrivingNuplan6VADataset(Emu3SFTDataset):
    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__(args, tokenizer=tokenizer)
        self.use_previous_actions = args.use_previous_actions
        self.cur_idx = args.cur_frame_idx
        self.use_flip = args.use_flip
        self.rng = random.Random(args.seed if hasattr(args, 'seed') else 42)
        self.vq_root = args.vq_root if hasattr(args, 'vq_root') else ''
        self.pre_action_frames = args.pre_action_frames if hasattr(args, 'pre_action_frames') else 3
        self.resolution = args.resolution if hasattr(args, 'resolution') else (36, 64)
        self.action_hz = args.action_hz if hasattr(args, 'action_hz') else 2  # action采样频率，默认2hz
        self.va_pair_num = args.va_pair_num if hasattr(args, 'va_pair_num') else 6  # 每个样本的视觉-动作对数量

    def random_frames_to_tensor(self, img_list, T, num_frames, action_prompt=None, gripper=None, do_flip=False, cur_idx=None):
        selected_frames = []
        selected_frames.append(np.load(osp.join(self.vq_root, img_list[cur_idx])).reshape(1, *self.resolution))
        tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
        img_tensor = torch.stack(tensor_frames, dim=1)
        selected_actions = np.array(action_prompt[cur_idx,:2])  # 只选择前两个动作
        return img_tensor.squeeze(0), selected_actions

    def train_test_split(self, test_size=0.05, seed=42):
        total_size = len(self.data)
        indices = np.arange(total_size)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        split = int(total_size * (1 - test_size))
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_set = Subset(self, train_indices)
        val_set = Subset(self, val_indices)
        return {"train": train_set, "test": val_set}

    def __getitem__(self, index: int):
        scene = self.data[index]

        start_idx = 15  # TODO: hard code
        end_idx = len(scene["text"]) - 10 * self.va_pair_num 
        
        cur_idx = self.rng.randint(start_idx, end_idx - 1)
        
        idx_list = [cur_idx+10*i for i in range(self.va_pair_num)]

        image_tokens_path = scene["image"]

        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames


        do_flip = self.use_flip and self.rng.random() < 0.5

        # Prepare tensors for all pairs
        all_input_ids = []
        all_labels = []
        all_attention_masks = []

        for i, idx in enumerate(idx_list):
            if self.actions:
                action = scene["action"]
                prompt = scene["text"][idx]
                if do_flip:
                    action = action.copy()
                    action[:, 1:] *= -1
                
                image_tokens, action_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, self.T, action_prompt=action, do_flip=do_flip, cur_idx=idx)

            if self.actions_format == "fast":
                if isinstance(action_tokens, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                    action_tokens = torch.cat(tensor_list, dim=0)
                action_ids = self.action_tokenizer(action_tokens)[0]
                self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                action_ids = [self.last_vocab_idx - id for id in action_ids]
            
            image_prompt = self.format_video_prompt(image_tokens)
            if i == 0:
                input = self.tokenizer.bos_token + prompt + image_prompt
            else:
                input = prompt + image_prompt

            sample = self.tokenizer(
                input,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            labels = sample["input_ids"]

            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if self.actions and self.use_previous_actions:
                frame_interval = int(10 / self.action_hz)  
                previous_actions = []      
                for i in range(self.pre_action_frames):
                    past_frame_idx = idx - (i + 1) * frame_interval
                    if past_frame_idx >= 0:
                        previous_actions.append(action[past_frame_idx][0])
                    else:
                        previous_actions.append(action[idx][0])                
                previous_actions = previous_actions[::-1]
                previous_actions = np.array(previous_actions)
                
                previous_action_ids = self.action_tokenizer(previous_actions)[0]
                previous_action_ids = [self.last_vocab_idx - id for id in previous_action_ids]
                previous_action_sample = torch.tensor(previous_action_ids, dtype=torch.long)
                # previous_action_sample = self.wrap_action_sequence(previous_action_ids)
                previous_action_mask = torch.ones_like(previous_action_sample, dtype=torch.long)
                previous_action_label = torch.full_like(previous_action_sample, fill_value=self.args.ignore_index, dtype=torch.long)

                for key, value in zip(["input_ids", "attention_mask", "labels"], [previous_action_sample, previous_action_mask, previous_action_label]):
                    sample[key] = torch.cat([sample[key], value], dim=-1)

            if self.actions and self.actions_format == "fast":
                if self.args.apply_loss_on_only_action:
                    sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                sample = self.append_action_to_sample(sample, action_ids)

            all_input_ids.append(sample["input_ids"])
            all_labels.append(sample["labels"])
            all_attention_masks.append(sample["attention_mask"])

        # for key, _ in sample.items():
        #     sample[key] = torch.cat([sample[key], next_sample[key]], dim=-1)
        sample['input_ids'] = torch.cat(all_input_ids, dim=-1)
        sample['labels'] = torch.cat(all_labels, dim=-1)
        sample['attention_mask'] = torch.cat(all_attention_masks, dim=-1)

        sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

        for k, v in sample.items():
            sample[k] = v.squeeze(0)

        if "labels" in sample:
            sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)

        return sample


class Emu3DrivingVAVA_AR_Dataset(Emu3DrivingVAVADataset):
    """
    专用于 Emu3AutoRegressive 训练/推理的数据集：
    - 产出完整的 VLM 输入序列：vlm_input_ids / vlm_attention_mask（包含两段 <boa>...<eoa>）。
    - 从完整序列尾部截取第二段 <boa>...<eoa> 作为 AR 的 action_input_ids 与 labels。
    - 同时提供 pre_action（连续）、cmd（one-hot）以供模型构造 <state> token。
    注意：不再把完整序列作为 labels 回传；labels 专用于 AR 的 token 监督。
    """

    def __getitem__(self, index: int):
        scene = self.data[index]

        # 如果scene有token这个key，Token转为16进制tensor
        token = scene.get("token", None)
        if token is not None:
            b = bytes.fromhex(token)  # 8 bytes
            token = torch.tensor(list(b), dtype=torch.uint8)  # shape [8]
        else:
            # token置为全0 tensor
            token = torch.zeros(8, dtype=torch.uint8)

        # 当前与前一秒提示与图像序列
        prompt = scene["text"][self.cur_idx]
        image_tokens_path = scene["image"]

        pre_prompt = scene["pre_1s_text"][self.cur_idx]
        pre_image_tokens_path = scene["pre_1s_image"]

        # 采样帧数
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)

        do_flip = self.use_flip and self.rng.random() < 0.5

        # 动作（连续）
        action = scene["action"]
        pre_action = scene["pre_1s_action"]

        # 图像与动作采样（当前与前一秒）
        image_tokens, action_tokens = self.random_frames_to_tensor(
            image_tokens_path, frames_num, self.T, action_prompt=action, do_flip=do_flip
        )
        pre_image_tokens, pre_action_tokens = self.random_frames_to_tensor(
            pre_image_tokens_path, 2, self.T, action_prompt=pre_action, do_flip=do_flip
        )
        pre_action_tokens = np.array(pre_action_tokens)

        # 离散动作 ids（fast）
        if self.actions_format != "fast":
            raise ValueError(f"Invalid actions_format for AR dataset: {self.actions_format}")

        if isinstance(action_tokens, list):
            tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
            action_tokens = torch.cat(tensor_list, dim=0)
        if isinstance(pre_action_tokens, list):
            pre_tensor_list = [torch.tensor(item).unsqueeze(0) for item in pre_action_tokens]
            pre_action_tokens = torch.cat(pre_tensor_list, dim=0)

        self.last_vocab_idx = self.tokenizer.pad_token_id - 1
        action_ids = self.action_tokenizer(action_tokens)[0]
        action_ids = [self.last_vocab_idx - id for id in action_ids]

        pre_action_ids = self.action_tokenizer(pre_action_tokens)[0]
        pre_action_ids = [self.last_vocab_idx - id for id in pre_action_ids]

        # 文本+图像 prompt
        image_prompt = self.format_video_prompt(image_tokens)
        input_text = prompt + image_prompt  # 当前不加 bos

        pre_image_prompt = self.format_video_prompt(pre_image_tokens)
        pre_input_text = self.tokenizer.bos_token + pre_prompt + pre_image_prompt

        # tokenizer 编码
        sample = self.tokenizer(
            input_text, padding=False, return_token_type_ids=False, return_tensors="pt",
        )
        for k, v in sample.items():
            sample[k] = v.squeeze(0)
        # 初始化 labels 为 ignore_index，便于后续追加 action 时对齐长度
        sample["labels"] = torch.full_like(sample["input_ids"], fill_value=self.args.ignore_index)

        pre_sample = self.tokenizer(
            pre_input_text, padding=False, return_token_type_ids=False, return_tensors="pt",
        )
        for k, v in pre_sample.items():
            pre_sample[k] = v.squeeze(0)
        pre_sample["labels"] = torch.full_like(pre_sample["input_ids"], fill_value=self.args.ignore_index)

        # 拼接 previous actions（若需要）
        if self.use_previous_actions:
            previous_actions = np.array(action[0:self.cur_idx])
            previous_action_ids = self.action_tokenizer(previous_actions)[0]
            previous_action_ids = [self.last_vocab_idx - id for id in previous_action_ids]
            prev_action_tensor = torch.tensor(previous_action_ids, dtype=torch.long)
            prev_mask = torch.ones_like(prev_action_tensor, dtype=torch.long)
            prev_label = torch.full_like(prev_action_tensor, fill_value=self.args.ignore_index, dtype=torch.long)

            for key, value in zip(["input_ids", "attention_mask", "labels"], [prev_action_tensor, prev_mask, prev_label]):
                sample[key] = torch.cat([sample[key], value], dim=-1)

            pre_previous_actions = np.array(pre_action[0:self.cur_idx])
            pre_previous_action_ids = self.action_tokenizer(pre_previous_actions)[0]
            pre_previous_action_ids = [self.last_vocab_idx - id for id in pre_previous_action_ids]
            pre_prev_action_tensor = torch.tensor(pre_previous_action_ids, dtype=torch.long)
            pre_prev_mask = torch.ones_like(pre_prev_action_tensor, dtype=torch.long)
            pre_prev_label = torch.full_like(pre_prev_action_tensor, fill_value=self.args.ignore_index, dtype=torch.long)
            for key, value in zip(["input_ids", "attention_mask", "labels"], [pre_prev_action_tensor, pre_prev_mask, pre_prev_label]):
                pre_sample[key] = torch.cat([pre_sample[key], value], dim=-1)

        # 追加 <boa>action<eoa>
        sample = self.append_action_to_sample(sample, action_ids)
        pre_sample = self.append_action_to_sample(pre_sample, pre_action_ids)

        # 完整序列：先前一秒，再当前
        full = {}
        for k in sample.keys():
            full[k] = torch.cat([pre_sample[k], sample[k]], dim=-1)

        # 对完整序列做 max_length padding（只用于 VLM 通道）
        full = self.tokenizer.pad(full, padding="max_length", return_tensors="pt")
        full = {k: v.squeeze(0) for k, v in full.items()}

        # 从尾部截取第二段 <boa>...<eoa> 作为 AR 输入/label
        boa_token_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
        eoa_token_id = self.tokenizer.encode(self.tokenizer.eoa_token)[0]
        vlm_ids_no_pad = full["input_ids"][full["attention_mask"].bool()]

        boa_positions = (vlm_ids_no_pad == boa_token_id).nonzero(as_tuple=False).flatten()
        if boa_positions.numel() < 2:
            # 保险：若仅找到1个 <boa>，则使用最后一个
            boa_idx = boa_positions[-1].item()
        else:
            boa_idx = boa_positions[-1].item()
        # 从 boa_idx 往后找 eoa
        try:
            eoa_rel_idx = (vlm_ids_no_pad[boa_idx:] == eoa_token_id).nonzero(as_tuple=False).flatten()[0].item()
        except IndexError:
            raise RuntimeError("无法在完整序列中定位 <eoa>，请检查数据构造是否包含动作片段")
        eoa_idx = boa_idx + eoa_rel_idx

        action_ids_tensor = vlm_ids_no_pad[boa_idx:eoa_idx + 1]

        # 构造 VLM 监督标签：
        # - vision token：参与监督（非视觉文本被 -100 掩蔽）
        # - 追加的当前 action token（最后一段 <boa>...<eoa>）：参与监督
        # - 之前动作与 PAD：被 -100 掩蔽
        vlm_input_ids_full = full["input_ids"].clone()
        vlm_attn_mask_full = full["attention_mask"]

        # 基于 attention_mask 屏蔽 PAD
        vlm_labels_full = vlm_input_ids_full.clone()
        vlm_labels_full[vlm_attn_mask_full == 0] = self.args.ignore_index

        # 视觉 token 区间 [bov, eov]
        vision_mask_full = (vlm_input_ids_full >= self.bov) & (vlm_input_ids_full <= self.eov)

        # 计算最后一段 <boa>...<eoa> 的位置（映射回 padding 后的 index）
        nonpad_positions = torch.nonzero(vlm_attn_mask_full, as_tuple=False).flatten()
        action_pos_full = nonpad_positions[boa_idx:eoa_idx + 1]
        action_mask_full = torch.zeros_like(vlm_attn_mask_full, dtype=torch.bool)
        action_mask_full[action_pos_full] = True

        # 仅保留 vision 或 最后一段 action 的标签，其余置为 -100
        keep_mask_full = vision_mask_full | action_mask_full
        vlm_labels_full = torch.where(
            keep_mask_full,
            vlm_labels_full,
            torch.full_like(vlm_labels_full, self.args.ignore_index)
        )

        # 固定长度 padding 到 15; 超长截断并确保以 <eoa> 结尾
        MAX_AR_LEN = 15
        pad_id = self.tokenizer.pad_token_id
        if action_ids_tensor.shape[0] >= MAX_AR_LEN:
            action_fixed = action_ids_tensor[:MAX_AR_LEN].clone()
            action_fixed[-1] = eoa_token_id  # 保证结尾为 <eoa>
            labels_fixed = action_fixed.clone()
        else:
            pad_len = MAX_AR_LEN - action_ids_tensor.shape[0]
            action_fixed = torch.cat([
                action_ids_tensor,
                torch.full((pad_len,), pad_id, dtype=action_ids_tensor.dtype)
            ], dim=0)
            labels_fixed = torch.cat([
                action_ids_tensor,
                torch.full((pad_len,), self.args.ignore_index, dtype=torch.long)
            ], dim=0)

        # 输出字典（仅保留必要字段）
        out = {
            "input_ids": action_fixed,
            "vlm_input_ids": full["input_ids"],
            "vlm_attention_mask": full["attention_mask"],
            "vlm_labels": vlm_labels_full,
            "action_input_ids": action_fixed,   # 固定 15 长度
            "labels": labels_fixed,             # pad 区域使用 ignore_index
            # 连续动作与状态
            "action": torch.tensor(action_tokens, dtype=torch.float),
            "pre_action": torch.tensor(np.array(action[0:self.cur_idx]), dtype=torch.float),
            "cmd": self.prompt2vec[prompt],
            "token": token
        }

        return out