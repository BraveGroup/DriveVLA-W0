import os
import sys
import json
import yaml
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import LogitsProcessor, AutoConfig
from transformers.configuration_utils import PretrainedConfig
from tqdm import tqdm


# Project root - dynamically get the project root directory
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

# Add train and qwen-vl reference paths
train_dir = os.path.join(PROJECT_ROOT, "train")
if train_dir not in sys.path:
    sys.path.insert(0, train_dir)

qwen_vl_path = os.path.join(PROJECT_ROOT, "reference", "Qwen2.5-VL", "qwen-vl-finetune")
if qwen_vl_path not in sys.path:
    sys.path.insert(0, qwen_vl_path)


# Training-side classes and helpers
from qwenvl.train.argument import DataArguments
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Qwen2VLImageProcessor,
)

from qwenvl.dataset.data_qwen_vla import preprocess_qwen_2_visual_vla_sources
from qwenvl.dataset.rope2d import get_rope_index_25, get_rope_index_2

# Token utils
sys.path.append(os.path.join(qwen_vl_path, "qwenvl", "utils"))
from token_utils import smart_load_model_and_tokenizer, prepare_action_tokenizer_mapping
from transformers import AutoProcessor as HF_AutoProcessor


class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: List[int]):
        self.allowed_token_ids = torch.tensor(sorted(set(allowed_token_ids)), dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: (batch_size, vocab_size)
        if scores.ndim == 2:
            mask = torch.full_like(scores, fill_value=float('-inf'))
            mask[:, self.allowed_token_ids.to(scores.device)] = 0.0
            scores = scores + mask
        else:
            raise ValueError("Unexpected scores ndim for logits processor")
        return scores


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL VLA inference")
    # Required key paths (avoid hard-coded defaults)
    parser.add_argument("--qwen_hub", type=str, required=True, help="Path to fine-tuned Qwen-VL hub (or HF id)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference results")
    parser.add_argument("--test_data_pkl", type=str, required=True, help="Path to test meta pickle file")
    parser.add_argument("--action_tokenizer_path", type=str, required=True, help="Path to FAST action tokenizer folder")
    parser.add_argument("--token_yaml", type=str, required=True, help="YAML with token names for output files")
    parser.add_argument("--norm_stats_path", type=str, required=True, help="Path to normalization stats JSON for de-normalization")
    parser.add_argument("--raw_img_root", type=str, required=True, help="Path to raw image root")
    # Optional configs with reasonable defaults
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cur_frame_idx", type=int, default=3)
    parser.add_argument("--use_previous_actions", action="store_true")
    parser.add_argument("--future_nums", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--model_max_length", type=int, default=1400)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_gt", action="store_true", help="Also save de-normalized ground-truth actions if available")
    # Optional path prefix replacement to adapt dataset absolute paths
    parser.add_argument("--path_replace_from", type=str, default="",
                        help="If non-empty, replace this prefix in any file path found in the dataset")
    parser.add_argument("--path_replace_to", type=str, default="",
                        help="Replacement prefix for any matched file path in the dataset")
    parser.add_argument("--image_cam_subdir", type=str, default="CAM_F0",
                        help="If set, try to open images from <parent>/<subdir>/<basename> first (e.g., CAM_F0)")
    # Align image sizing with training defaults
    parser.add_argument("--max_pixels", type=int, default=28 * 28 * 576,
                        help="Training-time longest_edge pixels for Qwen image processor")
    parser.add_argument("--min_pixels", type=int, default=28 * 28 * 16,
                        help="Training-time shortest_edge pixels for Qwen image processor")
    return parser.parse_args()


def setup_distributed() -> Tuple[int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def load_test_data(pkl_path: str) -> List[Dict[str, Any]]:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def build_image_inputs(image_processor, image_paths: List[str], cur_idx: int, cam_subdir: str, raw_img_root: str):
    from PIL import Image
    import copy
    # Choose image path by cur_idx or fallback to first
    if isinstance(image_paths, str):
        image_file = image_paths
    elif isinstance(image_paths, list) and len(image_paths) > 0:
        image_file = image_paths[cur_idx] if cur_idx < len(image_paths) else image_paths[0]
    else:
        image_file = None

    if image_file is None:
        return None, None, None

    # If requested, try to map to subdir path: dir/CAM_F0/basename
    if cam_subdir:
        log_id, token_id = image_file.split("/")[-2:]
        token_id = token_id[:-4] + ".jpg"
        candidate = os.path.join(raw_img_root, log_id, cam_subdir, token_id)
        if os.path.exists(candidate):
            image_file = candidate

    # Guard: only try to open typical image files
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    ext = os.path.splitext(str(image_file))[-1].lower()
    if ext not in valid_exts:
        return None, None, None

    processor = copy.deepcopy(image_processor)
    image = Image.open(image_file).convert("RGB")
    visual_processed = processor.preprocess(image, return_tensors="pt")
    image_tensor = visual_processed["pixel_values"]
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    grid_thw = visual_processed["image_grid_thw"][0]

    # merged grid for prompt replacement
    grid_thw_merged = grid_thw
    if not isinstance(grid_thw_merged, list):
        grid_thw_merged = [grid_thw_merged]
    grid_thw_merged = [thw.prod() // image_processor.merge_size ** 2 for thw in grid_thw_merged]

    return image_tensor, grid_thw, grid_thw_merged


def make_generation_prompt_and_position_ids(
    tokenizer,
    prompt_text: str,
    grid_thw_image_merged: List[int],
    user_action_token_ids: torch.Tensor,
    get_rope_index_fn,
    image_processor,
    image_grid_thw: torch.Tensor,
) -> Dict[str, Any]:
    # Ensure chat_template identical to training preprocess
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Build sources: user then assistant (empty)
    sources = [[
        {"from": "user", "value": prompt_text},
        {"from": "assistant", "value": ""},
    ]]

    # Get base input_ids via preprocess (no assistant actions)
    data_dict = preprocess_qwen_2_visual_vla_sources(
        sources,
        tokenizer,
        grid_thw_image=grid_thw_image_merged,
        grid_thw_video=None,
        action_token_ids=None,
        user_action_token_ids=user_action_token_ids,
    )

    input_ids: torch.Tensor = data_dict["input_ids"][0]  # (seq,)
    # Remove the last two tokens ("<|im_end|>", "\n") to position at assistant content start
    prefix = input_ids[:-2]
    # Let model compute position_ids internally for robustness
    return {"prompt_ids": prefix}


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    # Build data_args consistent with training
    data_args = DataArguments(
        data_path=args.test_data_pkl,
        use_actions=True,
        actions_format="fast",
        action_tokenizer_path=args.action_tokenizer_path,
        action_dim=args.action_dim,
        use_previous_actions=args.use_previous_actions,
        cur_frame_idx=args.cur_frame_idx,
        future_nums=args.future_nums,
    )

    # Load model + tokenizer with VLA tokens
    model_class = Qwen2_5_VLForConditionalGeneration
    model_type = "qwen2.5vl"

    torch_dtype = torch.bfloat16 if args.bf16 else None

    model, tokenizer, _ = smart_load_model_and_tokenizer(
        model_name_or_path=args.qwen_hub,
        model_class=model_class,
        add_vla_tokens=False,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        device_map=None,
    )

    image_processor = AutoProcessor.from_pretrained(args.qwen_hub).image_processor

    tokenizer.model_max_length = args.model_max_length
    tokenizer.padding_side = "right"

    model = model.to(device).eval()

    # Action tokenizer for decode/encode mapping
    action_tokenizer = AutoProcessor.from_pretrained(args.action_tokenizer_path, trust_remote_code=True)

    # Prepare mapping for action range and special tokens
    mapping = prepare_action_tokenizer_mapping(tokenizer)
    boa_token_id = mapping["boa_token_id"]
    eoa_token_id = mapping["eoa_token_id"]
    last_vocab_idx = mapping["last_vocab_idx"]

    # Allow only the reverse-mapped FAST id range and EOA
    start_id = last_vocab_idx - (action_tokenizer.vocab_size - 1)
    end_id = last_vocab_idx
    allowed_action_ids = list(range(start_id, end_id + 1))
    allowed_token_ids = allowed_action_ids + [eoa_token_id]

    logits_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)

    # Load test data and token names
    data = load_test_data(args.test_data_pkl)
    # Apply optional path prefix replacement across the loaded data
    def _replace_prefix(obj):
        if isinstance(obj, str) and args.path_replace_from and obj.startswith(args.path_replace_from):
            return args.path_replace_to + obj[len(args.path_replace_from):]
        elif isinstance(obj, list):
            return [_replace_prefix(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(_replace_prefix(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: _replace_prefix(v) for k, v in obj.items()}
        else:
            return obj
    data = [_replace_prefix(scene) for scene in data]
    with open(args.token_yaml, 'r') as f:
        token_list = yaml.safe_load(f)['tokens']

    # Normalization stats for de-normalization
    norm_cfg = json.load(open(args.norm_stats_path, 'r'))
    # The keys align with Emu3 example; adapt if needed
    action_low = torch.tensor(norm_cfg['norm_stats']['libero']['q01'], device=device, dtype=torch.float32)
    action_high = torch.tensor(norm_cfg['norm_stats']['libero']['q99'], device=device, dtype=torch.float32)

    os.makedirs(args.output_dir, exist_ok=True)

    # Rank-sharded iteration
    indices = list(range(len(data)))
    shard = indices[rank::world_size]

    pbar = tqdm(shard, total=len(shard), desc=f"Rank {rank}", dynamic_ncols=True, leave=False)
    for idx in pbar:
        scene = data[idx]

        # Prompt text
        if "text" in scene:
            if isinstance(scene["text"], list):
                prompt = scene["text"][args.cur_frame_idx] if args.cur_frame_idx < len(scene["text"]) else scene["text"][0]
            else:
                prompt = scene["text"]
        else:
            prompt = "Describe the driving scene."
            
        if "pre_1s_text" in scene:
            if isinstance(scene["pre_1s_text"], list):
                pre_prompt = scene["pre_1s_text"][args.cur_frame_idx] if args.cur_frame_idx < len(scene["pre_1s_text"]) else scene["pre_1s_text"][0]
            else:
                pre_prompt = scene["pre_1s_text"]
        else:
            pre_prompt = prompt

        # Image inputs
        image_tensor, image_grid_thw, grid_thw_merged = (None, None, None)
        if "image" in scene:
            image_tensor, image_grid_thw, grid_thw_merged = build_image_inputs(
                image_processor, scene["image"], args.cur_frame_idx, args.image_cam_subdir, args.raw_img_root
            )
        pre_image_tensor, pre_grid_thw, pre_grid_thw_merged = (None, None, None)
        if "pre_1s_image" in scene:
            pre_image_tensor, pre_grid_thw, pre_grid_thw_merged = build_image_inputs(
                image_processor, scene["pre_1s_image"], args.cur_frame_idx, args.image_cam_subdir, args.raw_img_root
            )

        # User historical actions -> user_action_token_ids (not wrapped)
        pre_user_action_token_ids = None
        pre_wrapped_action_token_ids = None

        if data_args.use_previous_actions and "pre_1s_action" in scene:
            pre_actions = scene["pre_1s_action"]
            if len(pre_actions) > 0 and args.cur_frame_idx > 0:
                pre_hist_actions = pre_actions[:args.cur_frame_idx]
                pre_fut_actions = pre_actions[args.cur_frame_idx:args.cur_frame_idx + 2]
                if isinstance(pre_hist_actions, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in pre_hist_actions]
                    pre_action_tokens = torch.cat(tensor_list, dim=0)
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in pre_fut_actions]
                    pre_fut_action_tokens = torch.cat(tensor_list, dim=0)
                else:
                    pre_action_tokens = torch.tensor(pre_hist_actions) if not torch.is_tensor(pre_hist_actions) else pre_hist_actions
                    pre_fut_action_tokens = torch.tensor(pre_fut_actions) if not torch.is_tensor(pre_fut_actions) else pre_fut_actions
                ids = action_tokenizer(pre_action_tokens)[0]
                mapped_ids = [last_vocab_idx - i for i in ids]
                pre_user_action_token_ids = torch.tensor(mapped_ids, dtype=torch.long)
                ids = action_tokenizer(pre_fut_action_tokens)[0]
                mapped_ids = [boa_token_id] + [last_vocab_idx - i for i in ids] + [eoa_token_id]
                pre_wrapped_action_token_ids = torch.tensor(mapped_ids, dtype=torch.long)

        user_action_token_ids = None
        if data_args.use_previous_actions and "action" in scene:
            actions = scene["action"]
            if len(actions) > 0 and args.cur_frame_idx > 0:
                hist_actions = actions[:args.cur_frame_idx]
                if isinstance(hist_actions, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in hist_actions]
                    action_tokens = torch.cat(tensor_list, dim=0)
                else:
                    action_tokens = torch.tensor(hist_actions) if not torch.is_tensor(hist_actions) else hist_actions
                ids = action_tokenizer(action_tokens)[0]
                mapped_ids = [last_vocab_idx - i for i in ids]
                user_action_token_ids = torch.tensor(mapped_ids, dtype=torch.long)


        # Build user content with <image> tag only if image exists
        user_content = [[
            {"from": "user", "value": f"<image>{pre_prompt}"},
            {"from": "assistant", "value": ""},
            {"from": "user", "value": f"<image>{prompt}"},
            {"from": "assistant", "value": ""},
        ]]

        grid_thw_image_flat = []
        grid_thw_image_flat.extend(pre_grid_thw_merged)
        grid_thw_image_flat.extend(grid_thw_merged)

        action_segments = [pre_user_action_token_ids, pre_wrapped_action_token_ids, user_action_token_ids]
        action_roles = ["user", "assistant", "user"]
        
        # Prompt ids and position ids
        data_dict = preprocess_qwen_2_visual_vla_sources(
            sources=user_content,
            tokenizer=tokenizer,
            grid_thw_image=grid_thw_image_flat,
            action_segments=action_segments if len(action_segments) > 0 else None,
            action_roles=action_roles if len(action_roles) > 0 else None,
        )   
        input_ids: torch.Tensor = data_dict["input_ids"][0]  # (seq,)
        # Remove the last two tokens ("<|im_end|>", "\n") to position at assistant content start
        prefix = input_ids[:-2]
        # Let model compute position_ids internally for robustness
        prompt_ids: torch.Tensor = prefix.to(device)  # (seq,)

        # Append BOA token to start action generation
        input_ids = torch.cat([prompt_ids, torch.tensor([boa_token_id], device=device, dtype=torch.long)], dim=0).unsqueeze(0)

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        model_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
         # Add image tensors
        all_images = [pre_image_tensor, image_tensor]
        all_image_grid_thw = [pre_grid_thw.unsqueeze(0), image_grid_thw.unsqueeze(0)]
        
        model_inputs["pixel_values"] = torch.cat(all_images, dim=0).to(device)
        model_inputs["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0).to(device)

        # Generate until EOA
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=50,
                eos_token_id=[eoa_token_id],
                logits_processor=[logits_processor],
            )

        # Slice new tokens
        gen_tokens = outputs[0][input_ids.size(1):]

        # Remove EOA and any BOA remnants; map back to action tokenizer ids
        filtered = [t.item() for t in gen_tokens if t.item() not in (boa_token_id, eoa_token_id)]
        action_tok_ids = [last_vocab_idx - t for t in filtered]

        # print(f"action_tok_ids:{action_tok_ids}")

        # Decode to continuous actions
        # Expect shape: (time_horizon, action_dim)
        decoded_actions = action_tokenizer.decode(
            [action_tok_ids],
            time_horizon=data_args.future_nums,
            action_dim=data_args.action_dim,
        )[0]

        pred = torch.tensor(decoded_actions, dtype=torch.float32, device=device)

        # De-normalize: x = 0.5 * (z + 1) * (high - low) + low
        action_denorm = 0.5 * (pred + 1.0) * (action_high - action_low) + action_low

        # Save
        token_name = token_list[idx]
        output_dict = {"action": action_denorm.detach().cpu().numpy().tolist()}
        # Optional: include GT for debugging if available and requested
        if args.save_gt and "action" in scene and scene["action"] is not None:
            gt = scene["action"]
            gt_tensor = torch.tensor(gt, dtype=torch.float32, device=device)
            gt_denorm = 0.5 * (gt_tensor + 1.0) * (action_high - action_low) + action_low
            output_dict["action_gt_denorm"] = gt_denorm.detach().cpu().numpy().tolist()

        out_path = os.path.join(args.output_dir, f"{token_name}.json")
        with open(out_path, "w") as f:
            json.dump(output_dict, f, indent=2)


    dist.barrier()
    if rank == 0:
        print(f"Inference complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()