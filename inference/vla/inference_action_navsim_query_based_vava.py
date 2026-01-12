import os, sys, yaml, json, pickle, argparse
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 导入配置模块并立即设置路径（必须在导入 emu3 之前）
from config import setup_paths_early
setup_paths_early()

from transformers import AutoProcessor
from emu3.mllm import Emu3Tokenizer, Emu3QFormer, Emu3Pi0Config
from utils.train_qformer import DataArguments # Import from training script
from datasets import Emu3DrivingVAVADataset # Import from training script

# 导入配置获取函数（用于在 main 中获取配置）
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Emu3QFormer with VAVA dataset format.")
    parser.add_argument("--emu_hub", required=True, type=str, help="Path to the trained Emu3QFormer model hub.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save inference results.")
    parser.add_argument("--test_data_pkl", required=True, type=str, help="Path to the test data meta pickle file.")
    parser.add_argument("--token_yaml", type=str, default=None, help="Path to the token YAML file for naming outputs.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of dataloader workers.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file.")
    return parser.parse_args()


def setup_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def main():
    args = parse_args()
    
    # 加载配置
    config = get_config(args.config)
    
    rank, world_size = setup_distributed()
    device = f"cuda:{rank}"

    # ========================================================================================
    # 1. Create DataArguments to EXACTLY match training configuration
    #    Values are taken from train_navsim_qformer.sh and DataArguments defaults
    # ========================================================================================
    action_tokenizer_path = config.get_path("paths.action_tokenizer")
    if action_tokenizer_path is None:
        raise ValueError("action_tokenizer_path must be set via config file or VLA_ACTION_TOKENIZER environment variable")
    
    data_args = DataArguments(
        data_path=args.test_data_pkl,
        actions=True,
        driving=True,
        use_previous_actions=True,
        actions_format="fast",
        action_tokenizer_path=action_tokenizer_path,
        frames=config.get("data.frames", 1),
        action_frames=config.get("data.action_frames", 8),
        action_dim=config.get("data.action_dim", 3),
        cur_frame_idx=config.get("data.cur_frame_idx", 3),
        pre_action_frames=config.get("data.pre_action_frames", 3),
        video_format=None, # Inferred from dataset type
        use_flip=False # No flipping during inference
    )

    # ========================================================================================
    # 2. Load Tokenizer and Dataset (Identical to training)
    # ========================================================================================
    # This model path is used for tokenizer config, not model weights
    vlm_model_path_for_tokenizer = config.get_path("paths.vlm_model")
    if vlm_model_path_for_tokenizer is None:
        raise ValueError("vlm_model path must be set via config file or VLA_VLM_MODEL environment variable")
    
    tokenizer = Emu3Tokenizer.from_pretrained(
        vlm_model_path_for_tokenizer,
        model_max_length=config.get("model.model_max_length", 1400),
        padding_side=config.get("model.padding_side", "right"),
        use_fast=config.get("model.use_fast", False),
    )

    dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    
    # ========================================================================================
    # 3. Setup DataLoader with DistributedSampler
    # ========================================================================================
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    num_workers = args.num_workers if args.num_workers is not None else config.get("data.num_workers", 12)
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("data.batch_size", 1),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=config.get("data.pin_memory", True),
        collate_fn=getattr(dataset, 'collate_fn', None) # Use dataset's collate if available
    )

    # ========================================================================================
    # 4. Load Model
    # ========================================================================================
    model_config = Emu3Pi0Config.from_pretrained(os.path.join(args.emu_hub, "config.json"))
    
    # 获取 torch_dtype
    torch_dtype_str = config.get("model.torch_dtype", "bfloat16")
    torch_dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_map.get(torch_dtype_str, torch.bfloat16)
    
    model, loading_info = Emu3QFormer.from_pretrained(
        args.emu_hub,
        config=model_config,
        pretrain_vlm_path=vlm_model_path_for_tokenizer, # Must match tokenizer's
        attn_implementation=config.get("model.attn_implementation", "sdpa"),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        output_loading_info=True
    )
    if rank == 0:
        print("Missing keys:", loading_info["missing_keys"])
        print("Unexpected keys:", loading_info["unexpected_keys"])
        print("Mismatched sizes:", loading_info.get("mismatched_keys", "N/A"))
        
    model = model.to(device).eval()

    # ========================================================================================
    # 5. Load external metadata (token names for saving, norm stats)
    # ========================================================================================
    token_yaml_path = args.token_yaml if args.token_yaml else config.get_path("paths.token_yaml")
    if token_yaml_path is None:
        raise ValueError("token_yaml path must be set via config file, command line argument, or VLA_TOKEN_YAML environment variable")
    
    with open(token_yaml_path, 'r') as f:
        token_list = yaml.safe_load(f)['tokens']

    norm_stats_path = config.get_path("paths.norm_stats")
    if norm_stats_path is None:
        raise ValueError("norm_stats path must be set via config file or VLA_NORM_STATS environment variable")
    
    norm_cfg = json.load(open(norm_stats_path, 'r'))
    action_low = torch.tensor(norm_cfg['norm_stats']['libero']['q01'], device=device)
    action_high = torch.tensor(norm_cfg['norm_stats']['libero']['q99'], device=device)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================================================
    # 6. Main Inference Loop
    # ========================================================================================
    # Get the indices that this rank will process to map back to token names
    rank_indices = list(sampler)
    pbar = tqdm(zip(dataloader, rank_indices), total=len(rank_indices), desc=f"Rank {rank}", position=rank)
    
    for batch, original_idx in pbar:
        # The dataloader now returns a collated batch dictionary
        # Move all tensors in the batch to the correct device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        # Ground truth is also in the batch, useful for debugging
        gt_action = batch.get("action")

        # Generate action using the model
        with torch.no_grad():
            predicted_action = model.sample_actions(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pre_action=batch["pre_action"].to(torch.bfloat16),
                cmd=batch["cmd"].to(torch.bfloat16),
                action_frames=data_args.action_frames,
                action_dim=data_args.action_dim,
            )

        # De-normalize the predicted action and remove batch dimension
        # action_denorm = 0.5 * (predicted_action.squeeze(0) + 1) * (action_high - action_low) + action_low
        action_denorm = predicted_action.squeeze(0) # [8, 3]
        
        # Prepare data for saving for the single item in the batch
        token_name = token_list[original_idx]
        
        output_dict = {
            "action": action_denorm.float().cpu().numpy().tolist()
        }

        # For debugging, save the ground truth action as well
        if gt_action is not None:
            gt_action_denorm = 0.5 * (gt_action.squeeze(0) + 1) * (action_high - action_low) + action_low
            output_dict["action_gt_denorm"] = gt_action_denorm.cpu().numpy().tolist()
        
        output_path = os.path.join(args.output_dir, f"{token_name}.json")
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=4)
    
    dist.barrier()
    if rank == 0:
        print(f"\nInference complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
