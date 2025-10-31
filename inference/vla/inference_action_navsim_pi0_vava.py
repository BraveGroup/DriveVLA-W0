import os, sys, yaml, json, pickle, argparse
import torch
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# project root
project_root = "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei"

# Add the 'train' directory for importing DataArguments
train_dir = os.path.join(project_root, 'train')
if train_dir not in sys.path:
    sys.path.insert(0, train_dir)

# Add the 'reference/Emu3' directory for Emu3 models
emu3_path = os.path.join(project_root, "reference", "Emu3")
if emu3_path not in sys.path:
    sys.path.insert(0, emu3_path)

from transformers import AutoProcessor
from emu3.mllm import Emu3Tokenizer, Emu3Pi0, Emu3Pi0Config
from train_pi0 import DataArguments # Import from training script
from datasets import Emu3DrivingVAVADataset # Import from training script


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Emu3Pi0 with VAVA dataset format.")
    parser.add_argument("--emu_hub", required=True, type=str, help="Path to the trained Emu3Pi0 model hub.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save inference results.")
    parser.add_argument("--test_data_pkl", required=True, type=str, help="Path to the test data meta pickle file.")
    parser.add_argument("--token_yaml", type=str, default="data/navsim/processed_data/scene_files/scene_filter/navtest.yaml"
                        , help="Path to the token YAML file for naming outputs.")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of Pi0 inference steps.")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of dataloader workers.")
    return parser.parse_args()


def setup_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def main():
    args = parse_args()
    rank, world_size = setup_distributed()
    device = f"cuda:{rank}"

    # ========================================================================================
    # 1. Create DataArguments to EXACTLY match training configuration
    # ========================================================================================
    data_args = DataArguments(
        data_path=args.test_data_pkl,
        actions=True,
        driving=True,
        use_previous_actions=True,
        actions_format="fast",
        action_tokenizer_path="/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/pretrained_models/fast",
        frames=1,
        action_frames=8,
        action_dim=3,
        cur_frame_idx=3,
        pre_action_frames=3,
        video_format=None,
        use_flip=False
    )

    # ========================================================================================
    # 2. Load Tokenizer and Dataset (Identical to training)
    # ========================================================================================
    vlm_model_path_for_tokenizer = "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/logs/train_nuplan_6va_v0.2_multi_node"
    tokenizer = Emu3Tokenizer.from_pretrained(
        vlm_model_path_for_tokenizer,
        model_max_length=1400,
        padding_side="right",
        use_fast=False,
    )

    dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    
    # ========================================================================================
    # 3. Setup DataLoader with DistributedSampler
    # ========================================================================================
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1, # Hardcoded batch size
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=getattr(dataset, 'collate_fn', None)
    )

    # ========================================================================================
    # 4. Load Model
    # ========================================================================================
    model_config = Emu3Pi0Config.from_pretrained(os.path.join(args.emu_hub, "config.json"))
    model, loading_info = Emu3Pi0.from_pretrained(
        args.emu_hub,
        config=model_config,
        pretrain_vlm_path=vlm_model_path_for_tokenizer,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
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
    with open(args.token_yaml, 'r') as f:
        token_list = yaml.safe_load(f)['tokens']

    norm_stats_path = "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/configs/normalizer_navsim_trainval/norm_stats.json"
    norm_cfg = json.load(open(norm_stats_path, 'r'))
    action_low = torch.tensor(norm_cfg['norm_stats']['libero']['q01'], device=device)
    action_high = torch.tensor(norm_cfg['norm_stats']['libero']['q99'], device=device)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================================================
    # 6. Main Inference Loop
    # ========================================================================================
    rank_indices = list(sampler)
    pbar = tqdm(zip(dataloader, rank_indices), total=len(rank_indices), desc=f"Rank {rank}", position=rank, disable=(rank!=0))
    
    for batch, original_idx in pbar:
        # Move all tensors in the batch to the correct device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        gt_action = batch.get("action")

        with torch.no_grad():
            predicted_action = model.sample_actions(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pre_action=batch["pre_action"].to(torch.bfloat16),
                cmd=batch["cmd"].to(torch.bfloat16),
                num_inference_steps=args.num_inference_steps,
                action_frames=data_args.action_frames,
                action_dim=data_args.action_dim,
            )

        # De-normalize the predicted action and remove batch dimension
        action_denorm = 0.5 * (predicted_action.squeeze(0) + 1) * (action_high - action_low) + action_low
        
        token_name = token_list[original_idx]
        
        output_dict = {
            "action": action_denorm.cpu().numpy().tolist()
        }

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
