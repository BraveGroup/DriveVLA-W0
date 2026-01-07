import os, sys, yaml, json, pickle, argparse
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# project root
project_root = "/mnt/nvme0n1p1/yingyan.li/repo/DriveVLA-W0"

# Add the 'train' directory for importing DataArguments
train_dir = os.path.join(project_root, 'train')
if train_dir not in sys.path:
    sys.path.insert(0, train_dir)

# Add the 'reference/Emu3' directory for Emu3 models
emu3_path = os.path.join(project_root, "reference", "Emu3")
if emu3_path not in sys.path:
    sys.path.insert(0, emu3_path)

from transformers import AutoProcessor
from emu3.mllm import Emu3Tokenizer, Emu3QFormer, Emu3Pi0Config
from train_qformer import DataArguments # Import from training script
from datasets import Emu3DrivingVAVADataset # Import from training script


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Emu3QFormer with VAVA dataset format.")
    parser.add_argument("--emu_hub", required=True, type=str, help="Path to the trained Emu3QFormer model hub.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save inference results.")
    parser.add_argument("--test_data_pkl", required=True, type=str, help="Path to the test data meta pickle file.")
    parser.add_argument("--token_yaml", type=str, default="data/navsim/processed_data/scene_files/scene_filter/navtest.yaml"
                        ,help="Path to the token YAML file for naming outputs.")
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
    #    Values are taken from train_navsim_qformer.sh and DataArguments defaults
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
        video_format=None, # Inferred from dataset type
        use_flip=False # No flipping during inference
    )

    # ========================================================================================
    # 2. Load Tokenizer and Dataset (Identical to training)
    # ========================================================================================
    # This model path is used for tokenizer config, not model weights
    vlm_model_path_for_tokenizer = "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/logs/train_nuplan_6va_v0.2_multi_node"
    tokenizer = Emu3Tokenizer.from_pretrained(
        vlm_model_path_for_tokenizer,
        model_max_length=1400, # From train script
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
        batch_size=1,
        sampler=sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=getattr(dataset, 'collate_fn', None) # Use dataset's collate if available
    )

    # ========================================================================================
    # 4. Load Model
    # ========================================================================================
    model_config = Emu3Pi0Config.from_pretrained(os.path.join(args.emu_hub, "config.json"))
    model, loading_info = Emu3QFormer.from_pretrained(
        args.emu_hub,
        config=model_config,
        pretrain_vlm_path=vlm_model_path_for_tokenizer, # Must match tokenizer's
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
