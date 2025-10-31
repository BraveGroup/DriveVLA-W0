#!/usr/bin/env python3
"""
NuPlan Pickle合并脚本 - Step 4
合并所有数据并生成最终pickle，包含归一化
"""

import os
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm

# 添加路径
sys.path.append("/mnt/vdb1/yingyan.li/repo/OmniSim")
from train.dataset.normalize_pi0 import RunningStats, save

# 配置
INPUT_FILE = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/video_segments.json"
ACTIONS_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/actions"
COMMANDS_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/commands"
OUTPUT_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/output"
NORMALIZER_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/configs/normalizer_nuplan_01"

def load_segment_data(segment_info):
    """加载单个段的所有数据"""
    segment_id = segment_info['segment_id']
    
    # 加载images paths (VQ codes) - 修复键名
    vq_paths = segment_info.get('npy_paths', segment_info.get('vq_paths', []))
    
    # 加载actions
    action_file = os.path.join(ACTIONS_DIR, f"{segment_id}.npy")
    if not os.path.exists(action_file):
        return None
    actions = np.load(action_file)  # shape: (frames, 8, 3)
    
    # 加载commands
    command_file = os.path.join(COMMANDS_DIR, f"{segment_id}.npy")
    if not os.path.exists(command_file):
        return None
    commands = np.load(command_file).tolist()  # list of strings
    
    # 验证数据一致性
    if len(vq_paths) != len(actions) or len(actions) != len(commands):
        print(f"❌ 数据长度不一致: {segment_id}")
        return None
    
    return {
        "segment_id": segment_id,
        "images": vq_paths,
        "actions": actions,
        "commands": commands
    }

def collect_all_actions(segments_data):
    """收集所有action数据用于归一化"""
    all_actions = []
    for segment_data in segments_data:
        actions = segment_data['actions']  # shape: (frames, 8, 3)
        # 展平为2D: (frames*8, 3)
        flattened = actions.reshape(-1, 3)
        all_actions.append(flattened)
    
    return np.concatenate(all_actions, axis=0)  # shape: (total_deltas, 3)

def normalize_actions(segments_data, norm_stats):
    """归一化所有action数据"""
    for segment_data in segments_data:
        actions = segment_data['actions'].copy()  # shape: (frames, 8, 3)
        
        # 归一化公式: 2 * (x - q01) / (q99 - q01) - 1
        normalized = 2 * (actions - norm_stats.q01) / (norm_stats.q99 - norm_stats.q01 + 1e-8) - 1
        segment_data['actions'] = np.clip(normalized, -1, 1)

def main():
    print("🚀 开始合并Pickle数据...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(NORMALIZER_DIR, exist_ok=True)
    
    # 加载视频段信息
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    segments = data['segments']
    print(f"📁 待处理段数: {len(segments)}")
    
    # 加载所有段数据
    segments_data = []
    failed_count = 0
    
    for segment_info in tqdm(segments, desc="加载数据"):
        segment_data = load_segment_data(segment_info)
        if segment_data is not None:
            segments_data.append(segment_data)
        else:
            failed_count += 1
    
    print(f"✅ 成功加载: {len(segments_data)} 段")
    print(f"❌ 失败: {failed_count} 段")
    
    if not segments_data:
        print("❌ 没有有效数据！")
        return
    
    # 计算归一化统计
    print("📊 计算归一化统计...")
    action_data = collect_all_actions(segments_data)
    print(f"Action数据形状: {action_data.shape}")
    
    normalizer = RunningStats()
    normalizer.update(action_data)
    norm_stats = normalizer.get_statistics()
    
    print(f"Mean: {norm_stats.mean}")
    print(f"Std: {norm_stats.std}")
    print(f"Q01: {norm_stats.q01}")
    print(f"Q99: {norm_stats.q99}")
    
    # 保存归一化参数
    norm_stats_save = {"libero": norm_stats}  # 保持与现有系统一致
    save(NORMALIZER_DIR, norm_stats_save)
    print(f"💾 归一化参数保存到: {NORMALIZER_DIR}")
    
    # 归一化actions
    print("🔄 应用归一化...")
    normalize_actions(segments_data, norm_stats)
    
    # 生成最终pickle格式
    result_file = []
    for segment_data in segments_data:
        result_file.append({
            "segment_id": segment_data["segment_id"],
            "image": segment_data["images"],
            "action": segment_data["actions"],
            "text": segment_data["commands"]
        })
    
    # 保存pickle文件
    output_file = os.path.join(OUTPUT_DIR, "nuplan_processed_data_01.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(result_file, f)
    
    # 统计信息
    total_frames = sum(len(seg["image"]) for seg in result_file)
    avg_frames = total_frames / len(result_file)
    
    print(f"\n📊 最终结果:")
    print(f"总段数: {len(result_file)}")
    print(f"总帧数: {total_frames}")
    print(f"平均每段帧数: {avg_frames:.1f}")
    
    # Command分布统计
    command_counts = {}
    for segment_data in result_file:
        for cmd in segment_data["text"]:
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
    
    print(f"\n📊 Command分布:")
    for cmd, count in command_counts.items():
        percentage = count / total_frames * 100
        print(f"{cmd}: {count} ({percentage:.1f}%)")
    
    print(f"\n💾 最终文件保存到: {output_file}")
    print("✅ Step 4 完成！")

if __name__ == "__main__":
    main() 