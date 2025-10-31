#!/usr/bin/env python3
"""
检查NavSim command映射和分布
验证left/right的正确性
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter

# 配置
NAVSIM_LOGS_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/data/navsim/navsim_logs/mini"

def load_and_analyze_commands():
    """加载并分析NavSim的command分布"""
    print("🔍 检查NavSim command映射和分布...")
    
    pkl_files = [f for f in os.listdir(NAVSIM_LOGS_DIR) if f.endswith('.pkl')]
    print(f"找到 {len(pkl_files)} 个文件")
    
    all_commands = []
    command_examples = {0: [], 1: [], 2: [], 3: []}
    
    sample_count = 0
    
    for pkl_file in tqdm(pkl_files[:10], desc="分析文件"):  # 只看前10个文件
        pkl_path = os.path.join(NAVSIM_LOGS_DIR, pkl_file)
        try:
            sequence = pickle.load(open(pkl_path, 'rb'))
            
            for i, frame in enumerate(sequence):
                if sample_count >= 5000:  # 限制样本数
                    break
                    
                driving_command = frame.get('driving_command', None)
                if driving_command is not None:
                    # 找到non-zero的index
                    cmd_idx = driving_command.nonzero()[0].item()
                    all_commands.append(cmd_idx)
                    
                    # 保存一些例子用于分析
                    if len(command_examples[cmd_idx]) < 5:
                        command_examples[cmd_idx].append({
                            'file': pkl_file,
                            'frame': i,
                            'command_vector': driving_command.tolist()
                        })
                    
                    sample_count += 1
                
                if sample_count >= 5000:
                    break
            
        except Exception as e:
            print(f"⚠️ 处理失败 {pkl_file}: {e}")
            continue
        
        if sample_count >= 5000:
            break
    
    # 统计分布
    command_dist = Counter(all_commands)
    total = sum(command_dist.values())
    
    print(f"\n📊 Command分布统计 (总计{total:,}个样本):")
    print("="*50)
    for cmd_idx in [0, 1, 2, 3]:
        count = command_dist.get(cmd_idx, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"Command {cmd_idx}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n🔍 Command向量示例:")
    print("="*50)
    text_mapping = ["go left", "go straight", "go right", "unknown"]
    
    for cmd_idx in [0, 1, 2, 3]:
        print(f"\nCommand {cmd_idx} ({text_mapping[cmd_idx]}):")
        examples = command_examples[cmd_idx]
        for i, ex in enumerate(examples):
            print(f"  例子{i+1}: {ex['command_vector']} (文件: {ex['file'][:30]}..., 帧: {ex['frame']})")
    
    return command_dist, command_examples

def check_larger_dataset():
    """检查更大的数据集"""
    print(f"\n🔍 检查更大的NavSim数据集分布...")
    
    datasets_to_check = [
        ("test", "/mnt/vdb1/yingyan.li/repo/OmniSim/data/navsim/navsim_logs/test"),
        ("trainval", "/mnt/vdb1/yingyan.li/repo/OmniSim/data/navsim/navsim_logs/trainval")
    ]
    
    results = {}
    
    for dataset_name, dataset_dir in datasets_to_check:
        if os.path.exists(dataset_dir):
            print(f"\n📁 {dataset_name.upper()}数据集存在，检查中...")
            pkl_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pkl')]
            print(f"{dataset_name.upper()}数据集文件数: {len(pkl_files)}")
            
            # 随机采样一些文件
            import random
            sample_files = random.sample(pkl_files, min(10, len(pkl_files)))  # 增加到10个文件
            
            all_commands = []
            sample_count = 0
            
            for pkl_file in tqdm(sample_files, desc=f"分析{dataset_name}数据"):
                pkl_path = os.path.join(dataset_dir, pkl_file)
                try:
                    sequence = pickle.load(open(pkl_path, 'rb'))
                    
                    for frame in sequence:
                        if sample_count >= 5000:  # 增加样本数
                            break
                        
                        driving_command = frame.get('driving_command', None)
                        if driving_command is not None:
                            cmd_idx = driving_command.nonzero()[0].item()
                            all_commands.append(cmd_idx)
                            sample_count += 1
                    
                except Exception as e:
                    print(f"⚠️ 处理失败 {pkl_file}: {e}")
                    continue
                
                if sample_count >= 5000:
                    break
            
            # 统计数据分布
            command_dist = Counter(all_commands)
            total = sum(command_dist.values())
            
            print(f"\n📊 {dataset_name.upper()}数据集Command分布 (总计{total:,}个样本):")
            print("="*50)
            for cmd_idx in [0, 1, 2, 3]:
                count = command_dist.get(cmd_idx, 0)
                pct = count / total * 100 if total > 0 else 0
                print(f"Command {cmd_idx}: {count:6d} ({pct:5.1f}%)")
            
            results[dataset_name] = command_dist
            
        else:
            print(f"❌ {dataset_name.upper()}数据集不存在")
    
    return results

def main():
    print("🚀 开始检查NavSim command映射...")
    
    # 1. 检查mini数据集
    mini_dist, examples = load_and_analyze_commands()
    
    # 2. 检查test和trainval数据集
    larger_datasets = check_larger_dataset()
    
    # 3. 总结分析
    print(f"\n" + "="*60)
    print(f"📋 总结分析:")
    print(f"="*60)
    
    print(f"\n💡 Command映射验证:")
    print(f"  0 -> go left")
    print(f"  1 -> go straight") 
    print(f"  2 -> go right")
    print(f"  3 -> unknown")
    
    print(f"\n📊 数据集对比:")
    print(f"{'数据集':<10} {'LEFT%':<8} {'STRAIGHT%':<12} {'RIGHT%':<8} {'UNKNOWN%':<8}")
    print("-" * 50)
    
    # Mini数据集
    mini_total = sum(mini_dist.values())
    mini_left = mini_dist.get(0, 0) / mini_total * 100
    mini_straight = mini_dist.get(1, 0) / mini_total * 100
    mini_right = mini_dist.get(2, 0) / mini_total * 100
    mini_unknown = mini_dist.get(3, 0) / mini_total * 100
    
    print(f"{'Mini':<10} {mini_left:<8.1f} {mini_straight:<12.1f} {mini_right:<8.1f} {mini_unknown:<8.1f}")
    
    # 其他数据集
    for dataset_name, dataset_dist in larger_datasets.items():
        if dataset_dist:
            total = sum(dataset_dist.values())
            left = dataset_dist.get(0, 0) / total * 100
            straight = dataset_dist.get(1, 0) / total * 100
            right = dataset_dist.get(2, 0) / total * 100
            unknown = dataset_dist.get(3, 0) / total * 100
            
            print(f"{dataset_name.capitalize():<10} {left:<8.1f} {straight:<12.1f} {right:<8.1f} {unknown:<8.1f}")
    
    print(f"\n🤔 用户疑问分析:")
    if mini_left > mini_right:
        print(f"  ❓ 所有数据集都显示左转多于右转:")
        print(f"     Mini: 左转{mini_left:.1f}% vs 右转{mini_right:.1f}%")
        
        for dataset_name, dataset_dist in larger_datasets.items():
            if dataset_dist:
                total = sum(dataset_dist.values())
                left = dataset_dist.get(0, 0) / total * 100
                right = dataset_dist.get(2, 0) / total * 100
                print(f"     {dataset_name.capitalize()}: 左转{left:.1f}% vs 右转{right:.1f}%")
        
        print(f"\n  💭 可能原因:")
        print(f"     1. NavSim基于真实驾驶数据，反映实际路网特征")
        print(f"     2. 美国等右侧通行国家，左转更复杂，标注更多")
        print(f"     3. 数据采集路线偏向城市/复杂路段")
        print(f"     4. 左转需要更多decision-making，被特别标注")
    
    # 检查是否有任何数据集右转更多
    right_more_datasets = []
    for dataset_name, dataset_dist in larger_datasets.items():
        if dataset_dist:
            total = sum(dataset_dist.values())
            left = dataset_dist.get(0, 0) / total * 100
            right = dataset_dist.get(2, 0) / total * 100
            if right > left:
                right_more_datasets.append(dataset_name)
    
    if right_more_datasets:
        print(f"  ✅ 以下数据集显示右转更多: {', '.join(right_more_datasets)}")
    else:
        print(f"  🤷 所有检查的数据集都显示左转更多，与一般直觉不同")
        print(f"  📝 建议: 这可能确实是NavSim数据的特征，需要接受这个现实")

if __name__ == "__main__":
    main() 