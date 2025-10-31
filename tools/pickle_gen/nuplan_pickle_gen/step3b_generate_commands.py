#!/usr/bin/env python3
"""
NuPlan Commands生成脚本 - Step 3B
使用Step 3A确定的阈值生成所有Commands
严格按照JSON顺序，为所有帧生成commands（包括最后的帧）
"""

import os
import json
import numpy as np
from tqdm import tqdm

# 配置
NUPLAN_JSON_DIR = "/mnt/vdb1/nuplan_json"
INPUT_FILE = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/video_segments.json"
THRESHOLD_FILE = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/command_threshold_analysis.json"
OUTPUT_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/commands"

COMMAND_TYPES = ["go left", "go straight", "go right", "unknown"]
FORWARD_DISTANCE = 20.0  # 前进20m时判断lateral movement

def load_threshold_config():
    """加载阈值配置"""
    try:
        with open(THRESHOLD_FILE, 'r') as f:
            data = json.load(f)
        threshold = data.get('selected_threshold', 2.0)
        print(f"📏 使用阈值: {threshold:.2f}m")
        return threshold, data
    except Exception as e:
        print(f"❌ 无法加载阈值配置: {e}")
        print(f"请先运行 Step 3A 确定阈值")
        return None, None

def load_poses_from_json(seq_name):
    """从JSON文件加载poses数据，严格按照JSON顺序"""
    json_file = os.path.join(NUPLAN_JSON_DIR, f"{seq_name}.json")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        poses = data.get('poses', [])
        print(f"📄 {seq_name}: JSON记录{len(poses)}个poses，按原始顺序处理")
        return poses
    except Exception as e:
        return []

def calculate_lateral_displacement_with_padding(poses, start_frame):
    """计算20m后的lateral displacement，不足时使用外推或填充"""
    if start_frame >= len(poses):
        return None
    
    current_pose = np.eye(4, dtype=np.float64)
    path_distance = 0.0
    last_position = np.array([0.0, 0.0])
    
    frame_idx = start_frame + 1
    
    while frame_idx < len(poses) and path_distance < FORWARD_DISTANCE:
        transform_matrix = np.array(poses[frame_idx], dtype=np.float64)
        current_pose = current_pose @ transform_matrix
        current_position = np.array([current_pose[0, 3], current_pose[1, 3]])
        distance_increment = np.linalg.norm(current_position - last_position)
        path_distance += distance_increment
        last_position = current_position
        frame_idx += 1
    
    # 处理不足20m的情况 - 更宽容的外推策略
    if path_distance < FORWARD_DISTANCE:
        if path_distance > 1.0:  # 降低阈值，更容易外推
            # 如果有一定的移动，进行外推
            if frame_idx > start_frame + 1:
                # 计算平均移动方向
                direction_sum = np.array([0.0, 0.0])
                valid_transforms = 0
                
                for i in range(start_frame + 1, min(frame_idx, len(poses))):
                    transform = np.array(poses[i], dtype=np.float64)
                    direction = np.array([transform[0, 3], transform[1, 3]])
                    if np.linalg.norm(direction) > 0.001:
                        direction_sum += direction
                        valid_transforms += 1
                
                if valid_transforms > 0:
                    avg_direction = direction_sum / valid_transforms
                    if np.linalg.norm(avg_direction) > 0.001:
                        avg_direction = avg_direction / np.linalg.norm(avg_direction)
                        remaining_distance = FORWARD_DISTANCE - path_distance
                        final_position = current_position + avg_direction * remaining_distance
                    else:
                        final_position = current_position
                else:
                    final_position = current_position
            else:
                final_position = current_position
        else:
            # 移动距离太短，假设继续直行
            final_position = np.array([FORWARD_DISTANCE, 0.0])
    else:
        final_position = current_position
    
    return final_position[1]

def calculate_command_from_lateral_displacement(lateral_displacement, threshold):
    """根据lateral displacement计算command"""
    if lateral_displacement is None:
        return "unknown"
    
    if lateral_displacement < -threshold:
        return "go left"
    elif lateral_displacement > threshold:
        return "go right"
    else:
        return "go straight"

def process_segment(segment_info, threshold):
    """处理单个视频段，为所有帧生成commands"""
    segment_id = segment_info['segment_id']
    frame_count = segment_info['frame_count']
    original_sequence = segment_info['original_sequence']
    start_frame = segment_info['start_frame']
    
    # 加载poses数据（严格按JSON顺序）
    poses = load_poses_from_json(original_sequence)
    if not poses:
        print(f"❌ poses文件不存在: {original_sequence}")
        return None
    
    # 为每帧生成command（包括所有帧，包括最后的帧）
    commands = []
    for i in range(frame_count):
        frame_idx = start_frame + i  # 在原始序列中的帧索引
        lateral_disp = calculate_lateral_displacement_with_padding(poses, frame_idx)
        command = calculate_command_from_lateral_displacement(lateral_disp, threshold)
        commands.append(command)
    
    # 保存到文件
    output_file = os.path.join(OUTPUT_DIR, f"{segment_id}.npy")
    np.save(output_file, np.array(commands))
    
    return len(commands)

def verify_command_generation(segments, threshold):
    """验证command生成结果"""
    print(f"\n🔍 验证Command生成结果...")
    
    total_commands = 0
    command_distribution = {cmd: 0 for cmd in COMMAND_TYPES}
    
    # 统计前100个segments的commands
    sample_segments = segments[:100]
    
    for segment_info in tqdm(sample_segments, desc="验证Commands"):
        segment_id = segment_info['segment_id']
        command_file = os.path.join(OUTPUT_DIR, f"{segment_id}.npy")
        
        if os.path.exists(command_file):
            try:
                commands = np.load(command_file)
                total_commands += len(commands)
                
                # 统计分布
                for cmd in commands:
                    if cmd in command_distribution:
                        command_distribution[cmd] += 1
                    else:
                        command_distribution['unknown'] += 1
            except Exception as e:
                print(f"❌ 加载失败: {command_file} - {e}")
    
    print(f"\n📊 验证结果 (前100个segments):")
    print(f"总Commands: {total_commands:,}")
    print(f"Commands分布:")
    for cmd, count in command_distribution.items():
        percentage = count / max(total_commands, 1) * 100
        print(f"  {cmd}: {count:6d} ({percentage:5.1f}%)")

def main():
    print("🚀 Step 3B: 开始生成Commands...")
    
    # 加载阈值配置
    threshold, threshold_data = load_threshold_config()
    if threshold is None:
        return
    
    print(f"📊 阈值分析信息:")
    if threshold_data:
        stats = threshold_data.get('distribution_stats', {})
        print(f"  样本数量: {stats.get('count', 'N/A'):,}")
        print(f"  分布均值: {stats.get('mean', 'N/A'):.3f}m")
        print(f"  分布标准差: {stats.get('std', 'N/A'):.3f}m")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载视频段信息
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    segments = data['segments']
    print(f"📁 待处理段数: {len(segments)}")
    
    # 处理所有段生成commands
    print(f"\n🔄 开始生成Commands...")
    stats = {
        'total_segments': len(segments),
        'processed_segments': 0,
        'failed_segments': 0,
        'total_frames': 0,
        'command_counts': {cmd: 0 for cmd in COMMAND_TYPES}
    }
    
    # 处理每个段
    for segment_info in tqdm(segments, desc="生成Commands"):
        result = process_segment(segment_info, threshold)
        
        if result is not None:
            stats['processed_segments'] += 1
            stats['total_frames'] += result
        else:
            stats['failed_segments'] += 1
    
    # 验证生成结果
    verify_command_generation(segments, threshold)
    
    # 统计最终结果
    print(f"\n📊 最终处理结果:")
    print(f"总段数: {stats['total_segments']}")
    print(f"成功处理: {stats['processed_segments']}")
    print(f"失败段数: {stats['failed_segments']}")
    print(f"成功率: {stats['processed_segments']/stats['total_segments']*100:.1f}%")
    
    # 保存处理统计
    summary = {
        'threshold_used': threshold,
        'processing_stats': stats,
        'output_directory': OUTPUT_DIR,
        'total_command_files': stats['processed_segments']
    }
    
    summary_file = os.path.join(OUTPUT_DIR, "command_generation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Commands保存到: {OUTPUT_DIR}")
    print(f"📊 处理统计保存到: {summary_file}")
    print("✅ Step 3B 完成！")
    
    print(f"\n🎯 下一步:")
    print("  1. 运行验证pickle生成 (Step 3.5)")
    print("  2. 或直接运行完整数据合并 (Step 4)")

if __name__ == "__main__":
    main() 