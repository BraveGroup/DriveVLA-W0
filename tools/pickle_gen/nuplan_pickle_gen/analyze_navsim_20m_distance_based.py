#!/usr/bin/env python3
"""
NavSim阈值优化分析脚本 - 基于20m累计距离版本
计算车辆行驶20m后的lateral displacement，分析不同阈值对正确率的影响
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import json
from collections import Counter
from pyquaternion import Quaternion
import sys

# 添加路径
sys.path.append("/mnt/vdb1/yingyan.li/repo/OmniSim/tools/pickle_gen")
from navsim_coor import StateSE2, convert_absolute_to_relative_se2_array

# 配置
NAVSIM_LOGS_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/data/navsim/navsim_logs/trainval"
OUTPUT_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/analysis/navsim_20m_distance"
SAMPLE_TARGET = 50000

# 关键参数：累计距离目标
TARGET_DISTANCE = 20.0  # 米
DISTANCE_TOLERANCE = 1.0  # 距离容差，米

# NavSim command映射
TEXT_NAME_LIST = ["go left", "go straight", "go right", "unknown"]
COMMAND_MAPPING = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT", 3: "UNKNOWN"}

# 候选阈值
CANDIDATE_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

def load_navsim_data():
    """加载NavSim数据"""
    print("📁 加载NavSim数据...")
    
    pkl_files = [f for f in os.listdir(NAVSIM_LOGS_DIR) if f.endswith('.pkl')]
    print(f"找到 {len(pkl_files)} 个NavSim序列文件")
    
    all_sequences = []
    for pkl_file in tqdm(pkl_files, desc="加载序列"):
        pkl_path = os.path.join(NAVSIM_LOGS_DIR, pkl_file)
        try:
            sequence = pickle.load(open(pkl_path, 'rb'))
            all_sequences.append({
                'name': pkl_file.replace('.pkl', ''),
                'data': sequence
            })
        except Exception as e:
            print(f"⚠️ 加载失败 {pkl_file}: {e}")
            continue
    
    print(f"✅ 成功加载 {len(all_sequences)} 个序列")
    return all_sequences

def calculate_20m_lateral_displacement(sequence_data, start_frame):
    """
    计算从start_frame开始行驶20m后的lateral displacement
    """
    num_frames = len(sequence_data)
    if start_frame >= num_frames - 1:
        return None
    
    # 1. 提取全局SE2 poses
    global_ego_poses = []
    for frame in sequence_data:
        t = frame["ego2global_translation"]
        q = Quaternion(*frame["ego2global_rotation"])
        yaw = q.yaw_pitch_roll[0]
        global_ego_poses.append([t[0], t[1], yaw])
    global_ego_poses = np.array(global_ego_poses, dtype=np.float64)
    
    # 2. 从start_frame开始累积距离
    cumulative_distance = 0.0
    current_frame = start_frame
    last_position = global_ego_poses[start_frame][:2]  # x, y
    
    # 逐帧累积距离直到达到20m
    while current_frame < num_frames - 1 and cumulative_distance < TARGET_DISTANCE:
        current_frame += 1
        current_position = global_ego_poses[current_frame][:2]
        
        # 计算这一步的距离
        step_distance = np.linalg.norm(current_position - last_position)
        cumulative_distance += step_distance
        last_position = current_position
    
    # 检查是否达到足够的距离
    if cumulative_distance < TARGET_DISTANCE - DISTANCE_TOLERANCE:
        # 距离不足，尝试外推
        if current_frame >= start_frame + 2:  # 至少有2帧的运动
            # 计算平均速度和方向
            total_displacement = global_ego_poses[current_frame][:2] - global_ego_poses[start_frame][:2]
            if cumulative_distance > 5.0:  # 有足够的运动数据
                # 外推到20m
                scale_factor = TARGET_DISTANCE / cumulative_distance
                final_position = global_ego_poses[start_frame][:2] + total_displacement * scale_factor
            else:
                return None  # 运动太少，无法可靠外推
        else:
            return None  # 数据不足
    else:
        # 已达到或超过20m
        final_position = global_ego_poses[current_frame][:2]
        
        # 如果超过20m太多，进行插值
        if cumulative_distance > TARGET_DISTANCE + DISTANCE_TOLERANCE:
            # 在最后两帧之间插值到精确的20m位置
            prev_position = global_ego_poses[current_frame - 1][:2]
            curr_position = global_ego_poses[current_frame][:2]
            
            # 计算前一帧的累积距离
            prev_cumulative = cumulative_distance - np.linalg.norm(curr_position - prev_position)
            
            # 计算需要在最后一段中行驶的距离
            remaining_distance = TARGET_DISTANCE - prev_cumulative
            last_step_distance = np.linalg.norm(curr_position - prev_position)
            
            if last_step_distance > 0:
                ratio = remaining_distance / last_step_distance
                final_position = prev_position + ratio * (curr_position - prev_position)
            else:
                final_position = curr_position
    
    # 3. 计算相对位移
    start_pose = StateSE2(*global_ego_poses[start_frame])
    final_pose_3d = np.array([final_position[0], final_position[1], global_ego_poses[start_frame][2]])  # 保持相同朝向
    
    # 转换到起始帧的局部坐标系
    relative_displacement = convert_absolute_to_relative_se2_array(start_pose, final_pose_3d.reshape(1, -1))[0]
    
    return {
        'lateral_displacement': relative_displacement[1],  # y轴偏移
        'forward_displacement': relative_displacement[0],  # x轴偏移
        'cumulative_distance': cumulative_distance,
        'frames_used': current_frame - start_frame,
        'time_used': (current_frame - start_frame) * 0.5  # 2Hz -> 0.5s per frame
    }

def extract_20m_samples(sequences, target_samples=SAMPLE_TARGET):
    """提取基于20m距离的样本"""
    print(f"🔍 提取基于20m累计距离的 {target_samples:,} 个样本...")
    
    samples = []
    total_extracted = 0
    successful_extractions = 0
    failed_extractions = 0
    
    for seq_info in tqdm(sequences, desc="提取样本"):
        if total_extracted >= target_samples:
            break
            
        seq_data = seq_info['data']
        seq_name = seq_info['name']
        num_frames = len(seq_data)
        
        if num_frames < 10:  # 序列太短
            continue
        
        # 采样帧进行分析
        sample_step = max(1, num_frames // 20)  # 每个序列最多20个样本
        
        for i in range(0, num_frames - 5, sample_step):  # 保留至少5帧的余量
            if total_extracted >= target_samples:
                break
            
            # 当前帧的ground truth command
            gt_command_onehot = seq_data[i]['driving_command']
            gt_command_idx = gt_command_onehot.nonzero()[0].item()
            gt_command_text = COMMAND_MAPPING[gt_command_idx]
            
            # 计算20m后的displacement
            displacement_info = calculate_20m_lateral_displacement(seq_data, i)
            
            if displacement_info is not None:
                samples.append({
                    'sequence': seq_name,
                    'frame_idx': i,
                    'gt_command': gt_command_text,
                    'gt_command_idx': gt_command_idx,
                    'displacement_info': displacement_info
                })
                total_extracted += 1
                successful_extractions += 1
            else:
                failed_extractions += 1
    
    print(f"✅ 提取完成:")
    print(f"  成功: {successful_extractions:,} 个样本")
    print(f"  失败: {failed_extractions:,} 个样本")
    print(f"  成功率: {successful_extractions/(successful_extractions+failed_extractions)*100:.1f}%")
    
    # 分析距离和时间统计
    if samples:
        distances = [s['displacement_info']['cumulative_distance'] for s in samples]
        times = [s['displacement_info']['time_used'] for s in samples]
        frames = [s['displacement_info']['frames_used'] for s in samples]
        
        print(f"\n📊 20m距离统计:")
        print(f"  平均距离: {np.mean(distances):.2f}m (std: {np.std(distances):.2f})")
        print(f"  平均用时: {np.mean(times):.2f}s (std: {np.std(times):.2f})")
        print(f"  平均帧数: {np.mean(frames):.1f} (std: {np.std(frames):.1f})")
        print(f"  距离范围: [{np.min(distances):.2f}, {np.max(distances):.2f}]m")
    
    return samples

def analyze_threshold_performance(samples):
    """分析不同阈值的准确率性能"""
    print(f"\n🔍 分析不同阈值的准确率性能...")
    
    # Ground truth分布
    gt_commands = [sample['gt_command'] for sample in samples]
    gt_distribution = Counter(gt_commands)
    total_samples = len(samples)
    
    print(f"\n📊 Ground Truth分布:")
    for cmd in ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]:
        count = gt_distribution.get(cmd, 0)
        pct = count / total_samples * 100
        print(f"  {cmd:10s}: {count:6d} ({pct:5.1f}%)")
    
    # 对每个阈值分析
    results = {}
    
    for threshold in CANDIDATE_THRESHOLDS:
        print(f"\n分析阈值 {threshold:.1f}m...")
        
        # 预测commands并计算准确率
        correct_predictions = 0
        predicted_commands = []
        
        for sample in samples:
            lateral_disp = sample['displacement_info']['lateral_displacement']
            gt_command = sample['gt_command']
            
            # 预测命令 - 修正逻辑
            if abs(lateral_disp) <= threshold:
                predicted_command = "STRAIGHT"
            elif lateral_disp > threshold:  # 正值 -> LEFT
                predicted_command = "LEFT"
            else:  # lateral_disp < -threshold, 负值 -> RIGHT
                predicted_command = "RIGHT"
            
            predicted_commands.append(predicted_command)
            
            # 检查是否正确（忽略UNKNOWN类别）
            if gt_command != "UNKNOWN" and predicted_command == gt_command:
                correct_predictions += 1
        
        pred_distribution = Counter(predicted_commands)
        
        # 计算准确率（排除UNKNOWN样本）
        valid_samples = sum(1 for sample in samples if sample['gt_command'] != "UNKNOWN")
        accuracy = correct_predictions / valid_samples if valid_samples > 0 else 0.0
        
        # 计算各类别的准确率
        class_accuracies = {}
        for cmd_class in ["LEFT", "STRAIGHT", "RIGHT"]:
            class_correct = 0
            class_total = 0
            for sample, pred_cmd in zip(samples, predicted_commands):
                if sample['gt_command'] == cmd_class:
                    class_total += 1
                    if pred_cmd == cmd_class:
                        class_correct += 1
            
            class_accuracies[cmd_class] = class_correct / class_total if class_total > 0 else 0.0
        
        results[threshold] = {
            'accuracy': accuracy,
            'predicted_distribution': dict(pred_distribution),
            'class_accuracies': class_accuracies,
            'correct_predictions': correct_predictions,
            'valid_samples': valid_samples,
            'distribution_percentages': {
                cmd: count/total_samples*100 for cmd, count in pred_distribution.items()
            }
        }
        
        print(f"  总体准确率: {accuracy:.3f} ({correct_predictions}/{valid_samples})")
        print(f"  各类别准确率: LEFT={class_accuracies['LEFT']:.3f}, STRAIGHT={class_accuracies['STRAIGHT']:.3f}, RIGHT={class_accuracies['RIGHT']:.3f}")
    
    return results, gt_distribution

def create_20m_visualization(samples, results, gt_distribution):
    """创建20m距离分析的可视化"""
    print(f"📊 生成20m距离分析图表...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Ground Truth分布
    plt.subplot(2, 4, 1)
    cmd_names = ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]
    gt_counts = [gt_distribution.get(cmd, 0) for cmd in cmd_names]
    colors = ['lightcoral', 'lightgreen', 'lightblue', 'lightgray']
    bars = plt.bar(cmd_names, gt_counts, color=colors, alpha=0.8)
    plt.title('NavSim Ground Truth\n(20m distance-based)')
    plt.ylabel('Count')
    for bar, count in zip(bars, gt_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + len(samples)*0.01,
                f'{count/len(samples)*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. 距离分布
    plt.subplot(2, 4, 2)
    distances = [s['displacement_info']['cumulative_distance'] for s in samples]
    plt.hist(distances, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(TARGET_DISTANCE, color='red', linestyle='--', label=f'Target: {TARGET_DISTANCE}m')
    plt.xlabel('Cumulative Distance (m)')
    plt.ylabel('Frequency')
    plt.title(f'Distance Distribution\n(mean: {np.mean(distances):.1f}m)')
    plt.legend()
    
    # 3. 时间分布
    plt.subplot(2, 4, 3)
    times = [s['displacement_info']['time_used'] for s in samples]
    plt.hist(times, bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Time Used (s)')
    plt.ylabel('Frequency')
    plt.title(f'Time Distribution\n(mean: {np.mean(times):.1f}s)')
    
    # 4. 准确率vs阈值
    plt.subplot(2, 4, 4)
    thresholds = list(results.keys())
    accuracies = [results[t]['accuracy'] for t in thresholds]
    plt.plot(thresholds, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Threshold (m)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # 找到最佳阈值
    best_threshold = max(results.keys(), key=lambda t: results[t]['accuracy'])
    best_accuracy = results[best_threshold]['accuracy']
    plt.axvline(best_threshold, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_threshold}m')
    plt.legend()
    
    # 5-8. 不同阈值的command分布对比
    top_thresholds = sorted(results.keys(), key=lambda t: results[t]['accuracy'], reverse=True)[:4]
    
    for i, threshold in enumerate(top_thresholds):
        plt.subplot(2, 4, 5 + i)
        
        pred_dist = results[threshold]['predicted_distribution']
        pred_counts = [pred_dist.get(cmd, 0) for cmd in cmd_names[:3]]  # 忽略UNKNOWN
        gt_counts_main = [gt_distribution.get(cmd, 0) for cmd in cmd_names[:3]]
        
        x = np.arange(len(cmd_names[:3]))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, gt_counts_main, width, label='GT', color='skyblue', alpha=0.8)
        bars2 = plt.bar(x + width/2, pred_counts, width, label='Pred', color='orange', alpha=0.8)
        
        plt.xlabel('Command')
        plt.ylabel('Count')
        plt.title(f'Threshold: {threshold:.1f}m\n(Accuracy: {results[threshold]["accuracy"]:.3f})')
        plt.xticks(x, cmd_names[:3])
        plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(OUTPUT_DIR, "navsim_20m_distance_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def main():
    print("🚀 开始基于20m累计距离的NavSim阈值分析...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    sequences = load_navsim_data()
    if not sequences:
        print("❌ 没有找到有效数据！")
        return
    
    # 2. 提取20m样本
    samples = extract_20m_samples(sequences, SAMPLE_TARGET)
    if not samples:
        print("❌ 样本提取失败！")
        return
    
    # 3. 分析阈值性能
    results, gt_distribution = analyze_threshold_performance(samples)
    
    # 4. 生成可视化
    plot_file = create_20m_visualization(samples, results, gt_distribution)
    
    # 5. 找到最佳阈值
    best_threshold = max(results.keys(), key=lambda t: results[t]['accuracy'])
    best_result = results[best_threshold]
    
    print(f"\n" + "="*80)
    print(f"🎯 基于20m累计距离的阈值分析完成！")
    print(f"="*80)
    print(f"📊 分析样本数: {len(samples):,}")
    print(f"🏆 最佳阈值: {best_threshold:.1f}m")
    print(f"📈 最佳准确率: {best_result['accuracy']:.3f}")
    print(f"🎯 正确预测数: {best_result['correct_predictions']}/{best_result['valid_samples']}")
    
    print(f"\n📊 最佳阈值各类别准确率:")
    for cmd_class in ["LEFT", "STRAIGHT", "RIGHT"]:
        acc = best_result['class_accuracies'][cmd_class]
        print(f"  {cmd_class:10s}: {acc:.3f}")
    
    print(f"\n💾 结果文件:")
    print(f"  📊 可视化图: {plot_file}")
    
    # 6. 保存结果
    final_results = {
        'method': '20m_cumulative_distance',
        'target_distance': TARGET_DISTANCE,
        'sample_count': len(samples),
        'best_threshold': best_threshold,
        'best_accuracy': best_result['accuracy'],
        'best_class_accuracies': best_result['class_accuracies'],
        'gt_distribution': dict(gt_distribution),
        'threshold_results': results,
        'distance_stats': {
            'mean_distance': float(np.mean([s['displacement_info']['cumulative_distance'] for s in samples])),
            'mean_time': float(np.mean([s['displacement_info']['time_used'] for s in samples])),
            'mean_frames': float(np.mean([s['displacement_info']['frames_used'] for s in samples]))
        }
    }
    
    results_file = os.path.join(OUTPUT_DIR, "navsim_20m_distance_results.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"  📄 详细结果: {results_file}")

if __name__ == "__main__":
    main() 