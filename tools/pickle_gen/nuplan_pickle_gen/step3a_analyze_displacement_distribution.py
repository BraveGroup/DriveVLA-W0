#!/usr/bin/env python3
"""
NuPlan 偏移分布分析脚本 - Step 3A
分析20m偏移分布，可视化后让用户确定阈值，不生成commands
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置
NUPLAN_JSON_DIR = "/mnt/vdb1/nuplan_json"
INPUT_FILE = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/video_segments.json"
OUTPUT_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate"
ANALYSIS_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/analysis"

FORWARD_DISTANCE = 20.0  # 前进20m时判断lateral movement

def load_poses_from_json(seq_name):
    """从JSON文件加载poses数据"""
    json_file = os.path.join(NUPLAN_JSON_DIR, f"{seq_name}.json")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get('poses', [])
    except Exception as e:
        return []

def calculate_lateral_displacement(poses, start_frame):
    """计算20m后的lateral displacement"""
    if start_frame >= len(poses):
        return None
    
    # 从start_frame开始累积转移矩阵
    current_pose = np.eye(4, dtype=np.float64)
    path_distance = 0.0
    last_position = np.array([0.0, 0.0])  # 起始位置
    current_position = np.array([0.0, 0.0])  # 初始化current_position
    
    frame_idx = start_frame + 1  # 从下一帧开始累积
    
    while frame_idx < len(poses) and path_distance < FORWARD_DISTANCE:
        # 获取当前帧的转移矩阵
        transform_matrix = np.array(poses[frame_idx], dtype=np.float64)
        
        # 累积转移矩阵
        current_pose = current_pose @ transform_matrix
        
        # 计算当前位置
        current_position = np.array([current_pose[0, 3], current_pose[1, 3]])
        
        # 计算路径距离增量
        distance_increment = np.linalg.norm(current_position - last_position)
        path_distance += distance_increment
        
        last_position = current_position
        frame_idx += 1
    
    # 如果路径距离不足20m，需要外推
    if path_distance < FORWARD_DISTANCE and frame_idx >= len(poses):
        if path_distance > 5.0:  # 有足够运动来外推
            # 计算最后的运动方向
            if frame_idx >= start_frame + 2:  # 至少有两个变换
                # 获取最后一个变换的方向
                last_transform = np.array(poses[frame_idx-1], dtype=np.float64)
                direction = np.array([last_transform[0, 3], last_transform[1, 3]])
                if np.linalg.norm(direction) > 0.01:
                    direction = direction / np.linalg.norm(direction)
                    # 外推到20m
                    remaining_distance = FORWARD_DISTANCE - path_distance
                    final_position = current_position + direction * remaining_distance
                else:
                    final_position = current_position
            else:
                final_position = current_position
        else:
            # 距离太短，直接用当前位置
            final_position = current_position
    else:
        # 已达到20m
        final_position = current_position
    
    # 返回y轴偏移（正值=右，负值=左）
    lateral_displacement = final_position[1]
    return lateral_displacement

def collect_displacement_data(segments, sample_limit=10000):
    """收集lateral displacement数据"""
    print(f"📊 开始收集20m偏移数据 (限制样本数: {sample_limit:,})...")
    
    all_displacements = []
    processed_segments = 0
    failed_segments = 0
    
    for segment_info in tqdm(segments, desc="收集偏移数据"):
        if len(all_displacements) >= sample_limit:
            break
            
        original_sequence = segment_info['original_sequence']
        start_frame = segment_info['start_frame']
        frame_count = segment_info['frame_count']
        
        # 加载poses数据
        poses = load_poses_from_json(original_sequence)
        if not poses:
            failed_segments += 1
            continue
        
        # 为每帧计算displacement（采样以加速）
        step = max(1, frame_count // 20)  # 每个segment最多采样20个点
        segment_samples = 0
        
        for i in range(0, frame_count, step):
            if len(all_displacements) >= sample_limit:
                break
                
            frame_idx = start_frame + i
            lateral_disp = calculate_lateral_displacement(poses, frame_idx)
            
            if lateral_disp is not None:
                all_displacements.append(lateral_disp)
                segment_samples += 1
        
        if segment_samples > 0:
            processed_segments += 1
        else:
            failed_segments += 1
        
        # 每100个segments显示进度
        if (processed_segments + failed_segments) % 100 == 0:
            print(f"已处理 {processed_segments} 个segments (失败{failed_segments})，收集到 {len(all_displacements):,} 个样本")
    
    print(f"\n📊 数据收集完成:")
    print(f"成功segments: {processed_segments}")
    print(f"失败segments: {failed_segments}")
    print(f"总样本数: {len(all_displacements):,}")
    
    if not all_displacements:
        print("❌ 未找到有效的displacement数据！")
        return None
    
    return np.array(all_displacements)

def analyze_distribution(displacements):
    """分析分布并返回统计数据"""
    abs_displacements = np.abs(displacements)
    
    stats = {
        'count': len(displacements),
        'mean': np.mean(displacements),
        'std': np.std(displacements),
        'min': np.min(displacements),
        'max': np.max(displacements),
        'median': np.median(displacements),
        'q25': np.percentile(displacements, 25),
        'q75': np.percentile(displacements, 75),
        'q90': np.percentile(displacements, 90),
        'q95': np.percentile(displacements, 95),
        'q99': np.percentile(displacements, 99)
    }
    
    abs_stats = {
        'abs_mean': np.mean(abs_displacements),
        'abs_std': np.std(abs_displacements),
        'abs_median': np.median(abs_displacements),
        'abs_q50': np.percentile(abs_displacements, 50),
        'abs_q75': np.percentile(abs_displacements, 75),
        'abs_q80': np.percentile(abs_displacements, 80),
        'abs_q85': np.percentile(abs_displacements, 85),
        'abs_q90': np.percentile(abs_displacements, 90),
        'abs_q95': np.percentile(abs_displacements, 95)
    }
    
    return stats, abs_stats, abs_displacements

def generate_candidate_thresholds(abs_stats):
    """生成候选阈值"""
    candidate_thresholds = [
        1.0, 1.5, 2.0, 2.5, 3.0,  # 常用阈值
        abs_stats['abs_q75'], 
        abs_stats['abs_q80'], 
        abs_stats['abs_q85'], 
        abs_stats['abs_q90']
    ]
    # 去重并排序
    candidate_thresholds = sorted(list(set([round(t, 2) for t in candidate_thresholds])))
    return candidate_thresholds

def create_distribution_visualization(displacements, abs_displacements, candidate_thresholds):
    """创建分布可视化图表"""
    try:
        print(f"📈 开始生成分布图表...")
        
        plt.figure(figsize=(20, 12))
        
        # 1. 原始分布（带正负）
        plt.subplot(2, 3, 1)
        plt.hist(displacements, bins=100, alpha=0.7, edgecolor='black')
        plt.xlabel('Lateral Displacement (m)')
        plt.ylabel('Frequency')
        plt.title('20m后横向偏移分布\n(负值=左转, 正值=右转)')
        plt.axvline(0, color='red', linestyle='-', alpha=0.8, label='中心线', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. 绝对值分布
        plt.subplot(2, 3, 2)
        plt.hist(abs_displacements, bins=100, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('|Lateral Displacement| (m)')
        plt.ylabel('Frequency')
        plt.title('绝对偏移分布')
        
        # 标注候选阈值
        colors = ['red', 'blue', 'green', 'purple', 'brown']
        for i, threshold in enumerate(candidate_thresholds[:5]):
            color = colors[i % len(colors)]
            plt.axvline(threshold, color=color, linestyle='--', alpha=0.8, 
                       label=f'{threshold}m')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 累积分布
        plt.subplot(2, 3, 3)
        sorted_abs = np.sort(abs_displacements)
        cumulative = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
        plt.plot(sorted_abs, cumulative, linewidth=2)
        plt.xlabel('|Lateral Displacement| (m)')
        plt.ylabel('累积概率')
        plt.title('累积分布曲线')
        plt.grid(True, alpha=0.3)
        
        # 标注分位数
        for i, threshold in enumerate(candidate_thresholds[:5]):
            color = colors[i % len(colors)]
            prob = np.sum(abs_displacements <= threshold) / len(abs_displacements)
            plt.axvline(threshold, color=color, linestyle='--', alpha=0.8, 
                       label=f'{threshold}m ({prob:.1%})')
            plt.text(threshold, prob + 0.02, f'{prob:.1%}', 
                    ha='center', fontsize=9, color=color, fontweight='bold')
        plt.legend()
        
        # 4-6. 不同阈值下的command分布预测
        for idx, threshold in enumerate(candidate_thresholds[:3]):
            plt.subplot(2, 3, 4 + idx)
            
            # 计算command分布
            commands = []
            for disp in displacements:
                if abs(disp) > threshold:
                    commands.append("左转" if disp < 0 else "右转")
                else:
                    commands.append("直行")
            
            cmd_counts = {cmd: commands.count(cmd) for cmd in ["左转", "直行", "右转"]}
            cmd_percentages = {cmd: count/len(commands)*100 for cmd, count in cmd_counts.items()}
            
            bars = plt.bar(cmd_counts.keys(), cmd_counts.values(), 
                          color=['lightcoral', 'lightgreen', 'lightblue'], alpha=0.8)
            plt.title(f'阈值 {threshold}m 下的指令分布')
            plt.ylabel('数量')
            
            # 添加百分比标签
            for bar, (cmd, pct) in zip(bars, cmd_percentages.items()):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + len(commands)*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_file = os.path.join(ANALYSIS_DIR, '20m_displacement_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 分析图保存到: {plot_file}")
        return plot_file
        
    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        return None

def print_threshold_analysis(displacements, candidate_thresholds):
    """打印不同阈值下的分布分析"""
    print(f"\n📊 不同阈值下的指令分布分析:")
    print("="*80)
    
    for threshold in candidate_thresholds:
        # 计算command分布
        left_count = np.sum((displacements < -threshold))
        right_count = np.sum((displacements > threshold))
        straight_count = np.sum(np.abs(displacements) <= threshold)
        total = len(displacements)
        
        left_pct = left_count / total * 100
        straight_pct = straight_count / total * 100
        right_pct = right_count / total * 100
        
        print(f"\n🎯 阈值 {threshold:.2f}m:")
        print(f"  左转:  {left_count:6d} ({left_pct:5.1f}%)")
        print(f"  直行:  {straight_count:6d} ({straight_pct:5.1f}%)")
        print(f"  右转:  {right_count:6d} ({right_pct:5.1f}%)")
        print(f"  总计:  {total:6d}")

def get_user_threshold_choice(candidate_thresholds):
    """获取用户选择的阈值"""
    print(f"\n🎯 请选择合适的阈值:")
    print("="*50)
    
    for i, threshold in enumerate(candidate_thresholds):
        print(f"{i+1:2d}. {threshold:.2f}m")
    print(f"{len(candidate_thresholds)+1:2d}. 自定义阈值")
    
    while True:
        try:
            choice = input(f"\n请输入选择 (1-{len(candidate_thresholds)+1}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(candidate_thresholds):
                selected_threshold = candidate_thresholds[choice_num - 1]
                print(f"✅ 选择阈值: {selected_threshold:.2f}m")
                return selected_threshold
            elif choice_num == len(candidate_thresholds) + 1:
                custom = input("请输入自定义阈值(米): ").strip()
                custom_threshold = float(custom)
                if 0.1 <= custom_threshold <= 10.0:
                    print(f"✅ 自定义阈值: {custom_threshold:.2f}m")
                    return custom_threshold
                else:
                    print("❌ 阈值应在0.1-10.0米之间")
            else:
                print("❌ 无效选择，请重新输入")
        except (ValueError, KeyboardInterrupt):
            print("❌ 输入格式错误，请重新输入")

def main():
    print("🚀 Step 3A: 开始20m偏移分布分析...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # 加载segments
    print(f"📁 加载segments数据...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    segments = data['segments']
    print(f"✅ 总segment数: {len(segments)}")
    
    # 收集displacement数据
    displacements = collect_displacement_data(segments, sample_limit=10000)
    if displacements is None:
        print("❌ 数据收集失败！")
        return
    
    print(f"✅ 收集到 {len(displacements):,} 个有效样本")
    
    # 分析分布
    print(f"📊 分析分布统计...")
    stats, abs_stats, abs_displacements = analyze_distribution(displacements)
    
    # 打印基础统计
    print(f"\n📊 基础统计:")
    print(f"样本数量: {stats['count']:,}")
    print(f"均值: {stats['mean']:.3f}m")
    print(f"标准差: {stats['std']:.3f}m")
    print(f"中位数: {stats['median']:.3f}m")
    print(f"范围: [{stats['min']:.3f}, {stats['max']:.3f}]m")
    print(f"\n绝对值分布:")
    print(f"Q50: {abs_stats['abs_q50']:.3f}m")
    print(f"Q75: {abs_stats['abs_q75']:.3f}m")
    print(f"Q80: {abs_stats['abs_q80']:.3f}m")
    print(f"Q85: {abs_stats['abs_q85']:.3f}m")
    print(f"Q90: {abs_stats['abs_q90']:.3f}m")
    print(f"Q95: {abs_stats['abs_q95']:.3f}m")
    
    # 生成候选阈值
    candidate_thresholds = generate_candidate_thresholds(abs_stats)
    print(f"\n🎯 候选阈值: {candidate_thresholds}")
    
    # 可视化分布
    plot_file = create_distribution_visualization(displacements, abs_displacements, candidate_thresholds)
    
    # 打印不同阈值分析
    print_threshold_analysis(displacements, candidate_thresholds)
    
    # 让用户选择阈值
    if plot_file:
        print(f"\n📈 请查看生成的分析图: {plot_file}")
    selected_threshold = get_user_threshold_choice(candidate_thresholds)
    
    # 保存结果
    results = {
        'selected_threshold': selected_threshold,
        'distribution_stats': stats,
        'abs_distribution_stats': abs_stats,
        'candidate_thresholds': candidate_thresholds,
        'sample_count': len(displacements),
        'analysis_plot': plot_file
    }
    
    results_file = os.path.join(OUTPUT_DIR, "command_threshold_analysis.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 分析结果保存到: {results_file}")
    print(f"🎯 确定阈值: {selected_threshold:.2f}m")
    print("\n✅ Step 3A 完成！")
    print(f"\n🎯 下一步:")
    print("  运行 Step 3B 使用此阈值生成Commands")

if __name__ == "__main__":
    main() 