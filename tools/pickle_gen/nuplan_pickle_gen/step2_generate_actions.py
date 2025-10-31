#!/usr/bin/env python3
"""
NuPlan Action生成脚本 - Step 2
从poses直接生成delta actions，然后计算waypoints
poses (4x4) → delta actions → waypoints
严格按照compare脚本逻辑：先delta后waypoints
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob
import pickle
import random
import sys
from pyquaternion import Quaternion

# 添加必要的路径
sys.path.append("/mnt/vdb1/yingyan.li/repo/OmniSim/tools/pickle_gen")
from navsim_coor import StateSE2, convert_absolute_to_relative_se2_array

# 配置
SEGMENTS_JSON = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/video_segments.json"
NUPLAN_JSON_DIR = "/mnt/vdb1/nuplan_json"
NAVSIM_LOGS_PATH = '/mnt/vdb1/yingyan.li/repo/OmniSim/data/navsim/navsim_logs/trainval'
OUTPUT_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate"
ANALYSIS_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/analysis"

# Action参数
SAMPLING_RATE = 10.0  # Hz
WAYPOINT_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # 未来8个时间点
WAYPOINT_FRAMES = [int(t * SAMPLING_RATE) for t in WAYPOINT_TIMES]  # [5, 10, 15, ..., 40]

def load_segments():
    """加载视频分割结果"""
    print("📋 加载视频分割结果...")
    with open(SEGMENTS_JSON, 'r') as f:
        data = json.load(f)
    return data['segments'], data['metadata']

def poses_to_delta_actions(poses_from_current):
    """
    直接从poses计算delta actions（按照compare脚本逻辑）
    每个action = 从当前帧累积5个连续poses（0.5秒）
    
    poses_from_current: 从当前帧开始的相对变换矩阵序列 (remaining_frames, 4, 4)
    返回: 8个delta action [x, y, yaw] (8, 3)
    """
    actions = []
    
    for action_idx in range(8):  # 8个action（每个0.5秒）
        # 每个action累积5个连续poses
        frame_start = action_idx * 5
        frame_end = frame_start + 5
        
        if frame_start < len(poses_from_current):
            # 累积变换
            cumulative_transform = np.eye(4, dtype=np.float64)
            for frame_idx in range(frame_start, min(frame_end, len(poses_from_current))):
                pose = poses_from_current[frame_idx].astype(np.float64)
                cumulative_transform = cumulative_transform @ pose
            
            # 提取位置和朝向变化
            dx = cumulative_transform[0, 3]
            dy = cumulative_transform[1, 3]
            dyaw = np.arctan2(cumulative_transform[1, 0], cumulative_transform[0, 0])
            
            actions.append([float(dx), float(dy), float(dyaw)])
        else:
            # 超出范围，用0填充
            actions.append([0.0, 0.0, 0.0])
    
    return np.array(actions, dtype=np.float64)

def delta_to_waypoints(delta_actions):
    """
    从delta actions计算waypoints（累积位置）
    delta_actions: (8, 3) [dx, dy, dyaw]
    返回: (8, 3) waypoints [x, y, yaw]
    """
    waypoints = []
    current_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    for delta in delta_actions:
        current_pos = current_pos + delta
        # 归一化角度
        current_pos[2] = np.arctan2(np.sin(current_pos[2]), np.cos(current_pos[2]))
        waypoints.append(current_pos.copy())
    
    return np.array(waypoints, dtype=np.float64)

def extract_image_name(cam_path):
    """从相机路径中提取图片文件名"""
    if isinstance(cam_path, str):
        return os.path.basename(cam_path)
    return None

def load_navsim_actions(log_name):
    """加载NavSim的actions，以图片名称为索引"""
    log_path = os.path.join(NAVSIM_LOGS_PATH, f"{log_name}.pkl")
    if not os.path.exists(log_path):
        return None
        
    with open(log_path, "rb") as f:
        scene = pickle.load(f)
    num_frames = len(scene)
    
    # 1. 读出所有全局 SE2 pose
    global_ego_poses = []
    for fi in scene:
        t = fi["ego2global_translation"]
        q = Quaternion(*fi["ego2global_rotation"])
        yaw = q.yaw_pitch_roll[0]
        global_ego_poses.append([t[0], t[1], yaw])
    global_ego_poses = np.array(global_ego_poses, dtype=np.float64)
    
    # 2. 批量计算 rel_all[i,j]
    rel_all = []
    for i in range(num_frames):
        origin = StateSE2(*global_ego_poses[i])
        rel = convert_absolute_to_relative_se2_array(origin, global_ego_poses)
        rel_all.append(rel)
    rel_all = np.stack(rel_all, axis=0)  # (N, N, 3)
    
    # 3. 创建以图片名称为键的字典
    image_to_data = {}
    
    for i, fi in enumerate(scene):
        cam_path = fi.get('cams', {}).get('CAM_F0', {}).get('data_path', '')
        image_name = extract_image_name(cam_path)
        
        if image_name:
            # 构造未来8个action（每个0.5秒）
            action_list = []
            for j in range(8):
                if i + j < num_frames - 1:
                    # 从第i+j帧到第i+j+1帧的相对变换
                    dx, dy, dtheta = rel_all[i + j, i + j + 1]
                else:
                    dx = dy = dtheta = 0.0
                action_list.append([float(dx), float(dy), float(dtheta)])
            
            image_to_data[image_name] = {
                'frame_idx': i,
                'actions': np.array(action_list, dtype=np.float64),  # (8, 3)
                'token': fi.get("token", f"frame_{i}"),
                'timestamp': fi.get('timestamp', None)
            }
    
    return {'image_to_data': image_to_data, 'num_frames': num_frames}

def process_segment_actions(segment):
    """处理单个segment的actions"""
    seq_name = segment['original_sequence']
    start_frame = segment['start_frame']
    frame_count = segment['frame_count']
    
    # 加载对应的JSON poses数据
    json_path = os.path.join(NUPLAN_JSON_DIR, f"{seq_name}.json")
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except:
        return None, None
    
    poses = json_data.get('poses', [])
    if len(poses) < start_frame + 1:
        return None, None
    
    # 获取segment对应的poses（严格按照JSON顺序）
    segment_poses = []
    for i in range(frame_count):
        frame_idx = start_frame + i
        if frame_idx < len(poses):
            segment_poses.append(np.array(poses[frame_idx], dtype=np.float64))
        else:
            # 如果超出范围，用单位矩阵填充
            segment_poses.append(np.eye(4, dtype=np.float64))
    
    poses_array = np.array(segment_poses)  # (frame_count, 4, 4)
    
    actions = []
    waypoints_list = []
    
    # 为每一帧生成action
    for frame_idx in range(frame_count):
        # 直接计算delta actions
        delta_actions = poses_to_delta_actions(poses_array[frame_idx:])
        actions.append(delta_actions)
        
        # 从delta计算waypoints
        waypoints = delta_to_waypoints(delta_actions)
        waypoints_list.append(waypoints)
    
    return np.array(actions, dtype=np.float64), waypoints_list

def compare_with_navsim(segment, actions, navsim_data):
    """与NavSim数据进行比较"""
    if navsim_data is None:
        return None
    
    seq_name = segment['original_sequence']
    start_frame = segment['start_frame']
    frame_count = segment['frame_count']
    
    # 加载NuPlan的图片信息
    json_path = os.path.join(NUPLAN_JSON_DIR, f"{seq_name}.json")
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except:
        return None
    
    images = json_data.get('images', [])
    if len(images) < start_frame + frame_count:
        return None
    
    # 随机选择几个帧进行比较
    num_samples = min(5, frame_count)
    sample_indices = random.sample(range(frame_count), num_samples)
    
    comparisons = []
    for local_idx in sample_indices:
        global_frame_idx = start_frame + local_idx
        if global_frame_idx < len(images):
            image_name = extract_image_name(images[global_frame_idx])
            
            if image_name and image_name in navsim_data['image_to_data']:
                nuplan_actions = actions[local_idx]  # (8, 3)
                navsim_actions = navsim_data['image_to_data'][image_name]['actions']  # (8, 3)
                
                # 计算差异
                diff = nuplan_actions - navsim_actions
                max_diff = np.max(np.abs(diff))
                
                comparisons.append({
                    'image_name': image_name,
                    'local_frame_idx': local_idx,
                    'global_frame_idx': global_frame_idx,
                    'nuplan_actions': nuplan_actions,
                    'navsim_actions': navsim_actions,
                    'diff': diff,
                    'max_diff': max_diff,
                    'rmse': np.sqrt(np.mean(diff**2)),
                    'x_rmse': np.sqrt(np.mean(diff[:, 0]**2)),
                    'y_rmse': np.sqrt(np.mean(diff[:, 1]**2)),
                    'heading_rmse': np.sqrt(np.mean(diff[:, 2]**2))
                })
    
    return comparisons

def save_segment_actions(segment, actions, waypoints_list):
    """保存单个segment的actions和waypoints到独立的.npy文件"""
    segment_id = segment['segment_id']
    actions_dir = os.path.join(OUTPUT_DIR, "actions")
    waypoints_dir = os.path.join(OUTPUT_DIR, "waypoints")
    os.makedirs(actions_dir, exist_ok=True)
    os.makedirs(waypoints_dir, exist_ok=True)
    
    # 保存actions (delta格式)
    actions_file = os.path.join(actions_dir, f"{segment_id}.npy")
    np.save(actions_file, actions)
    
    # 保存waypoints (绝对位置)
    waypoints_array = np.array(waypoints_list, dtype=np.float64)  # (frame_count, 8, 3)
    waypoints_file = os.path.join(waypoints_dir, f"{segment_id}.npy")
    np.save(waypoints_file, waypoints_array)
    
    return actions_file, waypoints_file

def analyze_trajectory_distribution(all_actions):
    """分析轨迹分布"""
    print("📊 分析轨迹分布...")
    
    # 分析0填充情况（改进版）
    # 1. 完全0填充的trajectory
    completely_zero_mask = np.all(all_actions.reshape(len(all_actions), -1) == 0, axis=1)
    completely_zero_count = np.sum(completely_zero_mask)
    
    # 2. 部分0填充的action个数
    total_actions = all_actions.shape[0] * all_actions.shape[1]  # total_trajectories * 8
    zero_actions_mask = np.all(all_actions == 0, axis=2)  # (N, 8) - 每个action是否为[0,0,0]
    zero_actions_count = np.sum(zero_actions_mask)
    
    # 3. 有效trajectory（至少有一个非零action）
    non_zero_mask = np.any(all_actions.reshape(len(all_actions), -1) != 0, axis=1)
    valid_actions = all_actions[non_zero_mask]
    
    if len(valid_actions) == 0:
        print("⚠️ 所有action都是0，可能数据有问题")
        valid_actions = all_actions
    
    # 4. 分析每个trajectory的0填充模式
    zero_padding_patterns = []
    for i, trajectory in enumerate(all_actions):
        zero_pattern = np.all(trajectory == 0, axis=1)  # (8,) 布尔数组
        first_zero_idx = np.where(zero_pattern)[0]
        if len(first_zero_idx) > 0:
            first_zero_position = first_zero_idx[0]
            consecutive_zeros = np.sum(zero_pattern[first_zero_position:])
        else:
            first_zero_position = -1
            consecutive_zeros = 0
        
        zero_padding_patterns.append({
            'trajectory_idx': i,
            'total_zero_actions': np.sum(zero_pattern),
            'first_zero_position': int(first_zero_position) if first_zero_position >= 0 else -1,
            'consecutive_tail_zeros': int(consecutive_zeros),
            'has_zero_padding': consecutive_zeros > 0
        })
    
    # 统计0填充模式
    trajectories_with_padding = sum(1 for p in zero_padding_patterns if p['has_zero_padding'])
    avg_consecutive_zeros = np.mean([p['consecutive_tail_zeros'] for p in zero_padding_patterns if p['has_zero_padding']]) if trajectories_with_padding > 0 else 0
    
    # 1. 轨迹总长度分布
    trajectory_lengths = []
    for action in valid_actions:
        # 计算8个action的累计距离
        distances = np.sqrt(action[:, 0]**2 + action[:, 1]**2)
        total_distance = np.sum(distances)
        trajectory_lengths.append(total_distance)
    
    # 2. 轨迹最终位置分布
    final_positions = []
    for action in valid_actions:
        cumulative_pos = np.cumsum(action, axis=0)  # 累积位移
        final_x, final_y = cumulative_pos[-1, 0], cumulative_pos[-1, 1]
        final_positions.append([final_x, final_y])
    
    # 3. 轨迹曲率/弯曲程度
    trajectory_curvatures = []
    for action in valid_actions:
        total_yaw_change = np.sum(np.abs(action[:, 2]))
        trajectory_curvatures.append(total_yaw_change)
    
    # 4. 各个时间点的分别统计
    waypoint_stats = {}
    for i in range(8):
        waypoint_data = valid_actions[:, i, :]  # (N, 3)
        waypoint_stats[f'action_{i+1}'] = {
            'time': WAYPOINT_TIMES[i],
            'dx_mean': float(np.mean(waypoint_data[:, 0])),
            'dx_std': float(np.std(waypoint_data[:, 0])),
            'dy_mean': float(np.mean(waypoint_data[:, 1])),
            'dy_std': float(np.std(waypoint_data[:, 1])),
            'dyaw_mean': float(np.mean(waypoint_data[:, 2])),
            'dyaw_std': float(np.std(waypoint_data[:, 2])),
            'dx_range': [float(np.min(waypoint_data[:, 0])), float(np.max(waypoint_data[:, 0]))],
            'dy_range': [float(np.min(waypoint_data[:, 1])), float(np.max(waypoint_data[:, 1]))],
            'dyaw_range': [float(np.min(waypoint_data[:, 2])), float(np.max(waypoint_data[:, 2]))]
        }
    
    # 5. 总体轨迹统计（更新版）
    trajectory_stats = {
        'total_trajectories': len(all_actions),
        'valid_trajectories': len(valid_actions),
        'completely_zero_trajectories': int(completely_zero_count),
        'trajectories_with_zero_padding': int(trajectories_with_padding),
        'total_actions': int(total_actions),
        'zero_filled_actions': int(zero_actions_count),
        'zero_padding_ratio': float(zero_actions_count) / float(total_actions),  # 修正：action级别的0填充比例
        'trajectory_zero_padding_ratio': float(trajectories_with_padding) / float(len(all_actions)),  # trajectory级别的0填充比例
        'avg_consecutive_zeros': float(avg_consecutive_zeros),
        'length_mean': float(np.mean(trajectory_lengths)),
        'length_std': float(np.std(trajectory_lengths)),
        'length_range': [float(np.min(trajectory_lengths)), float(np.max(trajectory_lengths))],
        'curvature_mean': float(np.mean(trajectory_curvatures)),
        'curvature_std': float(np.std(trajectory_curvatures)),
        'final_position_mean': [float(np.mean([pos[0] for pos in final_positions])), 
                               float(np.mean([pos[1] for pos in final_positions]))],
        'final_position_std': [float(np.std([pos[0] for pos in final_positions])), 
                              float(np.std([pos[1] for pos in final_positions]))]
    }
    
    return {
        'trajectory_stats': trajectory_stats,
        'waypoint_stats': waypoint_stats,
        'trajectory_lengths': np.array(trajectory_lengths, dtype=np.float64),
        'final_positions': np.array(final_positions, dtype=np.float64),
        'trajectory_curvatures': np.array(trajectory_curvatures, dtype=np.float64),
        'valid_actions': valid_actions,
        'zero_padding_patterns': zero_padding_patterns
    }

def analyze_navsim_comparison(all_comparisons):
    """分析NavSim对比结果"""
    if not all_comparisons:
        return None
    
    # 收集所有误差数据
    all_rmse = []
    all_x_rmse = []
    all_y_rmse = []
    all_heading_rmse = []
    all_max_diff = []
    
    for segment_comparisons in all_comparisons:
        for comp in segment_comparisons:
            all_rmse.append(comp['rmse'])
            all_x_rmse.append(comp['x_rmse'])
            all_y_rmse.append(comp['y_rmse'])
            all_heading_rmse.append(comp['heading_rmse'])
            all_max_diff.append(comp['max_diff'])
    
    return {
        'total_comparisons': len(all_rmse),
        'rmse_mean': float(np.mean(all_rmse)),
        'rmse_std': float(np.std(all_rmse)),
        'x_rmse_mean': float(np.mean(all_x_rmse)),
        'x_rmse_std': float(np.std(all_x_rmse)),
        'y_rmse_mean': float(np.mean(all_y_rmse)),
        'y_rmse_std': float(np.std(all_y_rmse)),
        'heading_rmse_mean': float(np.mean(all_heading_rmse)),
        'heading_rmse_std': float(np.std(all_heading_rmse)),
        'max_diff_mean': float(np.mean(all_max_diff)),
        'max_diff_std': float(np.std(all_max_diff)),
        'rmse_range': [float(np.min(all_rmse)), float(np.max(all_rmse))],
        'max_diff_range': [float(np.min(all_max_diff)), float(np.max(all_max_diff))]
    }

def plot_delta_distribution(stats_data, analysis_dir):
    """绘制各时刻Delta Action分布图"""
    print("📊 生成Delta Action分布图...")
    
    # 设置绘图风格
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Delta Action Distributions at Different Time Steps', fontsize=16, fontweight='bold')
    
    valid_actions = stats_data['valid_actions']  # (N, 8, 3)
    
    # 1. X方向分布 (前进/后退)
    axes[0, 0].set_title('X Direction (Forward/Backward)', fontweight='bold')
    for i in range(8):
        dx_data = valid_actions[:, i, 0]
        axes[0, 0].hist(dx_data, bins=50, alpha=0.6, label=f't={WAYPOINT_TIMES[i]}s', density=True)
    axes[0, 0].set_xlabel('Delta X (m)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Y方向分布 (左转/右转)
    axes[0, 1].set_title('Y Direction (Left/Right)', fontweight='bold')
    for i in range(8):
        dy_data = valid_actions[:, i, 1]
        axes[0, 1].hist(dy_data, bins=50, alpha=0.6, label=f't={WAYPOINT_TIMES[i]}s', density=True)
    axes[0, 1].set_xlabel('Delta Y (m)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Yaw方向分布 (旋转)
    axes[1, 0].set_title('Yaw Direction (Rotation)', fontweight='bold')
    for i in range(8):
        dyaw_data = valid_actions[:, i, 2]
        axes[1, 0].hist(dyaw_data, bins=50, alpha=0.6, label=f't={WAYPOINT_TIMES[i]}s', density=True)
    axes[1, 0].set_xlabel('Delta Yaw (rad)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 时间序列统计（箱线图）
    axes[1, 1].set_title('Delta Statistics Over Time', fontweight='bold')
    
    # 准备箱线图数据
    dx_means = [np.mean(valid_actions[:, i, 0]) for i in range(8)]
    dy_means = [np.mean(valid_actions[:, i, 1]) for i in range(8)]
    dyaw_means = [np.mean(valid_actions[:, i, 2]) for i in range(8)]
    
    dx_stds = [np.std(valid_actions[:, i, 0]) for i in range(8)]
    dy_stds = [np.std(valid_actions[:, i, 1]) for i in range(8)]
    dyaw_stds = [np.std(valid_actions[:, i, 2]) for i in range(8)]
    
    x_pos = np.array(WAYPOINT_TIMES)
    
    # 绘制均值和标准差
    axes[1, 1].errorbar(x_pos, dx_means, yerr=dx_stds, label='Delta X', marker='o', capsize=5)
    axes[1, 1].errorbar(x_pos, dy_means, yerr=dy_stds, label='Delta Y', marker='s', capsize=5)
    axes[1, 1].errorbar(x_pos, dyaw_means, yerr=dyaw_stds, label='Delta Yaw', marker='^', capsize=5)
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Delta Value (m for X,Y; rad for Yaw)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(WAYPOINT_TIMES)
    
    plt.tight_layout()
    
    # 保存图片
    fig_path = os.path.join(analysis_dir, "delta_action_distributions.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 分布图已保存: {fig_path}")
    return fig_path

def main():
    print("🚀 开始Action生成与分析...")
    print("📝 新逻辑: 先计算Delta Actions，再计算Waypoints (按照compare脚本)")
    print("🔧 配置: Float64 + 0填充 + NavSim对比")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # 加载segments
    segments, metadata = load_segments()
    print(f"📁 总segment数: {len(segments)}")
    
    # 检查NavSim数据可用性
    available_navsim_logs = []
    navsim_data_cache = {}
    
    print("🔍 检查NavSim数据可用性...")
    unique_seqs = list(set([s['original_sequence'] for s in segments]))
    print(f"📊 NuPlan unique sequences: {len(unique_seqs)}")
    
    # 快速检查前几个序列的日期范围
    sample_seqs = sorted(unique_seqs)[:5]
    print(f"📅 NuPlan样本序列日期: {[seq.split('_')[0] for seq in sample_seqs]}")
    
    # 检查NavSim数据
    navsim_files = os.listdir(NAVSIM_LOGS_PATH) if os.path.exists(NAVSIM_LOGS_PATH) else []
    navsim_seqs = [f.replace('.pkl', '') for f in navsim_files if f.endswith('.pkl')]
    sample_navsim_seqs = sorted(navsim_seqs)[:5]
    print(f"📅 NavSim样本序列日期: {[seq.split('_')[0] for seq in sample_navsim_seqs]}")
    
    # 寻找共同序列
    navsim_seq_set = set(navsim_seqs)
    common_seqs = []
    for seq_name in tqdm(unique_seqs[:100], desc="检查序列匹配"):  # 检查前100个序列
        if seq_name in navsim_seq_set:
            common_seqs.append(seq_name)
            navsim_data = load_navsim_actions(seq_name)
            navsim_data_cache[seq_name] = navsim_data
            if navsim_data is not None:
                available_navsim_logs.append(seq_name)
    
    print(f"📊 共同序列: {len(common_seqs)}")
    print(f"📊 可用NavSim logs: {len(available_navsim_logs)}")
    
    if len(common_seqs) == 0:
        print("⚠️  发现：NuPlan和NavSim数据集无重叠序列")
        print("💡 这意味着它们来自不同的数据收集批次")
        print("🔄 将跳过NavSim对比，继续处理NuPlan数据")
    
    # 处理每个segment的actions
    all_actions = []
    all_comparisons = []
    successful_segments = []
    failed_segments = []
    navsim_comparison_count = 0
    
    for segment in tqdm(segments, desc="生成Delta Actions"):
        actions, waypoints_list = process_segment_actions(segment)
        if actions is not None:
            # 保存到独立的.npy文件
            save_segment_actions(segment, actions, waypoints_list)
            
            # 收集用于统计分析
            all_actions.append(actions)
            successful_segments.append(segment['segment_id'])
            
            # NavSim对比（如果数据可用）
            seq_name = segment['original_sequence']
            if seq_name in navsim_data_cache and navsim_data_cache[seq_name] is not None:
                comparisons = compare_with_navsim(segment, actions, navsim_data_cache[seq_name])
                if comparisons:
                    all_comparisons.append(comparisons)
                    navsim_comparison_count += len(comparisons)
        else:
            failed_segments.append(segment['segment_id'])
    
    print(f"✅ 成功处理: {len(successful_segments)} 个segment")
    print(f"❌ 失败: {len(failed_segments)} 个segment")
    print(f"🔄 NavSim对比: {navsim_comparison_count} 个样本")
    
    if not all_actions:
        print("❌ 没有成功处理的segment！")
        return
    
    # 合并所有actions用于统计分析
    all_actions_array = np.concatenate(all_actions, axis=0)  # (total_frames, 8, 3)
    print(f"📊 总轨迹数: {all_actions_array.shape[0]:,}")
    
    # 分析轨迹分布
    stats_data = analyze_trajectory_distribution(all_actions_array)
    trajectory_stats = stats_data['trajectory_stats']
    
    print(f"\n📊 轨迹统计:")
    print(f"  总轨迹数: {trajectory_stats['total_trajectories']:,}")
    print(f"  有效轨迹数: {trajectory_stats['valid_trajectories']:,}")
    print(f"  完全0填充轨迹数: {trajectory_stats['completely_zero_trajectories']:,}")
    print(f"  有0填充轨迹数: {trajectory_stats['trajectories_with_zero_padding']:,}")
    print(f"  总动作数: {trajectory_stats['total_actions']:,}")
    print(f"  有0填充动作数: {trajectory_stats['zero_filled_actions']:,}")
    print(f"  0填充比例: {trajectory_stats['zero_padding_ratio']:.2%}")
    print(f"  轨迹0填充比例: {trajectory_stats['trajectory_zero_padding_ratio']:.2%}")
    print(f"  平均连续0填充动作数: {trajectory_stats['avg_consecutive_zeros']:.2f}")
    print(f"  平均轨迹长度: {trajectory_stats['length_mean']:.2f}m")
    print(f"  平均曲率: {trajectory_stats['curvature_mean']:.3f}rad")
    
    # 生成Delta分布可视化
    plot_delta_distribution(stats_data, ANALYSIS_DIR)
    
    # 分析NavSim对比结果
    comparison_stats = analyze_navsim_comparison(all_comparisons)
    if comparison_stats:
        print(f"\n🔄 NavSim对比统计:")
        print(f"  对比样本数: {comparison_stats['total_comparisons']}")
        print(f"  平均RMSE: {comparison_stats['rmse_mean']:.4f}")
        print(f"  X方向RMSE: {comparison_stats['x_rmse_mean']:.4f}")
        print(f"  Y方向RMSE: {comparison_stats['y_rmse_mean']:.4f}")
        print(f"  朝向RMSE: {comparison_stats['heading_rmse_mean']:.4f}")
        print(f"  最大差异: {comparison_stats['max_diff_mean']:.4f}")
    else:
        print(f"\n🔄 NavSim对比: 无可比较数据（数据集不重叠）")
    
    # 保存统计分析结果
    results = {
        'metadata': {
            'processing_method': 'delta_first_then_waypoints',
            'total_segments': len(segments),
            'successful_segments': len(successful_segments),
            'failed_segments': len(failed_segments),
            'total_trajectories': int(all_actions_array.shape[0]),
            'waypoint_times': WAYPOINT_TIMES,
            'zero_padding_enabled': True,
            'float64_precision': True,
            'actions_dir': f"{OUTPUT_DIR}/actions",
            'navsim_comparison_count': navsim_comparison_count,
            'available_navsim_logs': available_navsim_logs,
            'common_sequences_count': len(common_seqs),
            'dataset_overlap': len(common_seqs) > 0,
            'failed_segment_ids': failed_segments[:10]
        },
        'trajectory_stats': stats_data['trajectory_stats'],
        'waypoint_stats': stats_data['waypoint_stats'],
        'navsim_comparison_stats': comparison_stats,
        'action_shape': list(all_actions_array.shape)
    }
    
    # 保存统计分析结果
    results_file = os.path.join(OUTPUT_DIR, "action_analysis.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存actions数组
    actions_file = os.path.join(OUTPUT_DIR, "actions_raw.npz")
    np.savez_compressed(actions_file, actions=all_actions_array)
    
    # 保存NavSim对比详细结果
    if all_comparisons:
        comparison_file = os.path.join(OUTPUT_DIR, "navsim_comparisons.json")
        serializable_comparisons = []
        for segment_comps in all_comparisons:
            for comp in segment_comps:
                serializable_comparisons.append({
                    'image_name': comp['image_name'],
                    'local_frame_idx': comp['local_frame_idx'],
                    'global_frame_idx': comp['global_frame_idx'],
                    'rmse': float(comp['rmse']),
                    'x_rmse': float(comp['x_rmse']),
                    'y_rmse': float(comp['y_rmse']),
                    'heading_rmse': float(comp['heading_rmse']),
                    'max_diff': float(comp['max_diff'])
                })
        
        with open(comparison_file, 'w') as f:
            json.dump(serializable_comparisons, f, indent=2)
        
        print(f"  NavSim对比: {comparison_file}")
    
    # 打印各时间点统计
    print(f"\n📊 各时间点Delta Action统计:")
    for i in range(8):
        action_key = f'action_{i+1}'
        action_stats = stats_data['waypoint_stats'][action_key]
        time_val = action_stats['time']
        print(f"  {time_val}s: dx={action_stats['dx_mean']:.3f}±{action_stats['dx_std']:.3f}m, "
              f"dy={action_stats['dy_mean']:.3f}±{action_stats['dy_std']:.3f}m, "
              f"dyaw={action_stats['dyaw_mean']:.3f}±{action_stats['dyaw_std']:.3f}rad")
    
    # 打印文件位置信息
    actions_dir = os.path.join(OUTPUT_DIR, "actions")
    action_files_count = len([f for f in os.listdir(actions_dir) if f.endswith('.npy')])
    
    print(f"\n💾 结果保存:")
    print(f"  Actions文件: {actions_dir}/ ({action_files_count} 个.npy文件)")
    print(f"  统计分析: {results_file}")
    print(f"  Actions数组: {actions_file}")
    print(f"  分布图: {ANALYSIS_DIR}/delta_action_distributions.png")
    
    if len(common_seqs) == 0:
        print("✅ Step 2 完成！(Delta First + Float64 + 0填充，NavSim数据集不重叠)")
        print("\n💡 说明:")
        print("  - NuPlan和NavSim来自不同的数据收集批次")
        print("  - 无法进行直接对比，但数据处理成功")
        print("  - 可以使用生成的actions进行后续分析")
    else:
        print("✅ Step 2 完成！(Delta First + Float64 + 0填充 + NavSim对比)")
        print(f"  🔍 NavSim对比完成，发现平均RMSE: {comparison_stats['rmse_mean']:.4f}")
    
    print(f"\n🎯 下一步:")
    print("  运行 step3_analyze_commands.py 分析20m偏移和确定command阈值")
    print(f"  💡 Actions文件位置: {actions_dir}/")

if __name__ == "__main__":
    main() 