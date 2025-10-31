#!/usr/bin/env python3
"""
Actions和Waypoints分布可视化脚本
基于已保存的delta actions生成不同时刻的分布图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob
import json

# 配置路径
ACTIONS_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/actions"
OUTPUT_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/analysis"
ANALYSIS_JSON = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/action_analysis.json"

# Action参数
WAYPOINT_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

def load_all_actions():
    """加载所有保存的actions文件"""
    print("📁 加载所有actions文件...")
    
    action_files = glob.glob(os.path.join(ACTIONS_DIR, "*.npy"))
    print(f"发现 {len(action_files)} 个actions文件")
    
    all_actions = []
    successful_files = 0
    
    for file_path in tqdm(action_files, desc="加载actions"):
        try:
            actions = np.load(file_path)  # (frame_count, 8, 3)
            all_actions.append(actions)
            successful_files += 1
        except Exception as e:
            print(f"⚠️ 加载失败: {os.path.basename(file_path)} - {e}")
    
    if not all_actions:
        print("❌ 没有成功加载任何actions文件！")
        return None
    
    # 合并所有actions
    combined_actions = np.concatenate(all_actions, axis=0)  # (total_frames, 8, 3)
    
    print(f"✅ 成功加载 {successful_files} 个文件")
    print(f"📊 总轨迹数: {combined_actions.shape[0]:,}")
    print(f"📊 数据形状: {combined_actions.shape} (frames, timesteps, [dx,dy,dyaw])")
    
    return combined_actions

def delta_to_waypoints_batch(delta_actions):
    """
    批量将delta actions转换为waypoints（累积位置）
    delta_actions: (N, 8, 3) - N个轨迹的delta actions
    返回: (N, 8, 3) - N个轨迹的waypoints
    """
    print("🔄 计算waypoints（累积位置）...")
    
    # 累积求和得到waypoints
    waypoints = np.cumsum(delta_actions, axis=1)  # (N, 8, 3)
    
    # 归一化角度到[-π, π]
    waypoints[:, :, 2] = np.arctan2(np.sin(waypoints[:, :, 2]), np.cos(waypoints[:, :, 2]))
    
    return waypoints

def plot_delta_distributions(delta_actions, output_dir):
    """绘制Delta Actions在不同时刻的分布（2D散点图）"""
    print("📊 生成Delta Actions分布图...")
    
    # 设置绘图风格
    plt.style.use('default')
    sns.set_palette("tab10")
    
    # 创建4x2子图布局
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('Delta Actions Distributions at Different Timesteps', fontsize=16, fontweight='bold')
    
    for i in range(8):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # 提取第i个时刻的delta数据
        delta_data = delta_actions[:, i, :]  # (N, 3)
        dx = delta_data[:, 0]  # X方向
        dy = delta_data[:, 1]  # Y方向
        
        # 绘制散点图
        scatter = ax.scatter(dx, dy, alpha=0.6, s=1, c='blue', rasterized=True)
        
        # 设置标题和标签
        time_val = WAYPOINT_TIMES[i]
        ax.set_title(f'Delta Actions at t={time_val}s', fontweight='bold', fontsize=12)
        ax.set_xlabel('ΔX (m)')
        ax.set_ylabel('ΔY (m)')
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围（基于95%数据范围）
        dx_95 = np.percentile(np.abs(dx), 95)
        dy_95 = np.percentile(np.abs(dy), 95)
        ax.set_xlim(-dx_95*1.1, dx_95*1.1)
        ax.set_ylim(-dy_95*1.1, dy_95*1.1)
        
        # 添加统计信息
        dx_std = np.std(dx)
        dy_std = np.std(dy)
        ax.text(0.02, 0.98, f'σx={dx_std:.3f}m\nσy={dy_std:.3f}m', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    fig_path = os.path.join(output_dir, "delta_actions_distributions_2d.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Delta分布图已保存: {fig_path}")
    return fig_path

def plot_waypoint_distributions(waypoints, output_dir):
    """绘制Waypoints在不同时刻的分布（2D散点图）"""
    print("📊 生成Waypoints分布图...")
    
    # 设置绘图风格
    plt.style.use('default')
    sns.set_palette("tab10")
    
    # 创建4x2子图布局
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('Waypoints Distributions at Different Timesteps', fontsize=16, fontweight='bold')
    
    for i in range(8):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # 提取第i个时刻的waypoint数据
        waypoint_data = waypoints[:, i, :]  # (N, 3)
        x = waypoint_data[:, 0]  # X方向累积位置
        y = waypoint_data[:, 1]  # Y方向累积位置
        
        # 绘制散点图
        scatter = ax.scatter(x, y, alpha=0.6, s=1, c='red', rasterized=True)
        
        # 设置标题和标签
        time_val = WAYPOINT_TIMES[i]
        ax.set_title(f'Waypoint {i+1} ({time_val}s)', fontweight='bold', fontsize=12)
        ax.set_xlabel('ΔX (m)')
        ax.set_ylabel('ΔY (m)')
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围（基于95%数据范围）
        x_95 = np.percentile(np.abs(x), 95)
        y_95 = np.percentile(np.abs(y), 95)
        ax.set_xlim(-x_95*1.1, x_95*1.1)
        ax.set_ylim(-y_95*1.1, y_95*1.1)
        
        # 添加统计信息
        x_std = np.std(x)
        y_std = np.std(y)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ax.text(0.02, 0.98, f'μx={x_mean:.3f}m, σx={x_std:.3f}m\nμy={y_mean:.3f}m, σy={y_std:.3f}m', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    fig_path = os.path.join(output_dir, "waypoints_distributions_2d.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Waypoints分布图已保存: {fig_path}")
    return fig_path

def plot_single_timestep_comparison(delta_actions, waypoints, timestep_idx, output_dir):
    """绘制单个时刻的Delta vs Waypoint对比图"""
    time_val = WAYPOINT_TIMES[timestep_idx]
    print(f"📊 生成时刻 {time_val}s 的对比图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Delta Actions vs Waypoints at t={time_val}s', fontsize=16, fontweight='bold')
    
    # Delta Actions
    delta_data = delta_actions[:, timestep_idx, :]
    dx = delta_data[:, 0]
    dy = delta_data[:, 1]
    
    axes[0].scatter(dx, dy, alpha=0.6, s=2, c='blue', rasterized=True)
    axes[0].set_title(f'Delta Actions at t={time_val}s', fontweight='bold')
    axes[0].set_xlabel('ΔX (m)')
    axes[0].set_ylabel('ΔY (m)')
    axes[0].grid(True, alpha=0.3)
    
    dx_95 = np.percentile(np.abs(dx), 95)
    dy_95 = np.percentile(np.abs(dy), 95)
    axes[0].set_xlim(-dx_95*1.1, dx_95*1.1)
    axes[0].set_ylim(-dy_95*1.1, dy_95*1.1)
    
    # Waypoints
    waypoint_data = waypoints[:, timestep_idx, :]
    x = waypoint_data[:, 0]
    y = waypoint_data[:, 1]
    
    axes[1].scatter(x, y, alpha=0.6, s=2, c='red', rasterized=True)
    axes[1].set_title(f'Waypoint {timestep_idx+1} ({time_val}s)', fontweight='bold')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].grid(True, alpha=0.3)
    
    x_95 = np.percentile(np.abs(x), 95)
    y_95 = np.percentile(np.abs(y), 95)
    axes[1].set_xlim(-x_95*1.1, x_95*1.1)
    axes[1].set_ylim(-y_95*1.1, y_95*1.1)
    
    plt.tight_layout()
    
    # 保存图片
    fig_path = os.path.join(output_dir, f"comparison_t{time_val}s.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 对比图已保存: {fig_path}")
    return fig_path

def analyze_distributions(delta_actions, waypoints):
    """分析分布统计信息"""
    print("📊 分析分布统计...")
    
    stats = {
        'delta_stats': {},
        'waypoint_stats': {}
    }
    
    for i in range(8):
        time_val = WAYPOINT_TIMES[i]
        
        # Delta统计
        delta_data = delta_actions[:, i, :]
        stats['delta_stats'][f't_{time_val}s'] = {
            'dx_mean': float(np.mean(delta_data[:, 0])),
            'dx_std': float(np.std(delta_data[:, 0])),
            'dy_mean': float(np.mean(delta_data[:, 1])),
            'dy_std': float(np.std(delta_data[:, 1])),
            'dyaw_mean': float(np.mean(delta_data[:, 2])),
            'dyaw_std': float(np.std(delta_data[:, 2])),
            'dx_95_percentile': float(np.percentile(np.abs(delta_data[:, 0]), 95)),
            'dy_95_percentile': float(np.percentile(np.abs(delta_data[:, 1]), 95))
        }
        
        # Waypoint统计
        waypoint_data = waypoints[:, i, :]
        stats['waypoint_stats'][f't_{time_val}s'] = {
            'x_mean': float(np.mean(waypoint_data[:, 0])),
            'x_std': float(np.std(waypoint_data[:, 0])),
            'y_mean': float(np.mean(waypoint_data[:, 1])),
            'y_std': float(np.std(waypoint_data[:, 1])),
            'yaw_mean': float(np.mean(waypoint_data[:, 2])),
            'yaw_std': float(np.std(waypoint_data[:, 2])),
            'x_95_percentile': float(np.percentile(np.abs(waypoint_data[:, 0]), 95)),
            'y_95_percentile': float(np.percentile(np.abs(waypoint_data[:, 1]), 95))
        }
    
    return stats

def main():
    print("🚀 开始Actions和Waypoints分布可视化...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载所有actions数据
    delta_actions = load_all_actions()
    if delta_actions is None:
        return
    
    # 过滤掉全零的轨迹
    non_zero_mask = np.any(delta_actions.reshape(len(delta_actions), -1) != 0, axis=1)
    valid_delta_actions = delta_actions[non_zero_mask]
    print(f"📊 过滤后有效轨迹数: {valid_delta_actions.shape[0]:,}")
    
    # 计算waypoints
    waypoints = delta_to_waypoints_batch(valid_delta_actions)
    
    # 生成分布图
    print("\n📊 生成可视化图表...")
    
    # 1. Delta Actions分布图
    delta_fig_path = plot_delta_distributions(valid_delta_actions, OUTPUT_DIR)
    
    # 2. Waypoints分布图
    waypoint_fig_path = plot_waypoint_distributions(waypoints, OUTPUT_DIR)
    
    # 3. 生成几个关键时刻的对比图
    key_timesteps = [1, 3, 7]  # 0.5s, 2.0s, 4.0s
    comparison_paths = []
    for idx in key_timesteps:
        comp_path = plot_single_timestep_comparison(valid_delta_actions, waypoints, idx, OUTPUT_DIR)
        comparison_paths.append(comp_path)
    
    # 4. 分析统计信息
    stats = analyze_distributions(valid_delta_actions, waypoints)
    
    # 保存统计结果
    stats_file = os.path.join(OUTPUT_DIR, "distribution_analysis.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 打印总结
    print(f"\n✅ 可视化完成！")
    print(f"📊 处理轨迹数: {valid_delta_actions.shape[0]:,}")
    print(f"📊 时间点数: {len(WAYPOINT_TIMES)}")
    print(f"\n💾 输出文件:")
    print(f"  📊 Delta分布图: {delta_fig_path}")
    print(f"  📊 Waypoint分布图: {waypoint_fig_path}")
    print(f"  📊 统计分析: {stats_file}")
    for i, path in enumerate(comparison_paths):
        time_val = WAYPOINT_TIMES[key_timesteps[i]]
        print(f"  📊 对比图 t={time_val}s: {path}")
    
    print(f"\n📝 图表说明:")
    print(f"  - Delta Actions: 显示每个时间步的运动增量分布")
    print(f"  - Waypoints: 显示累积位置分布（类似你展示的图）")
    print(f"  - 每个子图显示该时刻所有轨迹的(X,Y)分布")
    print(f"  - 统计信息包含均值、标准差、95%分位数等")

if __name__ == "__main__":
    main() 