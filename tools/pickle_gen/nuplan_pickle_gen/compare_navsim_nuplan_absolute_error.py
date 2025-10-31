#!/usr/bin/env python3
"""
NavSim vs NuPlan 绝对误差对比脚本 - 简化版
专注于x、y、yaw绝对误差统计分析
"""

import os
import json
import numpy as np
from tqdm import tqdm
import pickle
import sys
from pyquaternion import Quaternion

# 添加必要的路径
sys.path.append("/mnt/vdb1/yingyan.li/repo/OmniSim/tools/pickle_gen")
from navsim_coor import StateSE2, convert_absolute_to_relative_se2_array

# 配置路径
SEGMENTS_JSON = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/video_segments.json"
ACTIONS_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/actions"
NUPLAN_JSON_DIR = "/mnt/vdb1/nuplan_json"
NAVSIM_LOGS_PATH = '/mnt/vdb1/yingyan.li/repo/OmniSim/data/navsim/navsim_logs/trainval'
OUTPUT_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/analysis/navsim_comparison"

# Action参数
WAYPOINT_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

def load_segments():
    """加载视频分割结果"""
    print("📋 加载视频分割结果...")
    with open(SEGMENTS_JSON, 'r') as f:
        data = json.load(f)
    return data['segments'], data['metadata']

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

def load_segment_actions(segment_id):
    """加载单个segment的保存的actions"""
    actions_file = os.path.join(ACTIONS_DIR, f"{segment_id}.npy")
    if os.path.exists(actions_file):
        return np.load(actions_file)  # (frame_count, 8, 3)
    return None

def compare_segment_with_navsim(segment, navsim_data):
    """与NavSim数据进行详细比较"""
    if navsim_data is None:
        return None
    
    segment_id = segment['segment_id']
    seq_name = segment['original_sequence']
    start_frame = segment['start_frame']
    frame_count = segment['frame_count']
    
    # 加载NuPlan actions
    nuplan_actions = load_segment_actions(segment_id)
    if nuplan_actions is None:
        return None
    
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
    
    # 比较所有帧
    comparisons = []
    matched_count = 0
    
    for local_idx in range(frame_count):
        global_frame_idx = start_frame + local_idx
        if global_frame_idx < len(images):
            image_name = extract_image_name(images[global_frame_idx])
            
            if image_name and image_name in navsim_data['image_to_data']:
                nuplan_frame_actions = nuplan_actions[local_idx]  # (8, 3)
                navsim_frame_actions = navsim_data['image_to_data'][image_name]['actions']  # (8, 3)
                
                # 计算绝对误差
                abs_diff = np.abs(nuplan_frame_actions - navsim_frame_actions)  # (8, 3)
                
                # 计算总体误差（所有时间点的平均）
                overall_error = np.mean(abs_diff)
                
                comparison = {
                    'segment_id': segment_id,
                    'image_name': image_name,
                    'local_frame_idx': local_idx,
                    'global_frame_idx': global_frame_idx,
                    'nuplan_actions': nuplan_frame_actions.tolist(),
                    'navsim_actions': navsim_frame_actions.tolist(),
                    'abs_diff': abs_diff.tolist(),
                    'overall_error': float(overall_error),
                    # 分时间点、分方向的误差
                    'timestep_errors': {
                        f't_{WAYPOINT_TIMES[t]}s': {
                            'x_error': float(abs_diff[t, 0]),
                            'y_error': float(abs_diff[t, 1]),
                            'yaw_error': float(abs_diff[t, 2])
                        } for t in range(8)
                    }
                }
                
                comparisons.append(comparison)
                matched_count += 1
    
    if matched_count == 0:
        return None
    
    return {
        'segment_id': segment_id,
        'seq_name': seq_name,
        'total_frames': frame_count,
        'matched_frames': matched_count,
        'match_ratio': matched_count / frame_count,
        'comparisons': comparisons
    }

def analyze_absolute_errors(all_comparison_results):
    """分析所有的绝对误差统计"""
    print("📊 分析绝对误差统计...")
    
    if not all_comparison_results:
        return None
    
    # 收集所有误差数据
    all_comparisons = []
    for result in all_comparison_results:
        all_comparisons.extend(result['comparisons'])
    
    total_comparisons = len(all_comparisons)
    
    # 初始化统计数据结构
    stats = {
        'total_comparisons': total_comparisons,
        'total_segments': len(all_comparison_results),
        'timestep_stats': {}
    }
    
    # 分时间点统计
    for t in range(8):
        time_val = WAYPOINT_TIMES[t]
        time_key = f't_{time_val}s'
        
        # 收集该时间点的所有误差
        x_errors = []
        y_errors = []
        yaw_errors = []
        
        for comp in all_comparisons:
            timestep_error = comp['timestep_errors'][time_key]
            x_errors.append(timestep_error['x_error'])
            y_errors.append(timestep_error['y_error'])
            yaw_errors.append(timestep_error['yaw_error'])
        
        # 转换为numpy数组便于计算
        x_errors = np.array(x_errors)
        y_errors = np.array(y_errors)
        yaw_errors = np.array(yaw_errors)
        
        # 计算统计量
        stats['timestep_stats'][time_key] = {
            'x_direction': {
                'mean': float(np.mean(x_errors)),
                'std': float(np.std(x_errors)),
                'max': float(np.max(x_errors)),
                'min': float(np.min(x_errors)),
                'median': float(np.median(x_errors)),
                'percentile_95': float(np.percentile(x_errors, 95)),
                'percentile_99': float(np.percentile(x_errors, 99))
            },
            'y_direction': {
                'mean': float(np.mean(y_errors)),
                'std': float(np.std(y_errors)),
                'max': float(np.max(y_errors)),
                'min': float(np.min(y_errors)),
                'median': float(np.median(y_errors)),
                'percentile_95': float(np.percentile(y_errors, 95)),
                'percentile_99': float(np.percentile(y_errors, 99))
            },
            'yaw_direction': {
                'mean': float(np.mean(yaw_errors)),
                'std': float(np.std(yaw_errors)),
                'max': float(np.max(yaw_errors)),
                'min': float(np.min(yaw_errors)),
                'median': float(np.median(yaw_errors)),
                'percentile_95': float(np.percentile(yaw_errors, 95)),
                'percentile_99': float(np.percentile(yaw_errors, 99))
            }
        }
    
    # 整体统计（所有时间点平均）
    all_x_errors = []
    all_y_errors = []
    all_yaw_errors = []
    
    for comp in all_comparisons:
        for t in range(8):
            time_key = f't_{WAYPOINT_TIMES[t]}s'
            timestep_error = comp['timestep_errors'][time_key]
            all_x_errors.append(timestep_error['x_error'])
            all_y_errors.append(timestep_error['y_error'])
            all_yaw_errors.append(timestep_error['yaw_error'])
    
    all_x_errors = np.array(all_x_errors)
    all_y_errors = np.array(all_y_errors)
    all_yaw_errors = np.array(all_yaw_errors)
    
    stats['overall_stats'] = {
        'x_direction': {
            'mean': float(np.mean(all_x_errors)),
            'std': float(np.std(all_x_errors)),
            'max': float(np.max(all_x_errors)),
            'min': float(np.min(all_x_errors)),
            'median': float(np.median(all_x_errors)),
            'percentile_95': float(np.percentile(all_x_errors, 95)),
            'percentile_99': float(np.percentile(all_x_errors, 99))
        },
        'y_direction': {
            'mean': float(np.mean(all_y_errors)),
            'std': float(np.std(all_y_errors)),
            'max': float(np.max(all_y_errors)),
            'min': float(np.min(all_y_errors)),
            'median': float(np.median(all_y_errors)),
            'percentile_95': float(np.percentile(all_y_errors, 95)),
            'percentile_99': float(np.percentile(all_y_errors, 99))
        },
        'yaw_direction': {
            'mean': float(np.mean(all_yaw_errors)),
            'std': float(np.std(all_yaw_errors)),
            'max': float(np.max(all_yaw_errors)),
            'min': float(np.min(all_yaw_errors)),
            'median': float(np.median(all_yaw_errors)),
            'percentile_95': float(np.percentile(all_yaw_errors, 95)),
            'percentile_99': float(np.percentile(all_yaw_errors, 99))
        }
    }
    
    # 找出误差最大的样本
    sorted_comparisons = sorted(all_comparisons, key=lambda x: x['overall_error'], reverse=True)
    top_error_samples = sorted_comparisons[:10]  # 取前10个误差最大的样本
    
    return stats, top_error_samples

def print_error_statistics(stats):
    """打印误差统计结果"""
    print(f"\n📊 绝对误差统计结果:")
    print(f"  总对比数: {stats['total_comparisons']}")
    print(f"  成功segments: {stats['total_segments']}")
    
    print(f"\n📊 整体统计 (所有时间点平均):")
    for direction in ['x_direction', 'y_direction', 'yaw_direction']:
        dir_stats = stats['overall_stats'][direction]
        unit = 'm' if direction != 'yaw_direction' else 'rad'
        print(f"  {direction.upper().replace('_', ' ')}:")
        print(f"    平均值: {dir_stats['mean']:.6f}{unit}")
        print(f"    标准差: {dir_stats['std']:.6f}{unit}")
        print(f"    最大值: {dir_stats['max']:.6f}{unit}")
        print(f"    最小值: {dir_stats['min']:.6f}{unit}")
        print(f"    中位数: {dir_stats['median']:.6f}{unit}")
    
    print(f"\n📊 分时间点统计:")
    for t in range(8):
        time_val = WAYPOINT_TIMES[t]
        time_key = f't_{time_val}s'
        print(f"\n  时间点 {time_val}s:")
        
        for direction in ['x_direction', 'y_direction', 'yaw_direction']:
            dir_stats = stats['timestep_stats'][time_key][direction]
            unit = 'm' if direction != 'yaw_direction' else 'rad'
            print(f"    {direction.upper().replace('_', ' ')}:")
            print(f"      平均值: {dir_stats['mean']:.6f}{unit}, 标准差: {dir_stats['std']:.6f}{unit}")
            print(f"      最大值: {dir_stats['max']:.6f}{unit}, 最小值: {dir_stats['min']:.6f}{unit}")

def main():
    print("🚀 开始NavSim vs NuPlan绝对误差对比分析（简化版）...")
    print("📝 专注于x、y、yaw绝对误差统计")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载segments
    segments, metadata = load_segments()
    print(f"📁 总segment数: {len(segments)}")
    
    # 检查NavSim数据可用性
    print("🔍 检查NavSim数据可用性...")
    unique_seqs = list(set([s['original_sequence'] for s in segments]))
    print(f"📊 NuPlan unique sequences: {len(unique_seqs)}")
    
    # 检查NavSim数据
    navsim_files = os.listdir(NAVSIM_LOGS_PATH) if os.path.exists(NAVSIM_LOGS_PATH) else []
    navsim_seqs = [f.replace('.pkl', '') for f in navsim_files if f.endswith('.pkl')]
    navsim_seq_set = set(navsim_seqs)
    
    # 寻找共同序列并加载NavSim数据
    navsim_data_cache = {}
    common_seqs = []
    
    for seq_name in tqdm(unique_seqs[:100], desc="加载NavSim数据"):  # 限制前100个序列
        if seq_name in navsim_seq_set:
            navsim_data = load_navsim_actions(seq_name)
            if navsim_data is not None:
                navsim_data_cache[seq_name] = navsim_data
                common_seqs.append(seq_name)
    
    print(f"📊 找到共同序列: {len(common_seqs)}")
    
    if len(common_seqs) == 0:
        print("❌ 没有找到共同序列，无法进行对比")
        return
    
    # 对比分析
    all_comparison_results = []
    total_segments = 0
    successful_comparisons = 0
    
    print("🔄 开始逐个segment对比...")
    for segment in tqdm(segments, desc="对比分析"):
        seq_name = segment['original_sequence']
        if seq_name in navsim_data_cache:
            total_segments += 1
            result = compare_segment_with_navsim(segment, navsim_data_cache[seq_name])
            if result is not None:
                all_comparison_results.append(result)
                successful_comparisons += 1
    
    print(f"✅ 成功对比: {successful_comparisons}/{total_segments} 个segment")
    
    if not all_comparison_results:
        print("❌ 没有成功的对比结果")
        return
    
    # 分析绝对误差
    stats, top_error_samples = analyze_absolute_errors(all_comparison_results)
    
    # 打印统计结果
    print_error_statistics(stats)
    
    # 保存统计结果
    stats_file = os.path.join(OUTPUT_DIR, "absolute_error_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 保存误差最大的样本
    top_errors_file = os.path.join(OUTPUT_DIR, "top_error_samples.json")
    with open(top_errors_file, 'w') as f:
        json.dump({
            'description': '误差最大的10个样本，包含完整的actions数据',
            'samples': top_error_samples
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 绝对误差对比分析完成！")
    print(f"📊 对比了 {stats['total_comparisons']} 个样本")
    print(f"\n💾 结果文件:")
    print(f"  📊 统计结果: {stats_file}")
    print(f"  📊 最大误差样本: {top_errors_file}")
    
    print(f"\n🎯 关键指标总结:")
    overall = stats['overall_stats']
    print(f"  X方向平均误差: {overall['x_direction']['mean']:.6f}m ± {overall['x_direction']['std']:.6f}m")
    print(f"  Y方向平均误差: {overall['y_direction']['mean']:.6f}m ± {overall['y_direction']['std']:.6f}m")
    print(f"  Yaw方向平均误差: {overall['yaw_direction']['mean']:.6f}rad ± {overall['yaw_direction']['std']:.6f}rad")

if __name__ == "__main__":
    main() 