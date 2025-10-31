#!/usr/bin/env python3
"""
NuPlan Pose质量检查脚本
检查原始JSON文件中的pose数据质量问题，识别坏帧并记录
只检查video_segments.json中实际使用的序列
"""

import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 配置
NUPLAN_JSON_DIR = "/mnt/vdb1/nuplan_json"
SEGMENTS_JSON = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate/video_segments.json"
OUTPUT_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/analysis/pose_quality"
REPORT_FILE = "pose_quality_report.json"
SUMMARY_FILE = "pose_quality_summary.txt"

# 阈值配置
THRESHOLDS = {
    'max_displacement_per_frame': 5.0,      # 最大帧间位移 (m)
    'max_rotation_per_frame': 0.3,          # 最大帧间旋转 (rad)
    'max_z_displacement': 2.0,              # 最大Z轴位移 (m)
    'max_velocity': 50.0,                   # 最大速度 (m/s, assuming 10Hz)
    'max_acceleration': 10.0,               # 最大加速度 (m/s²)
    'min_det_threshold': 0.01,              # 变换矩阵行列式最小值
    'max_det_threshold': 100.0,             # 变换矩阵行列式最大值
}

def load_used_sequences():
    """加载video_segments.json中实际使用的序列名称"""
    print("📋 加载实际使用的序列...")
    with open(SEGMENTS_JSON, 'r') as f:
        data = json.load(f)
    
    used_sequences = set([seg['original_sequence'] for seg in data['segments']])
    print(f"  📊 实际使用的序列数量: {len(used_sequences)}")
    return used_sequences

def validate_transformation_matrix(pose):
    """验证4x4变换矩阵的数学有效性"""
    try:
        pose = np.array(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            return False, "Invalid shape"
        
        # 检查旋转矩阵部分 (3x3)
        rotation = pose[:3, :3]
        
        # 计算行列式，应该接近1
        det = np.linalg.det(rotation)
        if abs(det - 1.0) > 0.1:
            return False, f"Invalid rotation determinant: {det:.4f}"
        
        # 检查是否为正交矩阵 R^T * R = I
        should_be_identity = rotation.T @ rotation
        identity = np.eye(3)
        if not np.allclose(should_be_identity, identity, atol=0.1):
            return False, "Not orthogonal matrix"
        
        # 检查是否有NaN或inf
        if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
            return False, "Contains NaN or inf"
        
        # 检查底行是否为 [0, 0, 0, 1]
        if not np.allclose(pose[3, :], [0, 0, 0, 1], atol=0.01):
            return False, f"Invalid bottom row: {pose[3, :]}"
        
        return True, "Valid"
    
    except Exception as e:
        return False, f"Exception: {str(e)}"

def calculate_pose_metrics(poses):
    """计算pose序列的各种指标"""
    poses_array = np.array(poses, dtype=np.float64)
    num_frames = len(poses_array)
    
    if num_frames < 2:
        return {}
    
    # 提取位置和旋转
    positions = poses_array[:, :3, 3]  # (N, 3)
    
    # 计算帧间位移
    displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)  # (N-1,)
    
    # 计算速度 (假设10Hz采样)
    velocities = displacements * 10.0  # m/s
    
    # 计算加速度
    accelerations = np.diff(velocities) * 10.0 if len(velocities) > 1 else np.array([])
    
    # 计算旋转角度变化
    rotation_changes = []
    for i in range(num_frames - 1):
        R1 = poses_array[i, :3, :3]
        R2 = poses_array[i + 1, :3, :3]
        # 计算相对旋转
        R_rel = R1.T @ R2
        # 提取旋转角度
        trace = np.trace(R_rel)
        # 避免数值误差
        trace = np.clip(trace, -1, 3)
        angle = np.arccos((trace - 1) / 2)
        rotation_changes.append(angle)
    
    rotation_changes = np.array(rotation_changes)
    
    return {
        'positions': positions,
        'displacements': displacements,
        'velocities': velocities,
        'accelerations': accelerations,
        'rotation_changes': rotation_changes,
        'z_positions': positions[:, 2],
        'z_changes': np.diff(positions[:, 2])
    }

def check_sequence_quality(seq_name, poses):
    """检查一个sequence的pose质量 - 简化版"""
    issues = []
    
    # 设置阈值 - pose中x、y位移超过此值就记录
    POSE_TRANSLATION_THRESHOLD = 5.0
    
    # 检查每个pose的x、y位移是否超过阈值
    for i, pose in enumerate(poses):
        try:
            pose_array = np.array(pose, dtype=np.float64)
            
            # 提取位置信息 (x, y, z) - pose矩阵的平移部分
            position = pose_array[:3, 3]
            x, y, z = position
            
            # 检查x、y位移是否超过阈值
            if abs(x) > POSE_TRANSLATION_THRESHOLD or abs(y) > POSE_TRANSLATION_THRESHOLD:
                issues.append({
                    'type': 'large_pose_translation',
                    'frame': i,
                    'position': [float(x), float(y), float(z)],
                    'threshold': POSE_TRANSLATION_THRESHOLD,
                    'message': f"Frame {i}: Large pose translation x={x:.3f}, y={y:.3f} (threshold={POSE_TRANSLATION_THRESHOLD})",
                    'original_pose': pose  # 保存完整的原始pose
                })
        except Exception as e:
            # 如果pose格式有问题，也记录
            issues.append({
                'type': 'invalid_pose_format',
                'frame': i,
                'message': f"Frame {i}: Invalid pose format - {str(e)}",
                'original_pose': pose  # 保存原始pose
            })
    
    # 简单的统计信息
    metrics = {
        'total_frames': len(poses),
        'problematic_frames': len(issues)
    }
    
    return issues, metrics

def generate_visualizations(all_metrics, bad_sequences, output_dir):
    """生成可视化图表 - 简化版"""
    print("📊 生成可视化图表...")
    
    # 简化版的可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('NuPlan Pose Quality Analysis (Simplified)', fontsize=16, fontweight='bold')
    
    # 1. 序列问题统计
    total_sequences = len(all_metrics)
    problem_sequences = len(bad_sequences)
    good_sequences = total_sequences - problem_sequences
    
    labels = ['Good Sequences', 'Problem Sequences']
    sizes = [good_sequences, problem_sequences]
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Sequence Quality Distribution')
    
    # 2. 问题类型统计
    issue_types = defaultdict(int)
    for seq_issues in bad_sequences.values():
        for issue in seq_issues:
            issue_types[issue['type']] += 1
    
    if issue_types:
        types = list(issue_types.keys())
        counts = list(issue_types.values())
        axes[1].bar(types, counts, alpha=0.7, edgecolor='black', color='#3498db')
        axes[1].set_xlabel('Issue Type')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Distribution of Pose Issues')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No issues found', ha='center', va='center', 
                    transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Distribution of Pose Issues')
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, "pose_quality_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 图表已保存: {plot_file}")

def main():
    print("🔍 开始NuPlan Pose质量检查 (仅检查实际使用的序列)...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载实际使用的序列
    used_sequences = load_used_sequences()
    
    # 检查结果
    all_results = {}
    bad_sequences = {}
    all_metrics = {}
    total_issues = 0
    missing_sequences = []
    
    print("🔄 逐个检查实际使用的序列...")
    for seq_name in tqdm(used_sequences, desc="检查序列"):
        json_path = os.path.join(NUPLAN_JSON_DIR, f"{seq_name}.json")
        
        if not os.path.exists(json_path):
            print(f"⚠️ 序列文件不存在: {seq_name}")
            missing_sequences.append(seq_name)
            continue
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            poses = data.get('poses', [])
            if not poses:
                continue
            
            # 检查质量
            issues, metrics = check_sequence_quality(seq_name, poses)
            
            # 记录结果
            all_results[seq_name] = {
                'total_frames': len(poses),
                'issues_count': len(issues),
                'issues': issues
            }
            
            if metrics:
                all_metrics[seq_name] = metrics
            
            if issues:
                bad_sequences[seq_name] = issues
                total_issues += len(issues)
        
        except Exception as e:
            print(f"❌ 处理 {seq_name} 时出错: {str(e)}")
            all_results[seq_name] = {
                'error': str(e)
            }
    
    # 生成统计报告
    print("📋 生成统计报告...")
    
    # 统计信息
    total_sequences = len(all_results)
    bad_sequences_count = len(bad_sequences)
    good_sequences_count = total_sequences - bad_sequences_count
    
    # 问题类型统计
    issue_type_stats = defaultdict(int)
    severity_stats = {'critical': 0, 'warning': 0}
    
    for seq_name, issues in bad_sequences.items():
        for issue in issues:
            issue_type_stats[issue['type']] += 1
            
            # 判断严重程度
            if issue['type'] in ['invalid_matrix', 'excessive_displacement', 'excessive_velocity']:
                severity_stats['critical'] += 1
            else:
                severity_stats['warning'] += 1
    
    # 生成详细报告
    report_data = {
        'summary': {
            'total_used_sequences': len(used_sequences),
            'total_checked_sequences': total_sequences,
            'missing_sequences': len(missing_sequences),
            'good_sequences': good_sequences_count,
            'bad_sequences': bad_sequences_count,
            'total_issues': total_issues,
            'bad_sequence_ratio': bad_sequences_count / total_sequences if total_sequences > 0 else 0,
            'issue_type_stats': dict(issue_type_stats),
            'severity_stats': dict(severity_stats)
        },
        'thresholds': THRESHOLDS,
        'missing_sequences': missing_sequences,
        'sequence_details': all_results,
        'bad_sequences': bad_sequences
    }
    
    # 保存详细报告
    report_path = os.path.join(OUTPUT_DIR, REPORT_FILE)
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    # 生成摘要文本
    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_FILE)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("NuPlan Pose质量检查报告 (实际使用的序列)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 总体统计:\n")
        f.write(f"  实际使用的序列数: {len(used_sequences):,}\n")
        f.write(f"  成功检查的序列数: {total_sequences:,}\n")
        f.write(f"  缺失的序列数: {len(missing_sequences):,}\n")
        f.write(f"  正常序列数: {good_sequences_count:,} ({good_sequences_count/total_sequences*100:.1f}%)\n")
        f.write(f"  问题序列数: {bad_sequences_count:,} ({bad_sequences_count/total_sequences*100:.1f}%)\n")
        f.write(f"  总问题数: {total_issues:,}\n\n")
        
        if missing_sequences:
            f.write("❌ 缺失的序列文件:\n")
            for seq in missing_sequences[:10]:  # 只显示前10个
                f.write(f"  {seq}\n")
            if len(missing_sequences) > 10:
                f.write(f"  ... 还有 {len(missing_sequences) - 10} 个缺失序列\n")
            f.write("\n")
        
        f.write("🔍 问题类型统计:\n")
        for issue_type, count in sorted(issue_type_stats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {issue_type}: {count:,}\n")
        f.write("\n")
        
        f.write("⚠️ 严重程度统计:\n")
        f.write(f"  严重问题: {severity_stats['critical']:,}\n")
        f.write(f"  警告问题: {severity_stats['warning']:,}\n\n")
        
        f.write("🎯 检查阈值:\n")
        for key, value in THRESHOLDS.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("🚨 最严重的问题序列 (前10个):\n")
        sorted_bad = sorted(bad_sequences.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (seq_name, issues) in enumerate(sorted_bad[:10]):
            f.write(f"  {i+1}. {seq_name}: {len(issues)} 个问题\n")
        
        if bad_sequences:
            f.write("\n📝 典型问题示例:\n")
            example_shown = set()
            for seq_name, issues in sorted_bad[:3]:
                for issue in issues[:2]:  # 每个序列最多显示2个问题
                    issue_type = issue['type']
                    if issue_type not in example_shown:
                        f.write(f"  {issue_type}: {issue.get('message', 'No message')}\n")
                        example_shown.add(issue_type)
    
    # 生成可视化
    generate_visualizations(all_metrics, bad_sequences, OUTPUT_DIR)
    
    # 打印结果
    print("\n" + "="*60)
    print("🎯 检查完成！")
    print("="*60)
    print(f"📊 实际使用的序列数: {len(used_sequences):,}")
    print(f"📊 成功检查的序列数: {total_sequences:,}")
    if missing_sequences:
        print(f"❌ 缺失的序列数: {len(missing_sequences):,}")
    print(f"✅ 正常序列: {good_sequences_count:,} ({good_sequences_count/total_sequences*100:.1f}%)")
    print(f"❌ 问题序列: {bad_sequences_count:,} ({bad_sequences_count/total_sequences*100:.1f}%)")
    print(f"🚨 总问题数: {total_issues:,}")
    
    if bad_sequences_count > 0:
        print(f"\n🔥 最严重的问题序列:")
        sorted_bad = sorted(bad_sequences.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (seq_name, issues) in enumerate(sorted_bad[:5]):
            print(f"  {i+1}. {seq_name}: {len(issues)} 个问题")
    
    print(f"\n💾 结果已保存:")
    print(f"  📄 详细报告: {report_path}")
    print(f"  📝 摘要报告: {summary_path}")
    print(f"  📊 可视化图表: {OUTPUT_DIR}/pose_quality_analysis.png")

if __name__ == "__main__":
    main() 