#!/usr/bin/env python3
"""
NuPlan视频分割脚本 - Step 1
将长视频切分成短视频段，考虑action生成的边界条件
检查img和npy文件存在性
严格按照JSON中的images顺序处理，不进行任何排序
"""

import os
import json
import glob
from tqdm import tqdm

# 配置
NUPLAN_JSON_DIR = "/mnt/vdb1/nuplan_json"
NUPLAN_IMAGES_DIR = "/mnt/vdb1/nuplan/images"
NUPLAN_VQ_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/data/nuplan/processed_data/vq_codes_merge"
EXCLUDE_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/data/navsim/processed_data/test_vq_codes"
OUTPUT_DIR = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/intermediate"

SAMPLING_RATE = 10.0  # Hz
SEGMENT_LENGTH_SECONDS = 20
MIN_SEGMENT_SECONDS = 8
FUTURE_HORIZON_SECONDS = 4  # 需要未来4秒来生成waypoints（但现在会用0填充）

def get_exclude_list():
    """获取要排除的序列列表"""
    exclude_set = set()
    if os.path.exists(EXCLUDE_DIR):
        for seq_name in os.listdir(EXCLUDE_DIR):
            if os.path.isdir(os.path.join(EXCLUDE_DIR, seq_name)):
                exclude_set.add(seq_name)
    return exclude_set

def get_all_image_logs():
    """获取images目录下的所有log目录"""
    image_logs = set()
    if not os.path.exists(NUPLAN_IMAGES_DIR):
        print(f"❌ Images目录不存在: {NUPLAN_IMAGES_DIR}")
        return image_logs
    
    for item in os.listdir(NUPLAN_IMAGES_DIR):
        item_path = os.path.join(NUPLAN_IMAGES_DIR, item)
        if os.path.isdir(item_path):
            cam_f0_path = os.path.join(item_path, "CAM_F0")
            if os.path.exists(cam_f0_path):
                image_logs.add(item)
    
    return image_logs

def validate_log_completeness(seq_name):
    """验证单个log的三个组件（images, JSON, VQ）是否都存在"""
    images_path = os.path.join(NUPLAN_IMAGES_DIR, seq_name, "CAM_F0")
    images_exist = os.path.exists(images_path)
    jpg_count = len(glob.glob(os.path.join(images_path, "*.jpg"))) if images_exist else 0
    
    json_path = os.path.join(NUPLAN_JSON_DIR, f"{seq_name}.json")
    json_exists = os.path.exists(json_path)
    
    vq_path = os.path.join(NUPLAN_VQ_DIR, seq_name)
    vq_exists = os.path.exists(vq_path)
    npy_count = len(glob.glob(os.path.join(vq_path, "*.npy"))) if vq_exists else 0
    
    return {
        'seq_name': seq_name,
        'images_exist': images_exist,
        'json_exists': json_exists,
        'vq_exists': vq_exists,
        'jpg_count': jpg_count,
        'npy_count': npy_count,
        'is_complete': images_exist and json_exists and vq_exists,
        'json_path': json_path if json_exists else None
    }

def validate_image_and_vq_paths(image_paths, seq_name):
    """验证图片和对应npy文件是否都存在，严格按照JSON顺序"""
    valid_pairs = []
    missing_img_count = 0
    missing_npz_count = 0
    missing_both_count = 0
    
    # 直接按照JSON中的顺序处理，不进行排序
    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        
        # 图片文件完整路径
        img_full_path = os.path.join(NUPLAN_IMAGES_DIR, seq_name, "CAM_F0", img_filename)
        img_exists = os.path.exists(img_full_path)
        
        # 对应的npy文件路径
        npy_filename = img_filename.replace('.jpg', '.npy')
        npy_full_path = os.path.join(NUPLAN_VQ_DIR, seq_name, npy_filename)
        npy_exists = os.path.exists(npy_full_path)
        
        # 统计缺失情况
        if not img_exists and not npy_exists:
            missing_both_count += 1
        elif not img_exists:
            missing_img_count += 1
        elif not npy_exists:
            missing_npz_count += 1
        else:
            # 两个文件都存在，添加到有效列表
            valid_pairs.append({
                'img_path': os.path.join(seq_name, "CAM_F0", img_filename),
                'npy_path': os.path.join(seq_name, npy_filename)
            })
    
    return valid_pairs, {
        'missing_img': missing_img_count,
        'missing_npz': missing_npz_count,
        'missing_both': missing_both_count,
        'total_missing': missing_img_count + missing_npz_count + missing_both_count
    }

def calculate_valid_segments(total_frames):
    """计算有效的视频段，现在不排除最后的帧（因为会用0填充action）"""
    min_frames = int(MIN_SEGMENT_SECONDS * SAMPLING_RATE)  # 80帧
    segment_frames = int(SEGMENT_LENGTH_SECONDS * SAMPLING_RATE)  # 200帧
    
    if total_frames < min_frames:
        return []
    
    segments = []
    if total_frames >= segment_frames:
        # 切分20秒段
        for start in range(0, total_frames, segment_frames):
            end = min(start + segment_frames, total_frames)
            if end - start >= min_frames:
                segments.append((start, end))
    elif total_frames >= min_frames:
        # 8-20秒，保留为一段
        segments.append((0, total_frames))
    
    return segments

def process_sequence(seq_name, json_path):
    """处理单个序列，严格按照JSON顺序"""
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取JSON失败: {json_path}")
        return [], {}
    
    image_list = json_data.get('images', [])
    if not image_list:
        return [], {}
    
    # 严格按照JSON中的顺序，不进行任何排序
    print(f"📄 {seq_name}: JSON记录{len(image_list)}张图片，按原始顺序处理")
    
    # 验证图片和npy文件路径
    valid_pairs, missing_stats = validate_image_and_vq_paths(image_list, seq_name)
    
    if missing_stats['total_missing'] > 0:
        missing_details = []
        if missing_stats['missing_img'] > 0:
            missing_details.append(f"{missing_stats['missing_img']} 张图片")
        if missing_stats['missing_npz'] > 0:
            missing_details.append(f"{missing_stats['missing_npz']} 个npy文件")
        if missing_stats['missing_both'] > 0:
            missing_details.append(f"{missing_stats['missing_both']} 对文件都缺失")
        
        print(f"⚠️ {seq_name}: 缺失 {' + '.join(missing_details)}")
    
    if len(valid_pairs) == 0:
        return [], missing_stats
    
    # 计算有效分割段
    segments = calculate_valid_segments(len(valid_pairs))
    if not segments:
        return [], missing_stats
    
    # 生成分割段信息
    segment_infos = []
    for seg_idx, (start_frame, end_frame) in enumerate(segments):
        segment_id = f"{seq_name}_seg_{seg_idx:03d}"
        frame_count = end_frame - start_frame
        duration_seconds = frame_count / SAMPLING_RATE
        
        # 提取该段的文件路径（保持JSON原始顺序）
        segment_pairs = valid_pairs[start_frame:end_frame]
        segment_images = [pair['img_path'] for pair in segment_pairs]
        segment_npy = [pair['npy_path'] for pair in segment_pairs]
        
        segment_info = {
            "segment_id": segment_id,
            "original_sequence": seq_name,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": frame_count,
            "duration_seconds": duration_seconds,
            "image_paths": segment_images,
            "npy_paths": segment_npy
        }
        segment_infos.append(segment_info)
    
    return segment_infos, missing_stats

def main():
    print("🚀 开始处理NuPlan视频分割...")
    print("🔍 以images目录为基准，验证JSON和VQ文件存在性...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取排除列表
    exclude_set = get_exclude_list()
    print(f"📋 排除序列数量: {len(exclude_set)}")
    
    # 获取所有image logs（以images为基准）
    image_logs = get_all_image_logs()
    print(f"📁 Images目录总log数: {len(image_logs)}")
    
    # 验证每个log的完整性
    print("🔍 验证各组件存在性...")
    complete_logs = []
    incomplete_logs = []
    
    for seq_name in tqdm(sorted(image_logs), desc="验证完整性"):
        if seq_name in exclude_set:
            continue
        
        validation = validate_log_completeness(seq_name)
        if validation['is_complete']:
            complete_logs.append(validation)
        else:
            incomplete_logs.append(validation)
    
    print(f"✅ 完整的log数: {len(complete_logs)}")
    print(f"❌ 不完整的log数: {len(incomplete_logs)}")
    
    # 显示不完整log的详情
    if incomplete_logs:
        print("\n⚠️ 不完整的log详情:")
        for item in incomplete_logs[:10]:  # 只显示前10个
            missing_parts = []
            if not item['images_exist']:
                missing_parts.append("Images")
            if not item['json_exists']:
                missing_parts.append("JSON")
            if not item['vq_exists']:
                missing_parts.append("VQ")
            
            print(f"  {item['seq_name']}: 缺失 {', '.join(missing_parts)}")
            print(f"    JPG: {item['jpg_count']}, NPY: {item['npy_count']}")
        
        if len(incomplete_logs) > 10:
            print(f"    ... 还有 {len(incomplete_logs) - 10} 个不完整log")
    
    # 处理完整的log
    stats = {
        'total_image_logs': len(image_logs),
        'excluded_sequences': len([seq for seq in image_logs if seq in exclude_set]),
        'complete_logs': len(complete_logs),
        'incomplete_logs': len(incomplete_logs),
        'processed_sequences': 0,
        'total_segments': 0,
        'file_stats': {
            'total_img_files': 0,
            'total_npy_files': 0,
            'missing_img_files': 0,
            'missing_npy_files': 0,
            'missing_both_files': 0,
            'valid_pairs': 0
        }
    }
    
    all_segments = []
    
    # 处理每个完整的log
    for validation in tqdm(complete_logs, desc="处理序列"):
        seq_name = validation['seq_name']
        json_path = validation['json_path']
        
        segments, missing_stats = process_sequence(seq_name, json_path)
        
        # 累积文件统计
        stats['file_stats']['missing_img_files'] += missing_stats.get('missing_img', 0)
        stats['file_stats']['missing_npy_files'] += missing_stats.get('missing_npz', 0)
        stats['file_stats']['missing_both_files'] += missing_stats.get('missing_both', 0)
        
        if segments:
            all_segments.extend(segments)
            stats['processed_sequences'] += 1
            stats['total_segments'] += len(segments)
    
    # 计算最终的文件统计
    total_img_files = 0
    total_npy_files = 0
    
    for segment in all_segments:
        total_img_files += len(segment['image_paths'])
        total_npy_files += len(segment['npy_paths'])
    
    stats['file_stats']['total_img_files'] = total_img_files
    stats['file_stats']['total_npy_files'] = total_npy_files
    stats['file_stats']['valid_pairs'] = total_img_files  # 应该相等
    
    # 保存结果
    output_data = {
        "metadata": {
            "sampling_rate": SAMPLING_RATE,
            "segment_length_seconds": SEGMENT_LENGTH_SECONDS,
            "min_segment_seconds": MIN_SEGMENT_SECONDS,
            "future_horizon_seconds": FUTURE_HORIZON_SECONDS,
            "processing_approach": "images_based_validation",
            "vq_source": "vq_codes_merge",
            "processing_stats": stats
        },
        "segments": all_segments
    }
    
    output_file = os.path.join(OUTPUT_DIR, "video_segments.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # 统计信息
    print(f"\n📊 处理结果:")
    print(f"Images总log数: {stats['total_image_logs']}")
    print(f"排除序列数: {stats['excluded_sequences']}")
    print(f"完整log数: {stats['complete_logs']}")
    print(f"不完整log数: {stats['incomplete_logs']}")
    print(f"成功处理: {stats['processed_sequences']}")
    print(f"生成段数: {stats['total_segments']}")
    
    print(f"\n📁 文件统计:")
    print(f"有效img文件: {stats['file_stats']['total_img_files']:,}")
    print(f"有效npy文件: {stats['file_stats']['total_npy_files']:,}")
    print(f"有效文件对: {stats['file_stats']['valid_pairs']:,}")
    
    if any(v > 0 for v in [stats['file_stats']['missing_img_files'], 
                          stats['file_stats']['missing_npy_files'], 
                          stats['file_stats']['missing_both_files']]):
        print(f"\n⚠️ 缺失文件统计:")
        if stats['file_stats']['missing_img_files'] > 0:
            print(f"缺失img文件: {stats['file_stats']['missing_img_files']:,}")
        if stats['file_stats']['missing_npy_files'] > 0:
            print(f"缺失npy文件: {stats['file_stats']['missing_npy_files']:,}")
        if stats['file_stats']['missing_both_files'] > 0:
            print(f"两者都缺失: {stats['file_stats']['missing_both_files']:,}")
    
    if all_segments:
        import numpy as np
        durations = [seg['duration_seconds'] for seg in all_segments]
        print(f"\n⏱️ 段长统计:")
        print(f"平均段长: {np.mean(durations):.1f}s")
        print(f"最短段长: {np.min(durations):.1f}s")
        print(f"最长段长: {np.max(durations):.1f}s")
    
    print(f"\n💾 结果保存到: {output_file}")
    print("✅ Step 1 完成！")

if __name__ == "__main__":
    main() 