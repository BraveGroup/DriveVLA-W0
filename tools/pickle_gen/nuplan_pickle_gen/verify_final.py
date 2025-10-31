#!/usr/bin/env python3
"""
验证最终pickle文件的完整性和格式
"""

import os
import pickle
import numpy as np

PICKLE_FILE = "/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/output/nuplan_processed_data.pkl"

def verify_pickle():
    print("🔍 验证Pickle文件...")
    
    if not os.path.exists(PICKLE_FILE):
        print(f"❌ 文件不存在: {PICKLE_FILE}")
        return
    
    # 加载pickle
    try:
        with open(PICKLE_FILE, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    print(f"✅ 成功加载，段数: {len(data)}")
    
    # 验证格式
    sample = data[0]
    required_keys = ["segment_id", "image", "action", "text"]
    
    for key in required_keys:
        if key not in sample:
            print(f"❌ 缺少key: {key}")
            return
    
    print("✅ 格式正确")
    
    # 验证数据类型和形状
    print(f"\n📊 数据分析:")
    total_frames = 0
    action_shapes = []
    
    for i, segment in enumerate(data[:5]):  # 检查前5个
        images = segment["image"]
        actions = segment["action"]
        texts = segment["text"]
        
        print(f"段 {i}: 帧数={len(images)}, Action形状={actions.shape}")
        
        total_frames += len(images)
        action_shapes.append(actions.shape)
        
        # 验证长度一致性
        if len(images) != len(actions) or len(actions) != len(texts):
            print(f"❌ 段 {i} 长度不一致")
            return
    
    print(f"✅ 数据一致性检查通过")
    print(f"总段数: {len(data)}")
    print(f"前5段总帧数: {total_frames}")
    
    # 验证action范围
    sample_actions = data[0]["action"]
    print(f"\nAction统计:")
    print(f"形状: {sample_actions.shape}")
    print(f"范围: [{sample_actions.min():.3f}, {sample_actions.max():.3f}]")
    print(f"均值: {sample_actions.mean():.3f}")
    print(f"标准差: {sample_actions.std():.3f}")
    
    # Command统计
    all_commands = []
    for segment in data:
        all_commands.extend(segment["text"])
    
    from collections import Counter
    cmd_counts = Counter(all_commands)
    
    print(f"\nCommand分布:")
    for cmd, count in cmd_counts.items():
        print(f"{cmd}: {count}")
    
    print("✅ 验证完成！")

if __name__ == "__main__":
    verify_pickle() 