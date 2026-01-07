#!/usr/bin/env python3
import pickle
import os

# 加载pickle文件
pickle_path = "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s.pkl"

print(f"Loading pickle file: {pickle_path}")
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

print(f"Data type: {type(data)}")
print(f"Data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")

# 检查前几个样本的结构
if isinstance(data, list) and len(data) > 0:
    print("\nFirst sample structure:")
    sample = data[0]
    print(f"Sample type: {type(sample)}")
    print(f"Sample keys: {sample.keys() if hasattr(sample, 'keys') else 'No keys'}")

    if 'image' in sample:
        print(f"Image paths: {sample['image'][:3]}...")  # 只显示前3个
        for i, img_path in enumerate(sample['image'][:3]):
            print(f"  Image {i}: {img_path}")
            print(f"    Exists: {os.path.exists(img_path)}")

    if 'action' in sample:
        print(f"Action shape: {sample['action'].shape if hasattr(sample['action'], 'shape') else 'No shape'}")
