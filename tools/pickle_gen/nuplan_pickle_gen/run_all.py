#!/usr/bin/env python3
"""
NuPlan数据处理主控脚本
一键执行所有处理步骤
"""

import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """运行单个脚本"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    script_path = os.path.join("/mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess", script_name)
    
    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        elapsed = time.time() - start_time
        print(f"✅ {description} 完成！用时: {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} 失败！用时: {elapsed:.1f}s")
        print(f"错误: {e}")
        return False

def main():
    print("🎯 开始NuPlan数据处理流水线...")
    start_time = time.time()
    
    # 处理步骤
    steps = [
        ("step1_segment_videos.py", "Step 1: 视频分割"),
        ("step2_generate_actions_seq_optimized.py", "Step 2: 生成Actions（优化版）"),
        ("step3a_analyze_displacement_distribution.py", "Step 3A: 分析位移分布"),
        ("step3b_generate_commands.py", "Step 3B: 生成Commands"),
        ("step4_merge_pickle.py", "Step 4: 合并Pickle"),
        ("verify_final.py", "验证: 检查最终结果")
    ]
    
    success_count = 0
    
    for script, description in steps:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\n💥 流水线在 '{description}' 步骤失败！")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"📊 流水线执行结果")
    print(f"{'='*60}")
    print(f"总步骤数: {len(steps)}")
    print(f"成功步骤: {success_count}")
    print(f"总用时: {total_time:.1f}s")
    
    if success_count == len(steps):
        print("🎉 所有步骤成功完成！")
        print("📁 最终文件: /mnt/vdb1/yingyan.li/repo/OmniSim/dataprocess/output/nuplan_processed_data.pkl")
    else:
        print("⚠️ 部分步骤失败，请检查错误信息")

if __name__ == "__main__":
    main() 