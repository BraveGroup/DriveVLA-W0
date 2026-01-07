#!/bin/bash
# NavSim 指标评估脚本
# 
# 使用方法：
# 1. 直接运行：bash run_emu_vla_navsim_metric_others.sh
# 2. 或者覆盖环境变量后运行：
#    export EXPERIMENT_PATH="/path/to/your/results"
#    export EXPERIMENT_NAME="your_experiment_name"
#    export ANCHOR_BASED="true"  # 或 "false"
#    bash run_emu_vla_navsim_metric_others.sh

# ============================================================================
# 配置区域：在这里设置所有路径和参数
# ============================================================================

# 项目根目录（自动检测，通常不需要修改）
if [ -z "$DRIVEVLA_ROOT" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export DRIVEVLA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# Python 环境（可通过环境变量覆盖）
export PYTHON_ENV="${PYTHON_ENV:-/mnt/nvme0n1p1/yingyan.li/miniconda3/envs/navsim/bin/python}"

# 实验路径和名称（可通过环境变量覆盖）
export EXPERIMENT_PATH="${EXPERIMENT_PATH:-/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/logs/train_navsim_qformer_anchor_vava/json_output_cursor_clean}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-train_navsim_qformer_anchor_vava/json_output_cursor_clean}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-data/navsim/metric_cache/test}"
export ANCHOR_BASED="${ANCHOR_BASED:-true}"

# ============================================================================
# 执行评估
# ============================================================================

# 设置 PYTHONPATH
export PYTHONPATH="${DRIVEVLA_ROOT}/inference/navsim/navsim:${DRIVEVLA_ROOT}:${PYTHONPATH}"

# 切换到项目根目录
cd "$DRIVEVLA_ROOT"

# 执行评估脚本
$PYTHON_ENV inference/navsim/navsim/navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  agent=emu_vla_agent \
  agent.experiment_path="$EXPERIMENT_PATH" \
  agent.anchor_based="$ANCHOR_BASED" \
  experiment_name="$EXPERIMENT_NAME" \
  metric_cache_path="$METRIC_CACHE_PATH"

