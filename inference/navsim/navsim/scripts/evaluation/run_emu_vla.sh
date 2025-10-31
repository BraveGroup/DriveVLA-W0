#!/bin/bash
export PYTHONPATH=/mnt/nvme0n1p1/bing.zhan/repo/OmniSim/inference/navsim/navsim:$PYTHONPATH

# 执行推理脚本
python inference/navsim/navsim/navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  agent=emu_vla_agent \
  agent.experiment_path=/mnt/nvme0n1p1/bing.zhan/repo/OmniSim/logs/navsim_8gpus_two_frame/json_output \
  experiment_name=navsim_8gpus_two_frame/json_output \
  metric_cache_path=data/navsim/metric_cache/test \
