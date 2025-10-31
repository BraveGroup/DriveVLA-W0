export PYTHONPATH=/mnt/nvme0n1p1/yingyan.li/repo/OmniSim//inference/navsim/navsim:$PYTHONPATH

# 执行推理脚本
/mnt/nvme0n1p1/yingyan.li/miniconda3/envs/navsim/bin/python inference/navsim/navsim/navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  agent=emu_vla_agent \
  agent.experiment_path=/mnt/nvme0n1p1/yingyan.li/repo/OmniSim/data/navsim/processed_data/jsons/navsim_single_frame_quick_baseline_4k_lr_5e4_vision_loss_use_previous_action \
  experiment_name=navsim_single_frame_quick_baseline_4k_lr_5e4_vision_loss_use_previous_action/json_output \
  metric_cache_path=data/navsim/metric_cache/test \
