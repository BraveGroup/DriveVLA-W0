export PYTHONPATH=/mnt/nvme0n1p1/yingyan.li/repo/DriveVLA-W0/inference/navsim/navsim:/mnt/nvme0n1p1/yingyan.li/repo/DriveVLA-W0:$PYTHONPATH

# 执行推理脚本
/mnt/nvme0n1p1/yingyan.li/miniconda3/envs/navsim/bin/python inference/navsim/navsim/navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  agent=emu_vla_agent \
  agent.experiment_path=/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/logs/train_navsim_qformer_anchor_vava/json_output_debug \
  experiment_name=train_navsim_qformer_anchor_vava/json_output_debug \
  metric_cache_path=data/navsim/metric_cache/test \
