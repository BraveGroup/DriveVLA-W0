#!/usr/bin/env bash

WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=23456
NGPUS=8

DATAPATH='/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl'
ACTION_TOKENIZER_PATH="/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/pretrained_models/fast"
EXP_NAME=train_navsim_flow_matching


MODEL_NAME_OR_PATH="/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/logs/train_nuplan_6va_v0.2_multi_node" # Emu3_NuPlan_Pretrain_Cktps
MODEL_CONFIG_PATH="/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/configs/pi0_fast_video.json"


export PYTHONPATH=$(pwd)
export PYTHONPATH="/mnt/nvme0n1p1/yingyan.li/repo/DriveVLA-W0:$PYTHONPATH"

torchrun \
  --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/train_pi0.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --model_config_path ${MODEL_CONFIG_PATH} \
  --actions_format fast \
  --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
  --deepspeed scripts/sft/zero3_offload.json \
  --output_dir logs/${EXP_NAME} \
  --learning_rate 5e-5 \
  --null_prompt_prob 0.15 \
  --weight_decay 0.1 \
  --min_learning_rate 1e-6 \
  --max_grad_norm 5.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-6 \
  --bf16 True \
  --tf32 True \
  --data_path ${DATAPATH} \
  --freeze_vlm False \
  --max_steps 10000 \
  --dataloader_num_workers 12 \
  --lr_scheduler_type cosine_with_min_lr \
  --warmup_steps 50 \
  --per_device_train_batch_size 12 \
  --frames 1 \
  --action_frames 8 \
  --max_position_embeddings 1400 \
  --seed 0 \
  --logging_steps 10 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 5000 \
  --eval_strategy no \
  --apply_loss_on_only_vision True \
  --apply_loss_on_only_action False \
  --actions True \
  --use_gripper False \
  --driving True \
  --evaluation_strategy steps \
  --eval_steps 400 \
  --per_device_eval_batch_size 4 \
  --eval_accumulation_steps 1 \
  --use_previous_actions True \
  --action_dim 3 \
  --train_action_only False \
  --action_loss_weight 1.0 \
  --action_sample_steps 10 \
  --report_to tensorboard \

# inference
#  torchrun --nproc_per_node=${NGPUS} inference/vla/inference_action_navsim_flow_matching_vava.py \
#      --emu_hub logs/${EXP_NAME} \
#      --output_dir logs/${EXP_NAME}/json_output \
#      --test_data_pkl /mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s.pkl
