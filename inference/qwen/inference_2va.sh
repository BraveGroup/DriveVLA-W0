#!/usr/bin/env bash
set -euo pipefail

# One-click inference runner for Qwen-VL VLA
# Edit the parameter values below as needed.

# GPUs to use
GPUS=8

# Key paths (modify these directly as needed)
QWEN_HUB=/mnt/nvme1n1p1/yingyan.li/logs/qwen25vl_fast_147456_4_000_2va_nuplan_pretrained_8k_2va
OUTPUT_DIR=/mnt/nvme1n1p1/yingyan.li/logs/qwen25vl_fast_147456_4_000_2va_nuplan_pretrained_8k_2va/json_output_debug
TEST_PKL=data/navsim/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s.pkl
ACTION_TOKENIZER=configs/fast
TOKEN_YAML=data/navsim/processed_data/scene_files/scene_filter/navtest.yaml
NORM_STATS=configs/normalizer_navsim_trainval/norm_stats.json

# Inference knobs (modify these directly as needed)
CUR_IDX=3
FUTURE_NUMS=8
ACTION_DIM=3
MODEL_MAX_LEN=1400
USE_PREV=1
BF16=1
SAVE_GT=1
# Optional path remap for absolute paths inside dataset, our pkl has abs path, so no need to replace. if pkl has correct rel path, no need to replace.
PATH_REPLACE_FROM=''
PATH_REPLACE_TO=''
# Optional camera subdir to insert before image basename
IMAGE_CAM_SUBDIR=CAM_F0
RAW_IMG_ROOT='data/navsim/sensor_blobs/test'

mkdir -p "${OUTPUT_DIR}"

CMD=(
  torchrun --nproc_per_node="${GPUS}" inference/qwen/inference_action_navsim_vava.py \
    --qwen_hub "${QWEN_HUB}" \
    --output_dir "${OUTPUT_DIR}" \
    --test_data_pkl "${TEST_PKL}" \
    --action_tokenizer_path "${ACTION_TOKENIZER}" \
    --token_yaml "${TOKEN_YAML}" \
    --norm_stats_path "${NORM_STATS}" \
    --cur_frame_idx "${CUR_IDX}" \
    --future_nums "${FUTURE_NUMS}" \
    --action_dim "${ACTION_DIM}" \
    --model_max_length "${MODEL_MAX_LEN}" \
    --path_replace_from "${PATH_REPLACE_FROM}" \
    --path_replace_to "${PATH_REPLACE_TO}" \
    --image_cam_subdir "${IMAGE_CAM_SUBDIR}" \
    --raw_img_root "${RAW_IMG_ROOT}"
)

if [[ "${USE_PREV}" == "1" ]]; then
  CMD+=(--use_previous_actions)
fi
if [[ "${BF16}" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "${SAVE_GT}" == "1" ]]; then
  CMD+=(--save_gt)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
