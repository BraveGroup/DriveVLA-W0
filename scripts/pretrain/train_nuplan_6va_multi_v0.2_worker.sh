#!/usr/bin/env bash

# Worker node training script (RANK=1) - nuplan v0.1
# Usage: ./worker_node.sh <master_ip> [port]

echo "NavSim Multi-Node Training - Worker Node v0.1"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <master_ip> [port]"
    echo "Example: $0 192.168.1.100 23456"
    exit 1
fi

WORLD_SIZE=3
RANK=1
MASTER_ADDR=$1
MASTER_PORT=${2:-23456}
NGPUS=8

# 硬编码适配本地仓库路径
REPO_ROOT="/mnt/vdb1/yingyan.li/repo/VLA_Emu"
DATAPATH="${REPO_ROOT}/data/nuplan/processed_data/meta/nuplan_processed_data.pkl"
VQ_ROOT="${REPO_ROOT}/data/nuplan/processed_data/vq_codes_low_res_corrected_merge"
ACTION_TOKENIZER_PATH="${REPO_ROOT}/pretrained_models/fast"
NORMALIZER_PATH="${REPO_ROOT}/configs/normalizer_nuplan"
MODEL_PATH="${REPO_ROOT}/pretrained_models/Emu3-Stage1"
CONFIG_PATH="${REPO_ROOT}/configs/moe_fast_video.json"
DEEPSPEED_CONFIG="${REPO_ROOT}/scripts/sft/zero3_offload.json"
EXP_NAME=train_nuplan_6va_v0.2_multi_node

echo "Master IP: $MASTER_ADDR"
echo "Worker IP: $(hostname -I | awk '{print $1}')"
echo "World size: $WORLD_SIZE, Rank: $RANK"
echo "Port: $MASTER_PORT, GPUs: $NGPUS"
echo "Repository root: $REPO_ROOT"

# Check connectivity to master
echo "Testing connection to master $MASTER_ADDR..."
if ping -c 3 $MASTER_ADDR > /dev/null 2>&1; then
    echo "Master connection: OK"
else
    echo "Master connection: FAILED"
    exit 1
fi

# Check data paths
if [ -d "$DATAPATH" ]; then
    echo "Data directory: OK"
else
    echo "Data directory: NOT FOUND - $DATAPATH"
fi

if [ -d "$ACTION_TOKENIZER_PATH" ]; then
    echo "Action tokenizer: OK"
else
    echo "Action tokenizer: NOT FOUND - $ACTION_TOKENIZER_PATH"
fi

if [ -d "$NORMALIZER_PATH" ]; then
    echo "Normalizer config: OK"
else
    echo "Normalizer config: NOT FOUND - $NORMALIZER_PATH"
fi

if [ -d "$VQ_ROOT" ]; then
    echo "VQ codes directory: OK"
else
    echo "VQ codes directory: NOT FOUND - $VQ_ROOT"
fi

if [ -d "$MODEL_PATH" ]; then
    echo "Model path: OK"
else
    echo "Model path: NOT FOUND - $MODEL_PATH"
fi

if [ -f "$CONFIG_PATH" ]; then
    echo "Config file: OK"
else
    echo "Config file: NOT FOUND - $CONFIG_PATH"
fi

# 优化的NCCL配置，适配以太网多节点训练
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO

# 以太网优化配置
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
export NCCL_P2P_DISABLE=1

# 缓冲区和线程优化
export NCCL_BUFFSIZE=33554432  # 32MB，增加缓冲区大小
export NCCL_NTHREADS=16
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_NCHANNELS=16

# 禁用低延迟模式（不适合以太网）
export NCCL_LL_THRESHOLD=0
export NCCL_LL128_THRESHOLD=0

# 超时配置
export NCCL_TIMEOUT=3600        # 增加超时时间
export NCCL_OP_TIMEOUT=3600000  # 增加操作超时
export NCCL_IB_TIMEOUT=15
export NCCL_IB_RETRY_CNT=5

# 错误处理和稳定性配置
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_NET_SHARED_BUFFERS=0

# CUDA优化
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置Python路径
export PYTHONPATH=$REPO_ROOT
export PYTHONPATH="${REPO_ROOT}/reference/Emu3:$PYTHONPATH"

echo "Environment check:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
echo "PyTorch: $(python -c 'import torch;print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch;print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch;print(torch.cuda.device_count())')"

echo "Starting worker node training v0.1..."
echo "Waiting for master to initialize..."

# 切换到仓库根目录
cd $REPO_ROOT

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train/train_moe.py \
    --model_name_or_path ${MODEL_PATH} \
    --model_config_path ${CONFIG_PATH} \
    --actions_format fast \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --output_dir logs/${EXP_NAME} \
    --learning_rate 2e-4 \
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
    --data_type nuplan_6va \
    --vq_root ${VQ_ROOT} \
    --max_steps 8000 \
    --dataloader_num_workers 16 \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_steps 400 \
    --per_device_train_batch_size 4 \
    --frames 1 \
    --action_frames 8 \
    --max_position_embeddings 4000 \
    --seed 42 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 4000 \
    --save_only_model False \
    --eval_strategy no \
    --apply_loss_on_only_vision True \
    --apply_loss_on_only_action False \
    --actions True \
    --use_gripper False \
    --driving True \
    --evaluation_strategy steps \
    --eval_steps 4050 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --use_previous_actions True \
    --resolution 18,32 \
    --action_hz 2 \
    --pre_action_frames 3 \
    --va_pair_num 6 \

echo "Worker training v0.1 completed" 