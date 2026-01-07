# VLA 推理脚本使用指南

本目录包含用于运行 VLA (Vision-Language-Action) 模型推理的脚本。所有硬编码路径已移除，支持通过环境变量或配置文件进行配置。

## 目录结构

```
inference/vla/
├── config.py                    # 配置管理模块
├── config.yaml.example          # 配置文件模板
├── inference_action_navsim_qformer_vava.py  # QFormer 模型推理脚本
├── inference_action_navsim_pi0_vava.py     # Pi0 模型推理脚本
├── inference_action_navsim_ar_vava.py      # AutoRegressive 模型推理脚本
├── infer_navsim_qformer.sh      # QFormer 推理启动脚本
├── run_emu_vla_navsim_metric_others.sh     # 指标评估脚本
└── README.md                    # 本文件
```

## 快速开始

### 方法 1：使用环境变量（推荐）

设置必需的环境变量：

```bash
# 项目根目录（可选，会自动检测）
export DRIVEVLA_ROOT="/path/to/DriveVLA-W0"

# 必需配置
export VLA_ACTION_TOKENIZER="/path/to/pretrained_models/fast"
export VLA_VLM_MODEL="/path/to/logs/train_nuplan_6va_v0.2_multi_node"
export VLA_NORM_STATS="/path/to/configs/normalizer_navsim_trainval/norm_stats.json"

# 可选配置
export VLA_TOKEN_YAML="data/navsim/processed_data/scene_files/scene_filter/navtest.yaml"
export VLA_NUM_WORKERS=12
export VLA_BATCH_SIZE=1
```

然后运行推理脚本：

```bash
# QFormer 模型
torchrun --nproc_per_node=8 inference/vla/inference_action_navsim_qformer_vava.py \
    --emu_hub "/path/to/trained/qformer/model" \
    --output_dir "/path/to/output" \
    --test_data_pkl "/path/to/test_data.pkl"

# Pi0 模型
torchrun --nproc_per_node=8 inference/vla/inference_action_navsim_pi0_vava.py \
    --emu_hub "/path/to/trained/pi0/model" \
    --output_dir "/path/to/output" \
    --test_data_pkl "/path/to/test_data.pkl"

# AutoRegressive 模型
torchrun --nproc_per_node=8 inference/vla/inference_action_navsim_ar_vava.py \
    --emu_hub "/path/to/trained/ar/model" \
    --output_dir "/path/to/output" \
    --test_data_pkl "/path/to/test_data.pkl"
```

### 方法 2：使用配置文件

1. 复制配置模板：

```bash
cp config.yaml.example config.yaml
```

2. 编辑 `config.yaml`，填写实际路径：

```yaml
paths:
  action_tokenizer: /path/to/pretrained_models/fast
  vlm_model: /path/to/logs/train_nuplan_6va_v0.2_multi_node
  norm_stats: /path/to/configs/normalizer_navsim_trainval/norm_stats.json
  token_yaml: data/navsim/processed_data/scene_files/scene_filter/navtest.yaml
```

3. 运行脚本时指定配置文件：

```bash
torchrun --nproc_per_node=8 inference/vla/inference_action_navsim_qformer_vava.py \
    --config config.yaml \
    --emu_hub "/path/to/trained/model" \
    --output_dir "/path/to/output" \
    --test_data_pkl "/path/to/test_data.pkl"
```

### 方法 3：使用 Shell 脚本

使用提供的 Shell 脚本可以更方便地运行：

```bash
# 设置环境变量
export EMU_HUB="/path/to/trained/model"
export OUTPUT_DIR="/path/to/output"
export TEST_DATA_PKL="/path/to/test_data.pkl"

# 运行脚本
bash infer_navsim_qformer.sh
```

## 配置说明

### 环境变量列表

| 环境变量 | 说明 | 必需 | 默认值 |
|---------|------|------|--------|
| `DRIVEVLA_ROOT` | 项目根目录 | 否 | 自动检测 |
| `VLA_CONFIG` | 配置文件路径 | 否 | `inference/vla/config.yaml` |
| `VLA_ACTION_TOKENIZER` | Action tokenizer 路径 | 是 | - |
| `VLA_VLM_MODEL` | VLM 模型路径（用于 tokenizer） | 是 | - |
| `VLA_NORM_STATS` | 归一化统计文件路径 | 是 | - |
| `VLA_TOKEN_YAML` | Token YAML 文件路径 | 否 | `data/navsim/.../navtest.yaml` |
| `VLA_BATCH_SIZE` | 批次大小 | 否 | 1 |
| `VLA_NUM_WORKERS` | DataLoader worker 数量 | 否 | 12 |
| `VLA_MODEL_MAX_LENGTH` | Tokenizer 最大长度 | 否 | 1400 |
| `VLA_NUM_INFERENCE_STEPS` | Pi0 推理步数 | 否 | 10 |

### 配置文件结构

配置文件使用 YAML 格式，参考 `config.yaml.example`。配置优先级：

1. 命令行参数（最高优先级）
2. 环境变量
3. 配置文件
4. 默认值（最低优先级）

## 命令行参数

所有推理脚本支持以下参数：

### 通用参数

- `--emu_hub` (必需): 训练好的模型路径
- `--output_dir` (必需): 输出目录
- `--test_data_pkl` (必需): 测试数据 pickle 文件路径
- `--token_yaml` (可选): Token YAML 文件路径
- `--num_workers` (可选): DataLoader worker 数量
- `--config` (可选): 配置文件路径

### Pi0 特定参数

- `--num_inference_steps` (可选): 推理步数，默认从配置文件读取

## 项目根目录自动检测

配置模块会自动检测项目根目录，检测逻辑：

1. 从当前脚本位置向上查找，直到找到包含 `train/` 和 `reference/` 目录的目录
2. 如果找不到，使用 `DRIVEVLA_ROOT` 环境变量
3. 最后回退到脚本位置的父目录的父目录

## 常见问题

### Q: 如何知道需要设置哪些环境变量？

A: 运行脚本时，如果缺少必需配置，会显示明确的错误信息，提示需要设置哪些环境变量或配置文件项。

### Q: 配置文件和环境变量哪个优先级更高？

A: 环境变量优先级更高。如果同时设置了环境变量和配置文件，环境变量的值会被使用。

### Q: 如何在不同环境中使用不同的配置？

A: 可以创建多个配置文件（如 `config.dev.yaml`, `config.prod.yaml`），然后通过 `--config` 参数或 `VLA_CONFIG` 环境变量指定。

### Q: 项目根目录检测失败怎么办？

A: 手动设置 `DRIVEVLA_ROOT` 环境变量指向项目根目录。

## 示例

### 完整示例：使用环境变量

```bash
# 设置所有必需的环境变量
export DRIVEVLA_ROOT="/home/user/DriveVLA-W0"
export VLA_ACTION_TOKENIZER="/data/models/fast"
export VLA_VLM_MODEL="/data/models/train_nuplan_6va_v0.2_multi_node"
export VLA_NORM_STATS="/data/configs/normalizer_navsim_trainval/norm_stats.json"

# 运行推理
torchrun --nproc_per_node=8 inference/vla/inference_action_navsim_qformer_vava.py \
    --emu_hub "/data/models/train_navsim_qformer_anchor_vava" \
    --output_dir "/data/output/json_output" \
    --test_data_pkl "/data/navsim/test_data.pkl"
```

### 完整示例：使用配置文件

```bash
# 创建配置文件
cat > config.yaml << EOF
paths:
  action_tokenizer: /data/models/fast
  vlm_model: /data/models/train_nuplan_6va_v0.2_multi_node
  norm_stats: /data/configs/normalizer_navsim_trainval/norm_stats.json
data:
  num_workers: 16
  batch_size: 1
EOF

# 运行推理
torchrun --nproc_per_node=8 inference/vla/inference_action_navsim_qformer_vava.py \
    --config config.yaml \
    --emu_hub "/data/models/train_navsim_qformer_anchor_vava" \
    --output_dir "/data/output/json_output" \
    --test_data_pkl "/data/navsim/test_data.pkl"
```

## 贡献

如果发现问题或有改进建议，请提交 Issue 或 Pull Request。

