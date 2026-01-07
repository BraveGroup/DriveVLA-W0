# DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving
> 📜 [[Arxiv](http://arxiv.org/abs/2510.12796)] 🤗 [[Model Weights](https://huggingface.co/liyingyan/DriveVLA-W0)]

Yingyan Li*, Shuyao Shang*, Weisong Liu*, Bing Zhan*, Haochen Wang*, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan†, Zhaoxiang Zhang†

This Paper presents **DriveVLA-W0**, a training paradigm that employs world modeling to predict future images. This task generates a dense, self-supervised signal that compels the model to learn the underlying dynamics of the driving environment, remedying the "supervision deficit" in VLA models and amplifying data scaling laws.

<p align="center">
  <img src="assets/fig1.png" alt="DriveVLA-W0" width="1000"/>
</p>


> Due to company policy, only the reviewed portion of our code is currently available. Please contact us if you have any questions.

## 📋 项目结构

```
DriveVLA-W0/
├── assets/                    # 项目资源文件（图片、文档等）
├── configs/                   # 模型配置文件和归一化统计
│   ├── fast/                 # fast action tokenizer
│   ├── normalizer_navsim_test/    # NAVSIM测试数据归一化配置
│   ├── normalizer_navsim_trainval/ # NAVSIM训练验证数据归一化配置
│   └── normalizer_nuplan/    # NuPlan数据集归一化配置
├── data/                      # 数据处理和配置
│   ├── navsim/               # NAVSIM数据集相关
│   └── others/               # 其他数据集
├── inference/                 # 推理脚本
│   ├── navsim/               # NAVSIM PDMS评测
│   ├── qwen/                 # Qwen模型推理
│   └── vla/                  # Emu模型推理
├── models/                    # 模型定义
│   ├── policy_head/          # 策略头实现
│   └── tokenizer/            # 分词器实现
├── scripts/                   # 训练和部署脚本
├── tools/                     # 工具脚本
│   ├── action_tokenizer/     # 动作分词器
│   └── pickle_gen/           # 数据预处理和pickle生成
├── train/                     # 训练代码
│   ├── datasets.py           # 数据集定义
│   ├── train_ar.py           # 自回归模型训练
│   ├── train_moe.py          # MoE模型训练
│   ├── train_pi0.py          # PI0模型训练
│   ├── train_qformer.py      # QFormer模型训练
│   ├── train_qwen_vla.py     # Qwen-VLA联合训练
│   └── train.py              # 主训练脚本
└── requirements.txt          # Python依赖
```

## 🚀 快速开始

### 5分钟上手示例

1. **下载预训练模型**
```bash
pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
mkdir pretrained_models
bash scripts/misc/download.sh
```

2. **环境设置**
```bash
conda create -n drivevla python=3.10
conda activate drivevla
pip install -r requirements.txt
```

3. **下载模型权重**
```bash
# 从Hugging Face下载预训练权重
# 权重将保存在 pretrained_models/ 目录下
```

4. **运行推理**
```bash
# 使用预训练模型进行推理
bash inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh
```

### 完整训练流程

如果您想从头训练模型，请参考 [Training.md](Training.md) 获取详细的训练指南。

## 📊 数据准备

### NAVSIM数据集

DriveVLA-W0 使用 NAVSIM (v1.1) 数据集进行训练和评估。您需要：

1. **获取NAVSIM数据集**
   - 访问 [NAVSIM官方仓库](https://github.com/autonomousvision/navsim/tree/v1.1)
   - 下载训练和测试数据
   - 数据包含传感器数据、场景信息和标注

2. **数据预处理**
   ```bash
   # 生成VQ索引
   python tools/pickle_gen/pickle_generation_navsim_pre_1s.py

   # 生成NAVSIM pickle文件
   bash scripts/tokenizer/extract_vq_emu3_navsim.sh
   ```

3. **数据格式**
   - 预处理后的数据保存在 `data/navsim/processed_data/`
   - 包含场景文件、元数据和预处理后的特征

### 数据量级
- **训练数据**: ~100K帧驾驶场景
- **验证数据**: ~10K帧
- **测试数据**: NAVSIM测试集

## 💻 硬件要求

### 训练资源消耗
8x L20 GPU (40GB memory), ~16小时


# Install



## CUDA install

如果您的系统没有CUDA 12.4+，请先安装：

```bash
# 下载CUDA 12.8.1 (推荐版本)
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run

# 安装CUDA工具包
bash cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.8

# 设置环境变量 (添加到 ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Conda 环境设置

```bash
# 创建Conda环境
conda create -n drivevla python=3.10
conda activate drivevla

# 安装PyTorch (CUDA 12.4)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 安装核心依赖
pip install -r requirements.txt
pip install "transformers[torch]"

# 安装训练相关依赖
pip install deepspeed          # 分布式训练
pip install scipy              # 科学计算
pip install tensorboard==2.14.0  # 可视化
pip install wandb              # 实验跟踪
```


### Testing

First, please download the checkpoints from [Hugging Face](https://huggingface.co/liyingyan/DriveVLA-W0). 

Then, run the corresponding testing script to get output actions as json files
```
bash inference/vla/infer_navsim_with_previous_action_last_vava.sh
```
Finally, run the following script to compute PDMS from json files (using the conda enviroment with [navsim](https://github.com/autonomousvision/navsim/tree/v1.1))
```
bash inference/vla/run_emu_vla_navsim_metric_others.sh
```

## ⚙️ 配置说明

### 配置文件结构

项目使用JSON格式的配置文件，位于 `configs/` 目录：

```
configs/
├── moe_fast_video.json          # MoE模型快速推理配置
├── moe_fast_video_pretrain.json # MoE模型预训练配置
├── normalizer_navsim_test/      # NAVSIM测试数据归一化参数
├── normalizer_navsim_trainval/  # NAVSIM训练数据归一化参数
└── normalizer_nuplan/           # NuPlan数据归一化参数
```

### 归一化统计

数据归一化参数根据训练数据集自动计算：

- `normalizer_navsim_trainval/` - 基于NAVSIM训练集
- `normalizer_navsim_test/` - 基于NAVSIM测试集
- `normalizer_nuplan/` - 基于NuPlan数据集

# 🏆 NAVSIM v1/v2 Benchmark SOTA

Here is a comparison with state-of-the-art methods on the NAVSIM test set, as presented in the paper. Our model, **DriveVLA-W0**, establishes a new state-of-the-art.

| Method | Reference | Sensors | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Human** | | | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |
| **_BEV-based Methods_** | | | | | | | | |
| LAW | ICLR'25 | 1x Cam | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| Hydra-MDP | arXiv'24 | 3x Cam + L | 98.3 | 96.0 | 94.6 | 100.0 | 78.7 | 86.5 |
| DiffusionDrive | CVPR'25 | 3x Cam + L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| WoTE | ICCV'25 | 3x Cam + L | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 |
| **_VLA-based Methods_** | | | | | | | | |
| AutoVLA | NeurIPS'25 | 3x Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| ReCogDrive | arXiv'25 | 3x Cam | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| **DriveVLA-W0*** | **Ours** | **1x Cam** | **98.7** | **99.1** | **95.3** | **99.3** | **83.3** | **90.2** |
| AutoVLA† | NeurIPS'25 | 3x Cam | 99.1 | 97.1 | 97.1 | 100.0 | 87.6 | 92.1 |
| **DriveVLA-W0†** | **Ours** | **1x Cam** | **99.3** | **97.4** | **97.0** | **99.9** | **88.3** | **93.0** |

# ⭐ Star 
If you find our work useful for your research, please consider giving this repository a star ⭐.

# 📜 Citation
If you find this work useful for your research, please consider citing our paper:
```
@article{li2025drivevla,
  title={DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving},
  author={Li, Yingyan and Shang, Shuyao and Liu, Weisong and Zhan, Bing and Wang, Haochen and Wang, Yuqi and Chen, Yuntao and Wang, Xiaoman and An, Yasong and Tang, Chufeng and others},
  journal={arXiv preprint arXiv:2510.12796},
  year={2025}
}
```

# Acknowledgements
We would like to acknowledge the following related works:

[**LAW (ICLR 2025)**](https://github.com/BraveGroup/LAW): Using latent world models for self-supervised feature learning in end-to-end autonomous driving.

[**WoTE (ICCV 2025)**](https://github.com/liyingyanUCAS/WoTE): Using BEV world models for online trajectory evaluation in end-to-end autonomous driving.

[**UniVLA**](https://github.com/baaivision/UniVLA): World modeling in the broader field of robotics.
