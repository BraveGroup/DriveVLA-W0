import warnings
warnings.filterwarnings("ignore")

import os
import os.path as osp
import torch
from dataclasses import dataclass, field
from typing import Optional, List
import pathlib
import transformers as tf
from datasets import Emu3SFTDataset
import sys
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录(即包含train和reference的目录)
parent_dir = os.path.dirname(current_dir)
# 添加reference/Emu3路径到sys.path
sys.path.append(os.path.join(parent_dir, "reference", "Emu3"))
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM, Emu3MoE, Emu3MoEConfig, Emu3Pi0, Emu3Pi0Config, Emu3QFormer, Emu3AutoRegressive
from transformers import AutoModel,Trainer
from datasets import Emu3DrivingDataset
from datasets import Emu3DrivingVAVADataset, Emu3DrivingVAVA_AR_Dataset
from torch.utils.data import WeightedRandomSampler, DataLoader

class WeightedSamplerTrainer(Trainer):
    def get_train_dataloader(self):
        # 从 train_dataset 中获取 sample_weights
        sample_weights = torch.tensor(
            self.train_dataset.sample_weights, dtype=torch.double
        )
        # 用 sample_weights 构建 WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    model_config_path: Optional[str] = field(default="pretrain/Emu3-Base")

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    apply_loss_on_only_action: bool = field(default=False) 
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)
    frames: int = field(default=4)
    VL: bool = field(default=False)
    actions: bool = field(default=False)
    actions_format: str = field(default="openvla")
    action_frames: int = field(default=8)
    use_gripper: bool = field(default=False)
    action_tokenizer_path: Optional[str] = field(default=None)
    video_format: str = field(default=None)
    random_frame_sampling: bool = field(default=True)
    raw_image: bool = field(default=False)
    post_training: bool = field(default=False)
    datasets_weight: bool = field(default=False)
    without_text: bool = field(default=False)
    real_robot: bool = field(default=False)
    driving: bool = field(default=False)
    use_previous_actions: bool = field(default=False)
    use_only_lidar: bool = field(default=False)
    use_lidar_and_image: bool = field(default=False)
    use_flip: bool = field(default=False)
    cur_frame_idx: int = field(default=3)
    action_dim: int = field(default=3)  # Action dimension for Pi0 model
    pre_action_frames: int = field(default=3)

@dataclass
class TrainingArguments(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    from_scratch: bool = field(default=False)
    dataloader_num_workers: Optional[int] = field(default=0)
    evaluation_strategy: str = field(default="steps")  # or "epoch"
    eval_steps: Optional[int] = field(default=1000)     # 每 1000 step 验证一次
    per_device_eval_batch_size: Optional[int] = field(default=1)
    eval_accumulation_steps: Optional[int] = field(default=1)
    # Pi0 specific training arguments
    train_action_only: bool = field(default=False)
    action_loss_weight: float = field(default=10.0)
    freeze_vlm: bool = field(default=False)  # 新增：是否冻结VLM参数

def load_model(model_args, model_config, training_args):
    # 初始化 AutoRegressive 模型
    model = Emu3AutoRegressive(config=model_config, pretrain_vlm_path = model_args.model_name_or_path)

    # 冻结VLM参数（如果指定）
    if training_args.freeze_vlm:
        print("Freezing VLM parameters...")
        model.freeze_vlm()

    return model


def get_dataset(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.driving:
        # 使用专为 AutoRegressive 设计的数据集：输出 vlm_input_ids + action_input_ids
        return Emu3DrivingVAVA_AR_Dataset(data_args, tokenizer=tokenizer)
    return Emu3SFTDataset(data_args, tokenizer=tokenizer)

def get_dataset_split(data_args, tokenizer):
    if data_args.post_training:
        full_dataset = Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
    elif data_args.driving:
        full_dataset = Emu3DrivingVAVA_AR_Dataset(data_args, tokenizer=tokenizer)
    else:
        full_dataset = Emu3SFTDataset(data_args, tokenizer=tokenizer)
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]


def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)

def train():
    """
    Main function to train the model.
    """
    # Parse arguments
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set environment variable for WANDB logging
    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")

    # Load Model configuration
    model_config = Emu3Pi0Config.from_pretrained(model_args.model_config_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings", "action_loss_weight", "freeze_vlm"])
    if training_args.bf16:
        model_config.torch_dtype = torch.bfloat16
        model_config.vlm_config.torch_dtype = torch.bfloat16
        model_config.action_config.torch_dtype = torch.bfloat16
    
    # Initialize model
    model = load_model(model_args, model_config, training_args)

    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate
    
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    # Initialize dataset
    train_dataset, eval_dataset = get_dataset_split(data_args, tokenizer)

    if data_args.datasets_weight:
        trainer = WeightedSamplerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset, 
            tokenizer=tokenizer,
        )
    else:
        # Setup Trainer
        trainer = tf.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # ✅ 加上这个
            tokenizer=tokenizer,
        )


    # Check if resuming from checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model and training state
    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
