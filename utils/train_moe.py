import os
import os.path as osp
import torch
import gc
from dataclasses import dataclass, field
from typing import Optional, List
import pathlib
import transformers as tf
from datasets import Emu3SFTDataset
import sys
import json
import torch.distributed as dist
from datetime import datetime
import threading
from queue import Queue
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录(即包含train和reference的目录)
parent_dir = os.path.dirname(current_dir)
# 添加reference/Emu3路径到sys.path
sys.path.append(os.path.join(parent_dir, "reference", "Emu3"))
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM, Emu3MoE, Emu3MoEConfig
from transformers import AutoModel,Trainer
from datasets import Emu3DrivingDataset
from datasets import Emu3DrivingVAVADataset
from datasets import Emu3DrivingNuplan6VADataset
from torch.utils.data import WeightedRandomSampler, DataLoader

class MemoryEfficientTrainer(tf.Trainer):
    """最简单的显存回收Trainer"""
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        # 评估前清理显存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 执行评估
        result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        # 评估后清理显存
        torch.cuda.empty_cache()
        gc.collect()
        
        return result

class LoggingTrainer(tf.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_queue = None
        # Only the main process will handle file I/O and the logging thread.
        if self.state.is_world_process_zero:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"sample_log_consolidated_{timestamp}.json"
            self.log_filepath = osp.join(self.args.output_dir, log_filename)
            
            self.log_queue = Queue()
            self.logging_thread = threading.Thread(target=self._log_writer, daemon=True)
            self.logging_thread.start()

    def _log_writer(self):
        log_data = {}
        with open(self.log_filepath, 'w') as f:
            while True:
                # Blocks until an item is available
                data = self.log_queue.get()
                
                # A sentinel value (None) indicates the end of training
                if data is None:
                    break
                
                step, indices = data
                log_data[step] = indices
                
                # Overwrite the file with the updated data at each step
                f.seek(0)
                json.dump(log_data, f, indent=4)
                f.truncate()
                f.flush() # Ensure data is written to disk

    def training_step(self, model: torch.nn.Module, inputs: dict) -> torch.Tensor:
        # Pop the index first, since it's not a model input.
        # Handle the case where 'index' might not be present in inputs
        indices = inputs.pop("index", None)
        
        loss = super().training_step(model, inputs)

        # Only proceed with logging if we have indices and are in training mode
        if indices is not None and self.is_in_train:
            if dist.is_initialized():
                # 🔥 CRITICAL: ALL processes must participate in this collective operation
                gathered_indices_list = [None] * dist.get_world_size()
                dist.all_gather_object(gathered_indices_list, indices.cpu().tolist())
                
                # Only the main process will log the consolidated data
                if self.state.is_world_process_zero:
                    # Flatten the list of lists into a single list
                    consolidated_indices = [item for sublist in gathered_indices_list for item in sublist]
                    # Put the data into the queue for the logging thread to process.
                    self.log_queue.put((self.state.global_step, consolidated_indices))
            else:
                # Non-distributed case: only main process logs
                if self.state.is_world_process_zero:
                    consolidated_indices = indices.cpu().tolist()
                    self.log_queue.put((self.state.global_step, consolidated_indices))

        return loss

    def __del__(self):
        # Gracefully shut down the logging thread
        if self.state.is_world_process_zero and hasattr(self, 'log_queue') and self.log_queue is not None:
            # Signal the logging thread to terminate
            self.log_queue.put(None)
            # Wait for the logging thread to finish its work
            self.logging_thread.join()


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
    use_flip: bool = field(default=False)
    cur_frame_idx: int = field(default=3)
    data_type: str = field(default="navsim") 
    vq_root: str = field(default="/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/data/nuplan/processed_data/vq_codes")
    pre_action_frames: int = field(default=3)
    resolution: str = field(default="36,64")
    action_hz: float = field(default=2)
    va_pair_num: int = field(default=6)
    
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
    save_on_each_node: bool = field(default=False)  # 只在主节点保存
    save_only_model: bool = field(default=False)

def load_model(model_args, model_config, training_args):
    """
    Load model based on whether to train from scratch or fine-tune from a pre-trained model.
    """
    if training_args.from_scratch:
        model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else None
        model_config.attn_implementation = "flash_attention_2" if training_args.attn_type == "fa2" else None
        return Emu3MoE(config=model_config)
    else:
        return Emu3MoE.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
            torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        )

def get_dataset(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        return Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
    elif data_args.real_robot:
        return Emu3RealRobotDataset(data_args, tokenizer=tokenizer)
    elif data_args.driving:
        return Emu3DrivingDataset(data_args, tokenizer=tokenizer)
    return Emu3SFTDataset(data_args, tokenizer=tokenizer)

def get_dataset_split(data_args, tokenizer):
    if data_args.data_type == "navsim_vava":
        full_dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    elif data_args.data_type == "nuplan_6va":
        full_dataset = Emu3DrivingNuplan6VADataset(data_args, tokenizer=tokenizer)
    else:
        full_dataset = Emu3DrivingDataset(data_args, tokenizer=tokenizer)
        
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
    
    # Handle resolution parameter conversion from string to tuple
    if isinstance(data_args.resolution, str):
        print(f"Converting resolution from string '{data_args.resolution}' to tuple")
        data_args.resolution = tuple(map(int, data_args.resolution.split(',')))
        print(f"Resolution after conversion: {data_args.resolution}")

    # Set environment variable for WANDB logging
    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")

    # Load model configuration and tokenizer
    model_config = Emu3MoEConfig.from_pretrained(model_args.model_config_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    # Initialize model
    model = load_model(model_args, model_config, training_args)

    # Initialize dataset
    train_dataset, eval_dataset = get_dataset_split(data_args, tokenizer)
    # train_dataset = get_dataset(data_args, tokenizer)

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
