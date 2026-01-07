#!/usr/bin/env python3
"""
Unified training script for Qwen-VL VLA model
Combines Qwen-VL's multimodal capabilities with VLA action prediction
"""
import os
import sys
import logging
import pathlib
import pickle
from pathlib import Path
import torch
import transformers
from typing import Dict
from collections import defaultdict

# Add paths to import Qwen-VL components
qwen_vl_path = Path(__file__).parent.parent / "reference" / "Qwen2.5-VL" / "qwen-vl-finetune"
sys.path.append(str(qwen_vl_path))

# Ensure trainer monkey patches (optimizer grouping, print helpers) are applied
import qwenvl.train.trainer  # noqa: F401

# Optional utilities from trainer
from qwenvl.train.trainer import replace_qwen2_vl_attention_class  # noqa: F401

# Import Qwen-VL components
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoTokenizer,
    AutoProcessor,

)

from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl_ross import (
    Qwen2_5_VLConfigROSS, 
    Qwen2_5_VLForConditionalGenerationROSS,
)

from reference.transformers.src.transformers.trainer import Trainer



from qwenvl.dataset.data_qwen_vla import (
    make_supervised_data_module_navsim2_vla_ross,
    make_supervised_data_module_nuplan2_vla_ross
)
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

# Import VLA-specific components
sys.path.append(str(qwen_vl_path / "qwenvl" / "utils"))
from token_utils import smart_load_model_and_tokenizer, prepare_action_tokenizer_mapping, check_and_add_vla_tokens

local_rank = None


class RossTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acc_sum = defaultdict(float)
        self._acc_n = 0
        self._micro_in_accum = 0  # 当前累积里的 micro-step 计数

    def _log_mean_and_reset(self):
        if self._acc_n == 0:
            return
        mean_logs = {k: (s / float(self._acc_n)) for k, s in self._acc_sum.items()}

        # 仅主进程写
        acc = getattr(self, "accelerator", None)
        is_main = acc.is_main_process if acc is not None else self.is_world_process_zero()
        if is_main:
            self.log(mean_logs)

        self._acc_sum.clear()
        self._acc_n = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        # 累积 forward 写入的 _last_logs
        logs = getattr(model, "_last_logs", None)
        if logs:
            for k, v in logs.items():
                try:
                    self._acc_sum[k] += float(v)
                except Exception:
                    pass
            self._acc_n += 1

        # 统计当前累积内的 micro-step，并在到达 optimizer step 时输出
        self._micro_in_accum += 1
        gas = max(1, int(self.args.gradient_accumulation_steps))
        if self._micro_in_accum % gas == 0:
            self._log_mean_and_reset()

        return loss
    
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def setup_vla_model_and_tokenizer(model_args, training_args):
    """
    Complete VLA model setup: load model, configure training parameters, and set up tokenizer
    """
    
    model_type = "qwen2.5vl_ross"

    model_class = Qwen2_5_VLForConditionalGenerationROSS

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    base_cfg = Qwen2_5_VLConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
    )
    if "enable_ross" in base_cfg.to_dict().keys():
        ross_cfg = Qwen2_5_VLConfigROSS(
                **base_cfg.to_dict(),
        )
    else:
        ross_cfg = Qwen2_5_VLConfigROSS(
                **base_cfg.to_dict(),
                enable_ross=True,
                extract_image_hidden=True,
                extract_action_hidden=True,
                sd_model_path=model_args.sd_model_path,
            )
    model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=ross_cfg,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            trust_remote_code=True,
    )
    from diffusers import AutoencoderKL
    from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_ross.unet_2d_condition import UNet2DConditionModel   
    model.vae = AutoencoderKL.from_pretrained(model_args.sd_model_path.replace("/unet", "/vae"), torch_dtype=model.dtype)
    model.vae.eval()
    model.vae.requires_grad_(False)
    model.denoiser.unet = UNet2DConditionModel.from_pretrained(model_args.sd_model_path, torch_dtype=model.dtype)
    model.denoiser.unet.train()
    model.denoiser.unet.requires_grad_(True)

    tokenizer, model, _ = check_and_add_vla_tokens(tokenizer, model)
    
    rank0_print(f"Loaded {model_class.__name__} from {model_args.model_name_or_path}")

    image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor

    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"
    
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Configure which parts of the model to train
    # Vision encoder
    if getattr(model_args, 'tune_mm_vision', True):
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
        rank0_print("Vision encoder: TRAINABLE")
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False
        rank0_print("Vision encoder: FROZEN")
    
    # Vision-language connector (merger)
    if getattr(model_args, 'tune_mm_mlp', True):
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
        rank0_print("Vision-language merger: TRAINABLE")
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False
        rank0_print("Vision-language merger: FROZEN")
    
    # Language model
    if getattr(model_args, 'tune_mm_llm', True):
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
        rank0_print("Language model: TRAINABLE")
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
        rank0_print("Language model: FROZEN")
    
    # Denoiser
    for n, p in model.denoiser.named_parameters():
        p.requires_grad = True
    rank0_print("Denoiser: TRAINABLE")

    # VAE
    for n, p in model.vae.named_parameters():
        p.requires_grad = False
    rank0_print("VAE: FROZEN")
    
    def count(m):
        tot = sum(p.numel() for p in m.parameters())
        trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return tot, trn

    tot, trn = count(model)
    if local_rank in (0, None):
        print(f"[PARAM] total={tot/1e6:.1f}M, trainable={trn/1e6:.1f}M")

    # 粗估初始化上界（单卡、ZeRO-3 之前）
    bytes_per_param = 2   # bf16 权重
    adam_state = 16       # Adam m,v 各 8B
    master_fp32 = 4       # FP32 master
    rough = trn*(bytes_per_param+adam_state+master_fp32)/(1024**3)
    print(f"[EST] optimizer+master upper bound ~= {rough:.1f} GiB")

    return model, tokenizer, image_processor, model_type

def setup_vla_data_args(data_args, image_processor, model_type):
    """Setup data arguments for VLA training"""
    # Standard Qwen-VL data args
    data_args.image_processor = image_processor
    data_args.model_type = model_type
    if hasattr(data_args, 'max_pixels'):
        data_args.image_processor.max_pixels = getattr(data_args, 'max_pixels', 1280*28*28)
    if hasattr(data_args, 'min_pixels'):
        data_args.image_processor.min_pixels = getattr(data_args, 'min_pixels', 256*28*28)
        
    data_args.image_processor.size = {
        "longest_edge": data_args.max_pixels,
        "shortest_edge": data_args.min_pixels
    }
    
    # VLA-specific args (with defaults)
    data_args.use_actions = getattr(data_args, 'use_actions', True)
    data_args.actions_format = getattr(data_args, 'actions_format', 'fast')
    data_args.action_tokenizer_path = getattr(data_args, 'action_tokenizer_path', None)
    data_args.action_dim = getattr(data_args, 'action_dim', 3)  # steering, acceleration, braking
    
    # Driving-specific args
    data_args.use_previous_actions = getattr(data_args, 'use_previous_actions', False)
    data_args.cur_frame_idx = getattr(data_args, 'cur_frame_idx', 3)
    
    rank0_print(f"VLA Data Config:")
    rank0_print(f"  - use_actions: {data_args.use_actions}")
    rank0_print(f"  - actions_format: {data_args.actions_format}")
    rank0_print(f"  - action_tokenizer_path: {data_args.action_tokenizer_path}")
    rank0_print(f"  - action_dim: {data_args.action_dim}")
    
    return data_args

def train():
    global local_rank
    
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    rank0_print("="*50)
    rank0_print("Qwen-VL VLA Training Setup")
    rank0_print("="*50)
    
    # Setup model and tokenizer (includes all training configuration)
    model, tokenizer, image_processor, model_type = setup_vla_model_and_tokenizer(model_args, training_args)
    
    # Setup data
    data_args = setup_vla_data_args(data_args, image_processor, model_type)
    
    if data_args.dataset_type == "navsim2va_ross":
        data_module = make_supervised_data_module_navsim2_vla_ross(tokenizer=tokenizer, data_args=data_args)
    elif data_args.dataset_type == "nuplan2va_ross":
        data_module = make_supervised_data_module_nuplan2_vla_ross(tokenizer=tokenizer, data_args=data_args)
    else:
        raise ValueError(f"Unsupported dataset type: {data_args.dataset_type}")
    rank0_print("Using standard VLA data format")
    
        
    # Create trainer
    trainer = RossTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )
    
    # Print dataloader len
    if torch.distributed.get_rank() == 0:
        rank0_print(f"dataloader len:")
        rank0_print(f"{len(trainer.get_train_dataloader())}")
        rank0_print(f"{training_args.gradient_accumulation_steps}")
        print("len(train_dataset) =", len(data_module["train_dataset"]))
        print("per_device_train_batch_size =", training_args.per_device_train_batch_size)
        print("TrainingArguments.world_size =", training_args.world_size)
        print("torch.distributed.get_world_size() =", torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
    
    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Checkpoint found, resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting training from scratch...")
        trainer.train()
    
    # Save final model
    trainer.save_state()
    
    # Save image processor
    if hasattr(data_args, 'image_processor'):
        data_args.image_processor.save_pretrained(training_args.output_dir)
    
    # Save tokenizer (with new VLA tokens)
    tokenizer.save_pretrained(training_args.output_dir)
    
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    rank0_print("="*50)
    rank0_print("Training completed successfully!")
    rank0_print(f"Model saved to: {training_args.output_dir}")
    rank0_print("="*50)

if __name__ == "__main__":
    train()