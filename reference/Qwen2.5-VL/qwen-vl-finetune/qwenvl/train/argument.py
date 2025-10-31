import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_type: Optional[str] = field(default="qwen2.5vl")
    sd_model_path: Optional[str] = field(default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    model_type: str = field(default="qwen2.5vl")
    sd_model_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    dataset_type: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    
    # VLA-specific arguments
    data_path: Optional[str] = field(default=None)
    use_actions: bool = field(default=True)
    actions_format: str = field(default="fast")
    action_tokenizer_path: Optional[str] = field(default=None)
    action_dim: int = field(default=3)
    use_previous_actions: bool = field(default=False)
    cur_frame_idx: int = field(default=0)
    future_nums: int = field(default=8)
    data_root: Optional[str] = field(default=None)
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    dataloader_shuffle: bool = field(default=True)
    save_only_model: bool = field(default=False, metadata={"help": "Save only model weights, not optimizer states"})
    min_learning_rate: Optional[float] = field(default=None)