"""
配置管理模块
支持从环境变量和 YAML 配置文件加载配置
"""
import os
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


def get_project_root() -> Path:
    """
    自动检测项目根目录
    基于当前文件位置向上查找，直到找到包含特定标记文件的目录
    """
    current_file = Path(__file__).resolve()
    # 从当前文件位置向上查找，直到找到项目根目录
    # 项目根目录应该包含 train/ 和 reference/ 目录
    for parent in current_file.parents:
        if (parent / "train").exists() and (parent / "reference").exists():
            return parent
    # 如果找不到，使用环境变量
    if "DRIVEVLA_ROOT" in os.environ:
        return Path(os.environ["DRIVEVLA_ROOT"])
    # 最后回退到当前文件的父目录的父目录（inference/vla -> inference -> root）
    return current_file.parent.parent.parent


def setup_paths_early():
    """
    早期路径设置函数，在导入其他模块之前调用
    不依赖配置文件，只设置基本的 Python 路径
    """
    project_root = get_project_root()
    
    # 添加 train 目录
    train_dir = project_root / "train"
    if train_dir.exists() and str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    
    # 添加 reference/Emu3 目录
    emu3_path = project_root / "reference" / "Emu3"
    if emu3_path.exists() and str(emu3_path) not in sys.path:
        sys.path.insert(0, str(emu3_path))


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，如果为 None，则尝试从环境变量或默认位置加载
        """
        self.project_root = get_project_root()
        self.config = {}
        
        # 加载配置文件
        if config_path is None:
            config_path = os.environ.get("VLA_CONFIG", None)
            if config_path is None:
                # 尝试默认位置
                default_config = self.project_root / "inference" / "vla" / "config.yaml"
                if default_config.exists():
                    config_path = str(default_config)
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        
        # 设置默认值
        self._set_defaults()
        
        # 环境变量覆盖（优先级最高）
        self._load_from_env()
    
    def _set_defaults(self):
        """设置默认配置值"""
        defaults = {
            "project_root": str(self.project_root),
            "paths": {
                "action_tokenizer": None,
                "vlm_model": None,
                "norm_stats": None,
                "token_yaml": "data/navsim/processed_data/scene_files/scene_filter/navtest.yaml",
            },
            "model": {
                "model_max_length": 1400,
                "padding_side": "right",
                "use_fast": False,
                "attn_implementation": "sdpa",
                "torch_dtype": "bfloat16",
            },
            "data": {
                "batch_size": 1,
                "num_workers": 12,
                "pin_memory": True,
                "frames": 1,
                "action_frames": 8,
                "action_dim": 3,
                "cur_frame_idx": 3,
                "pre_action_frames": 3,
            },
            "inference": {
                "num_inference_steps": 10,
            },
        }
        
        # 合并默认值
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in self.config[key]:
                        self.config[key][sub_key] = sub_value
    
    def _load_from_env(self):
        """从环境变量加载配置（覆盖配置文件）"""
        env_mapping = {
            "DRIVEVLA_ROOT": ("project_root", None),
            "VLA_ACTION_TOKENIZER": ("paths", "action_tokenizer"),
            "VLA_VLM_MODEL": ("paths", "vlm_model"),
            "VLA_NORM_STATS": ("paths", "norm_stats"),
            "VLA_TOKEN_YAML": ("paths", "token_yaml"),
            "VLA_BATCH_SIZE": ("data", "batch_size", int),
            "VLA_NUM_WORKERS": ("data", "num_workers", int),
            "VLA_MODEL_MAX_LENGTH": ("model", "model_max_length", int),
            "VLA_NUM_INFERENCE_STEPS": ("inference", "num_inference_steps", int),
        }
        
        for env_var, (section, key, *type_hint) in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                if type_hint and type_hint[0] is int:
                    value = int(value)
                elif type_hint and type_hint[0] is bool:
                    value = value.lower() in ("true", "1", "yes")
                
                if key is None:
                    self.config[section] = value
                else:
                    if section not in self.config:
                        self.config[section] = {}
                    self.config[section][key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的路径
        
        Args:
            key_path: 配置路径，如 "paths.action_tokenizer"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split(".")
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_path(self, key_path: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取路径配置，如果是相对路径则相对于项目根目录
        
        Args:
            key_path: 配置路径
            default: 默认值
            
        Returns:
            绝对路径字符串
        """
        path = self.get(key_path, default)
        if path is None:
            return None
        path = str(path)
        if os.path.isabs(path):
            return path
        return str(self.project_root / path)
    
    def setup_paths(self):
        """设置 Python 路径"""
        # 添加 train 目录
        train_dir = self.project_root / "train"
        if train_dir.exists() and str(train_dir) not in sys.path:
            sys.path.insert(0, str(train_dir))
        
        # 添加 reference/Emu3 目录
        emu3_path = self.project_root / "reference" / "Emu3"
        if emu3_path.exists() and str(emu3_path) not in sys.path:
            sys.path.insert(0, str(emu3_path))


# 全局配置实例
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    获取全局配置实例（单例模式）
    
    Args:
        config_path: 配置文件路径，如果为 None 且已有实例，则返回现有实例
        
    Returns:
        Config 实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
        # setup_paths 已经在早期调用过了，这里不需要重复调用
    elif config_path is not None:
        # 如果提供了新的配置文件路径，重新加载配置
        _config_instance = Config(config_path)
    return _config_instance


def reset_config():
    """重置全局配置实例（主要用于测试）"""
    global _config_instance
    _config_instance = None

