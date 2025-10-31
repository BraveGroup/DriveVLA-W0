import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# 定义VLA需要的特殊tokens
VLA_SPECIAL_TOKENS = {
    "boa_token": "<action_start>",  # Begin of Action
    "eoa_token": "<action_end>",  # End of Action
}


def check_and_add_vla_tokens(
    tokenizer: PreTrainedTokenizer, 
    model: PreTrainedModel,
    force_add: bool = True
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, bool]:
    """
    检查并添加VLA特殊tokens
    
    Args:
        tokenizer: tokenizer
        model: model
        force_add: 强制添加tokens（即使已存在）
        
    Returns:
        (updated_tokenizer, updated_model, tokens_added)
    """
    
    # 检查当前tokenizer是否已有VLA tokens
    existing_tokens = set(tokenizer.additional_special_tokens or [])
    vla_tokens_needed = set(VLA_SPECIAL_TOKENS.values())
    
    missing_tokens = vla_tokens_needed - existing_tokens
    
    if not missing_tokens and not force_add:
        logger.info("VLA tokens already exist in tokenizer")
        # 为已存在的tokens添加属性访问
        tokenizer.boa_token = VLA_SPECIAL_TOKENS["boa_token"]
        tokenizer.eoa_token = VLA_SPECIAL_TOKENS["eoa_token"]
        return tokenizer, model, False
    
    # 添加缺失的特殊tokens
    new_tokens = list(missing_tokens) if not force_add else list(vla_tokens_needed)
    
    # 更新additional_special_tokens
    current_additional = tokenizer.additional_special_tokens or []
    updated_additional = list(set(current_additional + new_tokens))
    
    num_added_tokens = tokenizer.add_special_tokens({
        "additional_special_tokens": updated_additional
    })
    
    logger.info(f"Added {num_added_tokens} new VLA tokens: {new_tokens}")
    
    # 为tokenizer添加属性访问方式
    tokenizer.boa_token = VLA_SPECIAL_TOKENS["boa_token"]
    tokenizer.eoa_token = VLA_SPECIAL_TOKENS["eoa_token"]
    
    # 简单持久化到init_kwargs
    if not hasattr(tokenizer, "init_kwargs"):
        tokenizer.init_kwargs = {}
    tokenizer.init_kwargs["boa_token"] = tokenizer.boa_token
    tokenizer.init_kwargs["eoa_token"] = tokenizer.eoa_token
    
    # 如果添加了tokens，检查是否需要resize model embeddings
    if num_added_tokens > 0:
        new_tokenizer_size = len(tokenizer)
        current_model_vocab_size = model.get_input_embeddings().weight.shape[0]
        
        if new_tokenizer_size > current_model_vocab_size:
            logger.info(f"Resizing model embeddings from {current_model_vocab_size} to {new_tokenizer_size}")
            model.resize_token_embeddings(new_tokenizer_size)
        else:
            logger.info(f"No resize needed: tokenizer size {new_tokenizer_size} <= model vocab size {current_model_vocab_size}")
    
    return tokenizer, model, num_added_tokens > 0



def smart_load_model_and_tokenizer(
    model_name_or_path: str,
    model_class: type,
    add_vla_tokens: bool = True,
    cache_dir: str = None,
    torch_dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    device_map = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, bool]:
    from transformers import AutoTokenizer
    
    safe_model_kwargs = {}
    if cache_dir is not None:
        safe_model_kwargs["cache_dir"] = cache_dir
    if torch_dtype is not None:
        safe_model_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation is not None:
        safe_model_kwargs["attn_implementation"] = attn_implementation
    if device_map is not None:
        safe_model_kwargs["device_map"] = device_map
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path, **safe_model_kwargs)
    
    logger.info(f"Loaded model and tokenizer from: {model_name_or_path}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    if safe_model_kwargs:
        logger.info(f"Model kwargs: {safe_model_kwargs}")
    
    vla_tokens_added = False
    
    if add_vla_tokens:
        tokenizer, model, vla_tokens_added = check_and_add_vla_tokens(tokenizer, model)
    else:
        # 即使不新增，也要确保属性存在
        tokenizer.boa_token = getattr(tokenizer, "boa_token", VLA_SPECIAL_TOKENS["boa_token"])
        tokenizer.eoa_token = getattr(tokenizer, "eoa_token", VLA_SPECIAL_TOKENS["eoa_token"])
    
    return model, tokenizer, vla_tokens_added


def prepare_action_tokenizer_mapping(tokenizer: PreTrainedTokenizer) -> Dict[str, int]:
    # 使用tokenizer的属性
    boa_token = tokenizer.boa_token
    eoa_token = tokenizer.eoa_token

    # 通过encode获取token id，避免依赖tokenizer属性
    boa_ids = tokenizer.encode(boa_token, add_special_tokens=False)
    eoa_ids = tokenizer.encode(eoa_token, add_special_tokens=False)
    if len(boa_ids) == 0 or len(eoa_ids) == 0:
        raise ValueError(f"Failed to obtain IDs via encode for VLA tokens: {boa_token}, {eoa_token}")

    boa_token_id = boa_ids[0]
    eoa_token_id = eoa_ids[0]

    vocab_size = len(tokenizer)
    # 倒序插入action tokens从 pad_token_id - 1 开始；若未设置pad则退化为 vocab_size - 1
    if tokenizer.pad_token_id is not None:
        last_vocab_idx = tokenizer.pad_token_id - 1
    else:
        raise ValueError("Tokenizer pad_token_id is not set")

    mapping = {
        "boa_token_id": boa_token_id,
        "eoa_token_id": eoa_token_id,
        "last_vocab_idx": last_vocab_idx,
        "vocab_size": vocab_size,
        "action_token_range": (0, last_vocab_idx),
    }

    logger.info(f"Action tokenizer mapping prepared (encode-based): {mapping}")
    return mapping
