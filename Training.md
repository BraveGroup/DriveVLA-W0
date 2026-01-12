# Download pretrain model
```
pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
mkdir pretrained_models
scripts/misc/download_emu3_pretrain.sh
```

## vq_index generation
```
python tools/pickle_gen/pickle_generation_navsim_pre_1s.py
```
*Note: You can also directly download `train_vq_codes.zip` and `test_vq_codes.zip` from Hugging Face.*


## navsim pickle generation
```
bash scripts/tokenizer/extract_vq_emu3_navsim.sh
```
*Note: You can also download existing pickle `navsim_emu_vla_256_144_test_pre_1s.pkl`, `navsim_emu_vla_256_144_trainval_pre_1s.pkl` directly from Hugging Face.*


## Stage 1: Pretraining world model on Nuplan dataset
```
bash scripts/pretrain/train_nuplan_6va_multi_v0.2_worker.sh
```
*Note: You can also download existing ckpt `Emu3_NuPlan_Pretrain_Cktps` directly from Hugging Face.*

## Stage 2: Finetuning world model on Navsim dataset
### train base AR model without MoE
```
bash scripts/scripts_train/train_base_ar_withou_moe.sh
```

### train ar with MoE
```
bash scripts/scripts_train/train_navsim_ar.sh
```

### train flow matching with MoE
```
bash scripts/scripts_train/train_navsim_flow_matching.sh
```

### train query-based with MoE
```
bash scripts/scripts_train/train_navsim_query_based.sh
```