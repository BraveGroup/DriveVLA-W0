## Download pretrain model
```
pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
mkdir pretrained_models
bash scripts/misc/download.sh
```

### vq_index generation
```
python tools/pickle_gen/pickle_generation_navsim_pre_1s.py
```

### navsim pickle generation
```
bash /mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/scripts/tokenizer/extract_vq_emu3_navsim.sh
```

### training world model
```
bash scripts/pretrain/train_nuplan_6va_multi_v0.2_worker.sh
```

### training base ar
```
bash scripts/simulator/navsim_quick/finetune_navsim_quick_8GPUs_vision_loss_vava_pre1s_lr8e5.sh
```