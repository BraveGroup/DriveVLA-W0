torchrun --nproc_per_node=8 inference/vla/inference_action_navsim_qformer_vava.py \
    --emu_hub "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/logs/train_navsim_qformer_anchor_vava" \
    --output_dir "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/logs/train_navsim_qformer_anchor_vava/json_output" \
    --test_data_pkl "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s.pkl" 

