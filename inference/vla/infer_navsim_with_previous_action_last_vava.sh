torchrun --nproc_per_node=8 inference/vla/inference_action_navsim_with_previous_action_last_VAVA.py \
    --emu_hub "/home/yingyan.li/repo/VLA_Emu/logs/train_navsim_action_token_last_small_256_144_use_nuplan_pretrain_vava" \
    --output_dir "/home/yingyan.li/repo/VLA_Emu/logs/train_navsim_action_token_last_small_256_144_use_nuplan_pretrain_vava/json_output_VAVA" \
    --train_meta_pkl "/home/yingyan.li/repo/VLA_Emu/data/navsim/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s.pkl" \
    --input_num_frame "1"

