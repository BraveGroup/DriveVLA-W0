import os
import os.path as osp
import pickle
from tqdm import tqdm
import numpy as np
import sys
import yaml

# project-specific imports
from pyquaternion import Quaternion
sys.path.append("/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/tools/pickle_gen")
from navsim_coor import StateSE2, convert_absolute_to_relative_se2_array, normalize_angle


# split = 'test' 
# scene_filter = 'navtest'
split = 'trainval'
scene_filter = 'navtrain'

# --- 常量定义 ---
WINDOW = 12       # 从 -3 … +8 共 12 帧
CENTER_IDX = 3    # 当前帧在 window 中的索引

# --- 加载nuplan文件 ---
nuplan_pickle = '/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/data/nuplan/processed_data/meta/nuplan_processed_data.pkl'
nupaln_vq_root = '/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/data/nuplan/processed_data/vq_codes_low_res_corrected_merge'
with open(nuplan_pickle, 'rb') as f:
    nuplan_data = pickle.load(f)


logs_path     = f'/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/data/navsim/navsim_logs/{split}'
dataset_path  = "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/data/navsim/processed_data"
output_path   = "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/data/navsim/processed_data/meta"
vq_dir        = f"{dataset_path}/{split}_vq_codes_256_144/"
yaml_file     = f'data/navsim/processed_data/scene_files/scene_filter/{scene_filter}.yaml'
output_file_name = f"navsim_emu_vla_256_144_{split}_pre_1s.pkl"

os.makedirs(output_path, exist_ok=True)

text_name_list = [
    "go left",
    "go straight",
    "go right",
    'unknown',
]

# --- 载入 token 列表 ---
with open(yaml_file, 'r') as f:
    scene_list = yaml.safe_load(f)
token_list = scene_list['tokens']

# --- Phase 1: 构建 scene_dict_all，包含 action_list & image_list ---
scene_dict_all = {}
# debug_list = []

for log_name in tqdm(os.listdir(logs_path), desc="Processing logs"):
    log_path = osp.join(logs_path, log_name)
    if not log_path.endswith('.pkl'):
        continue

    scene = pickle.load(open(log_path, "rb"))
    num_frames = len(scene)

    # 1. 读出所有全局 SE2 pose
    global_ego_poses = []
    for fi in scene:
        t = fi["ego2global_translation"]
        q = Quaternion(*fi["ego2global_rotation"])
        yaw = q.yaw_pitch_roll[0]
        global_ego_poses.append([t[0], t[1], yaw])
    global_ego_poses = np.array(global_ego_poses, dtype=np.float64)  # (N,3)

    # 2. 批量计算 rel_all[i,j]：(dx,dy,dθ)
    rel_all = []
    for i in range(num_frames):
        origin = StateSE2(*global_ego_poses[i])
        rel = convert_absolute_to_relative_se2_array(origin, global_ego_poses)
        rel_all.append(rel)
    rel_all = np.stack(rel_all, axis=0)  # (N, N, 3) # rel_all[i,j] = i as origin, j's pose

    # 3. 对每帧写入 relative_action 和 image_list
    for i, fi in enumerate(scene):

        # 3.1 构造 action_list


        idxs = list(range(i - CENTER_IDX, i - CENTER_IDX + WINDOW))
        action_list = []
        for j in idxs:
            if 0 <= j < num_frames-1:
                dx, dy, dtheta = rel_all[j, j+1] #only relative
            else:
                dx = dy = dtheta = 0.0
            action_list.append([float(dx), float(dy), float(dtheta)])
        fi["relative_action_list"] = action_list

        # 3.2 构造 image_list
        image_list = []
        for j in idxs:
            if 0 <= j < num_frames:
                cam_path = scene[j]['cams']['CAM_F0']['data_path']
                log_n, _, fname = cam_path.split('/')
                vq_name = fname.replace('jpg', 'npy')
                image_list.append(osp.join(vq_dir, log_n, vq_name))
            else:
                image_list.append(None)
        fi["image_vq_list"] = image_list
        

        # 3.3 构造 text_list
        text_list = []
        for j in idxs:
            if 0 <= j < num_frames:
                driving_command = scene[j]['driving_command']
                driving_text_idx = scene[j]['driving_command'].nonzero()[0].item()
                text_driving_command_init = text_name_list[driving_text_idx]
                text_driving_command = f"Driving command: {text_driving_command_init}."

                # ego status
                ego_dynamic_state = scene[j]['ego_dynamic_state']
                ego_velocity = ego_dynamic_state[:2]
                ego_acceleration = ego_dynamic_state[2:4]
                local_ego_pose = rel_all[i, j].tolist()

                # text_local_ego_pose = f"Local ego car pose: x: {local_ego_pose[0]:.2f} m y: {local_ego_pose[1]:.2f} m yaw: {np.rad2deg(local_ego_pose[2]):.2f}."
                text_ego_velocity = f"Local ego car velocity: x: {ego_velocity[0]:.2f} m/s y: {ego_velocity[1]:.2f} m/s." 
                text_ego_acceleration = f"Local ego car acceleration: x: {ego_acceleration[0]:.2f} m/s^2  y: {ego_acceleration[1]:.2f} m/s^2."
                text_ego_status = f"{text_ego_velocity} {text_ego_acceleration}"
                
                text = text_driving_command_init
                text_list.append(text)
            else:
                text_list.append(None)
        fi["text_list"] = text_list

        # 对于navsim数据集的每个场景的前8帧图像的前4s数据在nuplan数据集中查找
        if i < 2:
            fi["pre_1s_relative_action_list"] = fi["relative_action_list"]
            fi["pre_1s_text_list"] = fi["text_list"]
            fi["pre_1s_image_vq_list"] = fi["image_vq_list"]

        else:
            fi["pre_1s_relative_action_list"] = scene[i-2]["relative_action_list"]
            fi["pre_1s_text_list"] = scene[i-2]["text_list"]
            fi["pre_1s_image_vq_list"] = scene[i-2]["image_vq_list"]
            if not osp.exists(fi['pre_1s_image_vq_list'][3]):
                # print(f"Warning: pre_1s_image_vq_list[3] does not exist in navsim, we well check it in nuplan dataset")
                img_nuplan = fi['pre_1s_image_vq_list'][3].replace(vq_dir, nupaln_vq_root)
                if osp.exists(img_nuplan):
                    fi['pre_1s_image_vq_list'][3] = img_nuplan
                else:
                    print(f"{fi['pre_1s_image_vq_list'][3]} do not in nuplan dataset, please check it")

        # 3.3 存入 scene_dict_all
        token = fi.pop("token")
        scene_dict_all[token] = fi
            
        # debug_list.append(fi)
# print(len(debug_list))

# --- Phase 2: 按 token_list 顺序生成 result_file ---
result_file = []
not_exist_file_num = 0
for token in tqdm(token_list, desc="Generating result_file"):
    info = scene_dict_all[token]
    result_file.append({
        "token":  token,
        "text":   info["text_list"],                  # 可根据需求添加文本
        "image":  info["image_vq_list"],  # WINDOW 长度列表
        "action": info["relative_action_list"],
        "pre_1s_text": info["pre_1s_text_list"],      # 前 4 秒的文本
        "pre_1s_image": info["pre_1s_image_vq_list"],
        "pre_1s_action": info["pre_1s_relative_action_list"],
    })

print(f"Total number of scenes: {len(result_file)}")
print(f"Number of scenes with no pre_1s_image: {not_exist_file_num}")


# --- Phase 3: 归一化 action_list ---
sys.path.append("/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu")
from train.dataset.normalize_pi0 import RunningStats, save, load
normalizer_path = f"/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/configs/normalizer_navsim_{split}"
os.makedirs(normalizer_path, exist_ok=True)

# Initialize RunningStats
normalizer = RunningStats()
# Aggregate action data
action_data = np.concatenate([scene["action"] for scene in result_file])

# Update statistics
normalizer.update(action_data)

# Get normalization statistics
norm_stats = normalizer.get_statistics()

# Print normalization parameters
print("Mean:", norm_stats.mean)
print("Standard Deviation:", norm_stats.std)
print("Q01 (1% quantile):", norm_stats.q01)
print("Q99 (99% quantile):", norm_stats.q99)

# Convert statistics to a JSON-compatible format
norm_stats_save = {
    "libero": norm_stats,
}

# Save normalizer parameters
save(normalizer_path, norm_stats_save)

# Normalize actions
for scene in result_file:
    action = scene["action"].copy()
    pre_1s_action = scene["pre_1s_action"].copy()
    # Normalize and clip
    normalized = 2 * (action - norm_stats.q01) / (norm_stats.q99 - norm_stats.q01 + 1e-8) - 1
    scene["action"] = np.clip(normalized, -1, 1)
    # Normalize pre_1s_action and clip
    pre_1s_normalized = 2 * (pre_1s_action - norm_stats.q01) / (norm_stats.q99 - norm_stats.q01 + 1e-8) - 1
    scene["pre_1s_action"] = np.clip(pre_1s_normalized, -1, 1)
    # Decode check
    # action_decoded = 0.5 * (scene["action"] + 1) * (norm_stats.q99 - norm_stats.q01) + norm_stats.q01

# --- 保存 ---
output_file = osp.join(output_path, output_file_name)
with open(output_file, "wb") as f:
    pickle.dump(result_file, f)


#
type1 = 0
with open(output_file, 'rb') as f:
    navsim_data = pickle.load(f)
if split == 'test':
    for item in navsim_data:
        if item['pre_1s_image'] == item['image']:
            navsim_img = item['image'][3].replace(vq_dir, "")
            for nuplan_item in nuplan_data:
                if navsim_img in nuplan_item['image']:
                    idx = nuplan_item['image'].index(navsim_img)
                    if idx >= 10+15:
                        previous_action_list = []
                        idx = idx - 10
                        item['pre_1s_image'][3] = nuplan_item['image'][idx]
                        item['pre_1s_text'][3] = nuplan_item['text'][idx]
                        for v in [15,10,5]:
                            previous_action_list.append(nuplan_item['action'][idx-v])
                        previous_action_list = np.array(previous_action_list)
                        item['pre_1s_action'] = np.concatenate([previous_action_list, nuplan_item['action'][idx]])
                    else :
                        type1 += 1 
                    break
    print(f"Total number of scenes: {len(navsim_data)}")
    print(f"Number of scenes with no pre_1s_image: {type1}")
    with open(output_file, 'wb') as f:
        pickle.dump(navsim_data, f)
    print(f"Processed data saved to {output_file}")

else:
    data_need_process = [item for item in navsim_data if item['pre_1s_image'] == item['image']]
    data_correct = [item for item in navsim_data if item['pre_1s_image'] != item['image']]
    data_process = []
    for idx, item in enumerate(data_need_process):
        navsim_img = item['image'][3].replace(vq_dir, "")
        for nuplan_item in nuplan_data:
            if navsim_img in nuplan_item['image']:
                idx = nuplan_item['image'].index(navsim_img)
                if idx >= 10+15:
                    previous_action_list = []
                    idx = idx - 10
                    item['pre_1s_image'][3] = nuplan_item['image'][idx]
                    item['pre_1s_text'][3] = nuplan_item['text'][idx]
                    for v in [15,10,5]:
                        previous_action_list.append(nuplan_item['action'][idx-v])
                    previous_action_list = np.array(previous_action_list)
                    item['pre_1s_action'] = np.concatenate([previous_action_list, nuplan_item['action'][idx]])
                    data_process.append(item)
                else:
                    type1 += 1 
                break
    data_save = data_process + data_correct
    print(f"Total number of scenes: {len(data_save)}")
    print(f"Number of scenes with no pre_1s_image: {type1}")
    with open(output_file, 'wb') as f:
        pickle.dump(data_save, f)
    print(f"Processed data saved to {output_file}")




