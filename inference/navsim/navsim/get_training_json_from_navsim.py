import os
import json
from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

SPLIT = "test"  # ["mini", "test", "trainval"]
FILTER = "navtest"

IMAGE_DIR = "/mnt/vdb1/yuntao.chen/data/nuplan/images/"

hydra.initialize(config_path="navsim/planning/script/config/common/train_test_split/scene_filter")
cfg = hydra.compose(config_name=FILTER)
scene_filter: SceneFilter = instantiate(cfg)
openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}",
    openscene_data_root / f"sensor_blobs/{SPLIT}",
    scene_filter,
    sensor_config=SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False),
)

tokens = scene_loader.tokens
for token in tqdm(tokens):
    scene = scene_loader.get_scene_from_token(token)
    scene_dict = {}
    scene_dict["images"] = []
    scene_dict["poses"] = []
    scene_dict["ego_status"] = []
    for frame in scene.frames:
        image_path = os.path.join(*str(frame.cameras.cam_f0.image).split("/")[-3:])
        assert os.path.exists(os.path.join(IMAGE_DIR, image_path)), f"Image path {image_path} does not exist"
        scene_dict["images"].append(image_path)
        ego_status = np.concatenate([frame.ego_status.ego_velocity, frame.ego_status.ego_acceleration, frame.ego_status.driving_command]).tolist()
        scene_dict["ego_status"].append(ego_status)
    poses = np.concatenate([scene.get_history_trajectory().poses, scene.get_future_trajectory().poses], axis=0)
    from get_rel_pose_from_abs_pose import calculate_relative_pose
    scene_dict["poses"] = calculate_relative_pose(poses).tolist()

    # trim length to the minimum length of images, poses, and ego_status
    min_length = min(len(scene_dict["images"]), len(scene_dict["poses"]), len(scene_dict["ego_status"]))
    assert min_length == 13, f"Min length is {min_length} for scene {scene.scene_metadata.log_name}_{token}"
    scene_dict["images"] = scene_dict["images"][:min_length]
    scene_dict["poses"] = scene_dict["poses"][:min_length]
    scene_dict["ego_status"] = scene_dict["ego_status"][:min_length]

    SAVE_DIR = "/mnt/vdb1/yuntao.chen/data/navsim_train_vanilla_json/"
    FILE_NAME_PATTERN = f"{scene.scene_metadata.log_name}_{token}.json"

    SAVE_DIR = "/mnt/vdb1/yuntao.chen/data/navsim_test_vanilla_json/"
    try:
        FILE_NAME_PATTERN = f"{scene.scene_metadata.initial_token}_{str(scene.frames[3].cameras.cam_f0.image).split('/')[-1].split('.')[0]}.json"
    except Exception as e:
        import ipdb; ipdb.set_trace()

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    with open(f"{SAVE_DIR}/{FILE_NAME_PATTERN}", "w") as f:
        json.dump(scene_dict, f)


