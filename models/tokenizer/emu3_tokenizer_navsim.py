import os
import os.path as osp
import torch
from transformers import AutoModel, AutoImageProcessor
import numpy as np
import sys
import json
import re
import time
from tqdm import tqdm
import argparse
from PIL import Image, ImageEnhance
import random

def random_shift(images, max_shift=0.1):
    """
    对一个图像列表进行随机平移
    :param images: List of PIL Images (视频的每一帧)
    :param max_shift: 最大平移比例
    :return: 平移后的图像列表
    """
    width, height = images[0].size
    shift_x = random.uniform(-max_shift, max_shift) * width
    shift_y = random.uniform(-max_shift, max_shift) * height
    
    # 对每一帧图像应用相同的平移
    shifted_images = [
        image.transform(
            (width, height),
            Image.AFFINE,
            (1, 0, shift_x, 0, 1, shift_y),
            resample=Image.BICUBIC
        ) for image in images
    ]
    
    return shifted_images

def random_color_enhance(images):
    """
    对一个图像列表进行随机颜色增强
    :param images: List of PIL Images (视频的每一帧)
    :return: 颜色增强后的图像列表
    """
    factor = random.uniform(0.5, 1.5)  # 随机调整颜色的增强因子
    
    # 对每一帧图像应用相同的颜色增强
    color_enhanced_images = [
        ImageEnhance.Color(image).enhance(factor) for image in images
    ]
    
    return color_enhanced_images

def random_brightness_enhance(images):
    """
    对一个图像列表进行随机亮度增强
    :param images: List of PIL Images (视频的每一帧)
    :return: 亮度增强后的图像列表
    """
    factor = random.uniform(0.5, 1.5)  # 随机调整亮度的增强因子
    
    # 对每一帧图像应用相同的亮度增强
    brightness_enhanced_images = [
        ImageEnhance.Brightness(image).enhance(factor) for image in images
    ]
    
    return brightness_enhanced_images

def load_images(folder_path, size, interval=2, augmentation=None):
    """Load and resize all images in the specified folder."""
    image_paths = sorted(os.listdir(folder_path),key=natural_sort_key)[::interval]
    images = [Image.open(osp.join(folder_path, img)).resize(size) for img in image_paths]

    if augmentation is not None:
        images = augmentation(images)

    return images,image_paths

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def load_images_from_json(json_path, video_root, size):
    """Load and resize images based on JSON paths."""
    with open(json_path, 'r') as f:
        image_paths = json.load(f)['images'][::2]
    images = [Image.open(osp.join(video_root, v)).resize(size) for v in image_paths]
    image_names = [osp.basename(v).split('.')[0] for v in image_paths]
    return images, image_names

def clip_level_enc_dec(images, model, processor, save_codes_path, save_recon_path, t=4):
    """Process images in batches and save codes and reconstructed images."""
    images_tensor = processor(images, return_tensors="pt")["pixel_values"].unsqueeze(0).cuda()
    num_frames = images_tensor.shape[1]
    num_batches = num_frames // t

    os.makedirs(save_codes_path, exist_ok=True)
    os.makedirs(save_recon_path, exist_ok=True)

    for batch in range(num_batches):
        image = images_tensor[:, batch * t : (batch + 1) * t]
        with torch.no_grad():
            codes = model.encode(image)
            x = codes.detach().cpu().numpy()
            np.save(f'{save_codes_path}/{batch:03d}.npy', x)
            recon = model.decode(codes)
        recon = recon.view(-1, *recon.shape[2:])
        recon_images = processor.postprocess(recon)["pixel_values"]
        for idx, im in enumerate(recon_images):
            im.save(f"{save_recon_path}/{batch * t + idx:03d}.jpg")

def image_level_enc_dec(images, model, processor, save_codes_path, save_recon_path, batch_size=1, image_paths=None):
    """Process images in batches: encode, decode, and save the reconstructed images and codes."""
    os.makedirs(save_codes_path, exist_ok=True)  # Ensure the codes directory exists
    # os.makedirs(save_recon_path, exist_ok=True) #A
    images_tensor = processor(images, return_tensors="pt")["pixel_values"].cuda()
    num_images = images_tensor.shape[0]
    for start_idx in range(0, num_images, batch_size):
        batch = images_tensor[start_idx:start_idx + batch_size]
        try:
            with torch.no_grad():
                # Encode the batch of images
                codes = model.encode(batch)
                # Decode the codes back to images
                # recon = model.decode(codes) #A
                # Save the encoded codes
                img_name = image_paths[start_idx].replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
                np.save(f'{save_codes_path}/{img_name}.npy', codes.detach().cpu().numpy())
            
            # recon = recon.view(-1, *recon.shape[2:])
            # recon_images = processor.postprocess(recon)["pixel_values"]
            # for idx, im in enumerate(recon_images):
            #     im.save(f"{save_recon_path}/image_{start_idx + idx:03d}.jpg")
        except Exception as e:
            print(f"Error processing batch starting at image {start_idx}: {e}")


def image_level_enc(images, image_paths, model, processor, save_codes_path, batch_size=8):
    """Process images in batches: encode and save the codes."""
    t1 = time.time()
    os.makedirs(save_codes_path, exist_ok=True)  # Ensure the codes directory exists
    images_tensor = processor(images, return_tensors="pt")["pixel_values"].cuda()
    num_images = images_tensor.shape[0]
    t2 = time.time()
    print(f"Time to load images: {(t2 - t1) * 1000} ms")
    for start_idx in range(0, num_images, batch_size):
        batch = images_tensor[start_idx:start_idx + batch_size]
        try:
            with torch.no_grad():
                t1 = time.time()
                # Encode the batch of images
                codes = model.encode(batch)
                t2 = time.time()
                print(f"Time to encode: {(t2 - t1) * 1000} ms")
                # Save the encoded codes
                for idx, code in enumerate(codes):
                    np.save(f'{save_codes_path}/{image_paths[start_idx + idx]}.npy', code.detach().cpu().numpy())
                t3 = time.time()
                print(f"Time to save codes: {(t3 - t2) * 1000} ms")
        except Exception as e:
            print(f"Error processing batch starting at image {start_idx}: {e}")

data_config = {
    'navsim': {
        'min_pixels': 256 * 144,
        'interval': 1,
        'SIZE': (256, 144),
        # 'VIDEO_ROOT': '/mnt/nvme0n1p1/yingyan.li/repo/OmniSim//data/navsim/sensor_blobs/trainval',
        # 'VIDEO_CODES_SAVE': '/mnt/nvme0n1p1/yingyan.li/repo/OmniSim//data/navsim/processed_data/trainval_vq_codes',
        'VIDEO_ROOT': '/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/data/navsim/sensor_blobs/test',
        'VIDEO_CODES_SAVE': '/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/data/navsim/processed_data/test_vq_codes',
        'VIDEO_RECON_SAVE': '/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/data/navsim/processed_data/recon'
    }
}

def get_data_config(process_data):
    cfg = data_config[process_data]

    interval = cfg['interval']
    size = cfg['SIZE']
    min_pixels = cfg['min_pixels']
    hz = cfg['hz_func'](interval) if 'hz_func' in cfg else None

    video_root = cfg['VIDEO_ROOT']
    video_codes_save = cfg['VIDEO_CODES_SAVE']
    video_recon_save = cfg['VIDEO_RECON_SAVE']

    return {
        'interval': interval,
        'min_pixels': min_pixels,
        'SIZE': size,
        'hz': hz,
        'VIDEO_ROOT': video_root,
        'VIDEO_CODES_SAVE': video_codes_save,
        'VIDEO_RECON_SAVE': video_recon_save
    }

if __name__ == "__main__":

    MODEL_HUB = "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/pretrained_models/Emu3-VisionTokenizer"
    path = "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu/pretrained_models/Emu3-VisionTokenizer"

    # choose the dataset to process
    process_data = 'navsim'

    model = AutoModel.from_pretrained(path, trust_remote_code=True).eval().cuda()
    processor = AutoImageProcessor.from_pretrained(MODEL_HUB, trust_remote_code=True)
    
    # Retrieve configuration for the selected dataset
    config = get_data_config(process_data)

    # Assign configuration values to variables used in your pipeline
    processor.min_pixels = config['min_pixels']  # Minimum valid pixel count
    interval = config['interval']                # Frame interval (e.g., sampling rate)
    SIZE = config['SIZE']                        # Desired image resolution (width, height)
    hz = config['hz']                            # Final video frequency (Hz)

    # Paths for loading raw video, saving codes, and reconstructed videos
    VIDEO_ROOT = config['VIDEO_ROOT']
    VIDEO_CODES_SAVE = config['VIDEO_CODES_SAVE']
    VIDEO_RECON_SAVE = config['VIDEO_RECON_SAVE']

    os.makedirs(VIDEO_CODES_SAVE, exist_ok=True)
    os.makedirs(VIDEO_RECON_SAVE, exist_ok=True)

    try:
        rank = int(sys.argv[1])
    except Exception as e:
        print(f"Error parsing rank: {e}")
    videos = sorted(os.listdir(VIDEO_ROOT))[rank::8]
    
    for video in tqdm(videos, desc="Processing videos"):
        print("processing videos: ", video)
        # Navsim Front camera
        images, image_paths = load_images(osp.join(VIDEO_ROOT, video, 'CAM_F0'), SIZE, interval)
        
        # Enumerate all images in the folder
        processed_codes_path = osp.join(VIDEO_CODES_SAVE, video) 
        recon_images_path = osp.join(VIDEO_RECON_SAVE, video)
        
        if os.path.exists(processed_codes_path):
            print(f"Skipping video {video} as it has already been processed.")
            continue
        # Clip-level encoding and decoding
        # clip_level_enc_dec(images_enumerated, model, processor, processed_codes_path, recon_images_path)
        
        # Image-level encoding and decoding for all enumerated images
        image_level_enc_dec(images, model, processor, processed_codes_path, recon_images_path, image_paths=image_paths)