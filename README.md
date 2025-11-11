# DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving
> 📜 [[Arxiv](http://arxiv.org/abs/2510.12796)] 🤗 [[Model Weights](https://huggingface.co/liyingyan/DriveVLA-W0)]

Yingyan Li*, Shuyao Shang*, Weisong Liu*, Bing Zhan*, Haochen Wang*, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan†, Zhaoxiang Zhang†

This Paper presents **DriveVLA-W0**, a training paradigm that employs world modeling to predict future images. This task generates a dense, self-supervised signal that compels the model to learn the underlying dynamics of the driving environment, remedying the "supervision deficit" in VLA models and amplifying data scaling laws.

<p align="center">
  <img src="assets/fig1.png" alt="DriveVLA-W0" width="1000"/>
</p>


> Due to company policy, only the reviewed portion of our code is currently available. Please contact us if you have any questions.

# Install
## CUDA install
```
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
bash cuda_12.8.1_570.124.06_linux.run  --silent --toolkit --toolkitpath=/mnt/vdb1/yingyan.li/cuda
export CUDA_HOME=/mnt/vdb1/yingyan.li/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/mnt/vdb1/yingyan.li/repo/VLA_Emu_Huawei/reference/Emu3:$PYTHONPATH
```

## Conda 
```
conda create -n emu_vla python=3.10
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "transformers[torch]"
pip install deepspeed
pip install scipy
pip install tensorboard==2.14.0
pip install wandb
```

### Testing

First, please download the checkpoints from [Hugging Face](https://huggingface.co/liyingyan/DriveVLA-W0). 

Then, run the corresponding testing script to get output actions as json files
```
bash inference/vla/infer_navsim_with_previous_action_last_vava.sh
```
Finally, run the following script to compute PDMS from json files (using the conda enviroment with [navsim](https://github.com/autonomousvision/navsim/tree/v1.1))
```
bash inference/vla/run_emu_vla_navsim_metric_others.sh
```

# 🏆 NAVSIM v1/v2 Benchmark SOTA

Here is a comparison with state-of-the-art methods on the NAVSIM test set, as presented in the paper. Our model, **DriveVLA-W0**, establishes a new state-of-the-art.

| Method | Reference | Sensors | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Human** | | | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |
| **_BEV-based Methods_** | | | | | | | | |
| LAW | ICLR'25 | 1x Cam | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| Hydra-MDP | arXiv'24 | 3x Cam + L | 98.3 | 96.0 | 94.6 | 100.0 | 78.7 | 86.5 |
| DiffusionDrive | CVPR'25 | 3x Cam + L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| WoTE | ICCV'25 | 3x Cam + L | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 |
| **_VLA-based Methods_** | | | | | | | | |
| AutoVLA | NeurIPS'25 | 3x Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| ReCogDrive | arXiv'25 | 3x Cam | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| **DriveVLA-W0*** | **Ours** | **1x Cam** | **98.7** | **99.1** | **95.3** | **99.3** | **83.3** | **90.2** |
| AutoVLA† | NeurIPS'25 | 3x Cam | 99.1 | 97.1 | 97.1 | 100.0 | 87.6 | 92.1 |
| **DriveVLA-W0†** | **Ours** | **1x Cam** | **99.3** | **97.4** | **97.0** | **99.9** | **88.3** | **93.0** |

# ⭐ Star 
If you find our work useful for your research, please consider giving this repository a star ⭐.

# 📜 Citation
If you find this work useful for your research, please consider citing our paper:
```
@article{li2025drivevla,
  title={DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving},
  author={Li, Yingyan and Shang, Shuyao and Liu, Weisong and Zhan, Bing and Wang, Haochen and Wang, Yuqi and Chen, Yuntao and Wang, Xiaoman and An, Yasong and Tang, Chufeng and others},
  journal={arXiv preprint arXiv:2510.12796},
  year={2025}
}
```

# Acknowledgements
We would like to acknowledge the following related works:

[**LAW (ICLR 2025)**](https://github.com/BraveGroup/LAW): Using latent world models for self-supervised feature learning in end-to-end autonomous driving.

[**WoTE (ICCV 2025)**](https://github.com/liyingyanUCAS/WoTE): Using BEV world models for online trajectory evaluation in end-to-end autonomous driving.

[**UniVLA**](https://github.com/baaivision/UniVLA): World modeling in the broader field of robotics.
