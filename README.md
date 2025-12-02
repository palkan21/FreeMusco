# FreeMusco: Motion-Free Learning of Latent Control for Morphology-Adaptive Locomotion in Musculoskeletal Characters
Official implementation of SIGGRAPH Asia 2025 Conference Paper: <BR>
FreeMusco: Motion-Free Learning of Latent Control for Morphology-Adaptive Locomotion in Musculoskeletal Characters <BR>

Authors: [Minkwan Kim](https://cgrhyu.github.io/people/minkwan-kim.html) and [Yoonsang Lee](https://cgrhyu.github.io/people/yoonsang-lee.html) (Hanyang University, Computer Grahpics and Robotics Lab) <BR>
Overview, demo results, video and paper can be found on [Paper website](https://cgrhyu.github.io/publications/2025-freemusco.html)

This code is developed based on open-source structure of [Control-VAE](https://github.com/heyuanYao-pku/Control-VAE) <BR>
Tested on Ubuntu 20.04, 22.04, and 24.04; issues may still occur depending on hardware or GPU.

## How to install
```bash
conda env create -f freemusco.yml
conda activate freemusco
#cd FreeMusco
conda install pytorch=*=*cuda* torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install mpi4py

pip install panda3d mujoco==3.2.3
pip install -e .
pip install gymnasium pygame
```

## Pretrained Checkpoints
[Google Drive](https://drive.google.com/drive/folders/1411vveAVjGZREY0TDieMdqhpoUeUpRl2?usp=drive_link) Please download and put data files into directory "Data/Pretrained"

## How to run (Inference)
```bash
python3 Run/playground.py #please run in base directory
```
Please select config files that you want to render. (in the directory "Data/Pretrained") <BR>
Corresponding checkpoint will be automatically loaded (First, you need to put data files into the directory "Data/Pretrained")

## Keyboard Shortcut (Inference)
Press **`R`** for reset, **`Space`** for pause/start, **`U`** for render muscle activation. <BR> 
Press **`Q`**, **`Z`**, **`E`** for velocity and direction change (**`Q`** is low, **`Z`** is medium, **`E`** is high velocity). <BR> 

Press **`O`** for energy decreasing and **`L`** for energy increasing (VelEnergyPose config). <BR> 
Press **`B`** for identity target pose, **`N`** for random target pose (VelEnergyPose config). <BR> 

Press **`V`** for change latent sampling method (goal-state-conditioned latent from posterior or state-conditioned latent from prior) <BR> 

## How to run (train)
```bash
python3 Run/train.py #please run in base directory
```
Train code will be updated soon.

## Citation
Citation information for our paper will be updated soon.

## Acknowledgement
This code is developed based on open-source structure of [Control-VAE](https://github.com/heyuanYao-pku/Control-VAE) <BR>
If you find that Conditional-Vae and Model-based RL system is helpful, please consider citing below: <BR>
```bash
@article{
    ControlVAE,
    author = {Yao, Heyuan and Song, Zhenhua and Chen, Baoquan and Liu, Libin},
    title = {ControlVAE: Model-Based Learning of Generative Controllers for Physics-Based Characters},
    year = {2022},
    issue_date = {December 2022},
    volume = {41},
    number = {6},
    url = {https://doi.org/10.1145/3550454.3555434},
    journal = {ACM Trans. Graph.},
    articleno = {183},
}
```
