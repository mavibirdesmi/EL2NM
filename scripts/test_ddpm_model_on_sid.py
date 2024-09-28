import argparse
import os
import sys
import glob

import json
import torchvision
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from el2nm.models import ddpm
import el2nm.helper.canon_supervised_dataset as dset
import rawpy
import colour_demosaicing
from typing import Tuple, List

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def normalize(raw_image, black_level, white_level, clip=True):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:

        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]
    normalized_image = raw_image.astype(np.float32) - black_level_mask
    # if some values were smaller than black level
    if clip:
        normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    return normalized_image


def get_all_patches (im : np.ndarray, patch_shape : Tuple[int, int]) -> List[np.ndarray]:
    # shape of im is # H, W, C
    h, w, c = im.shape
    for row_idx in range(0, h, patch_shape[0]):
        for col_idx in range(0, w, patch_shape[1]):
            im
    im[::patch_shape[0], ::patch_shape[1]]


with open("scripts/configs.yml", "r") as fileb:
    config = yaml.safe_load(fileb)

config = dict2namespace(config)
print(config)

model_path = "context_best_2_clear"

diffusion = ddpm.DenoisingDiffusion(config)
diffusion.load_ddm_ckpt(model_path)
image_path = "/home/desmin/Projects/audio_driven_vdlle/datasets/SID/Sony/long/10093_00_30s.ARW"
raw_image_data = rawpy.imread(image_path)
raw_img_arr = raw_image_data.raw_image_visible.copy()
raw_img_arr = np.expand_dims(raw_img_arr, -1)

in_raw_image = normalize(
    raw_img_arr,
    raw_image_data.black_level_per_channel,
    raw_image_data.camera_white_level_per_channel[0],
    clip=True
)

in_raw_image = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(in_raw_image, pattern="BGGR")
# 2848, 4256, 3

in_raw_image = in_raw_image[:128, :128, :]
x_tensor = torch.tensor(in_raw_image, dtype=torch.float32) # H, W, C
x_tensor = x_tensor.unsqueeze(0)
x_tensor = x_tensor.repeat_interleave(16, dim=0) # N, H, W, C
x_tensor = x_tensor.permute(3, 0, 1, 2) # C, N, H, W
x_tensor = x_tensor.unsqueeze(0) # 1, C, N, H, W


# image is loaded as N, H, W, C
# toTensor2() -> C, N, H, W
# batcher -> 1, C, N, H, W
# transpose -> N, C, 1, H, W
# unsqueeze -> N, C, H, W

# x = torch.randn(1, 16, 3, 128, 128).to("cuda")

n = 16
t = torch.randint(low=0, high=diffusion.num_timesteps, size=(n // 2 + 1,)).to(diffusion.device)
t = torch.cat([t, diffusion.num_timesteps - t - 1], dim=0)[:n]

out = diffusion.sample_one(
    im_clean=x_tensor,
    im_noisy=None,
    step=diffusion.step,
    t=t
)
print(out.shape)