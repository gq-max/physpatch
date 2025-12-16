import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import hashlib
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch
import torchvision
import argparse
from PIL import Image
import hydra
from omegaconf import DictConfig
from config_schema import MainConfig
from functools import partial
from typing import List, Dict, Optional
from torch import nn
from pytorch_lightning import seed_everything
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import List, Dict, Optional
from tqdm import tqdm
from surrogates import (
    ClipB16FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
    EnsembleFeatureLoss_ours,
    EnsembleFeatureExtractor_ours,
    EnsembleFeatureLoss_ours_auto,
)
from utils import *
from attacks import *
# Mapping from backbone names to model classes
BACKBONE_MAP: Dict[str, type] = {
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}
def get_models(backbone_name=["B16", "B32", "Laion"], device='cuda'):
    
    models = []
    for backbone_name in backbone_name: 
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
            )
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(device).requires_grad_(False)
        models.append(model)

    ensemble_extractor = EnsembleFeatureExtractor(models)

    return ensemble_extractor, models

def get_models_ours(backbone_name=["B16", "B32", "L14"], device='cuda', k=8):
    
    models = []
    for backbone_name in backbone_name:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
            )
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(device).requires_grad_(False)
        models.append(model)

    ensemble_extractor = EnsembleFeatureExtractor_ours(models, k=k)

    return ensemble_extractor, models

def get_ensemble_loss(models: List[nn.Module]):
    ensemble_loss = EnsembleFeatureLoss(models)
    return ensemble_loss

def get_ensemble_loss_ours(models: List[nn.Module]):
    ensemble_loss = EnsembleFeatureLoss_ours_auto(models, k=8)
    return ensemble_loss


def set_environment(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Transform PIL.Image to PyTorch Tensor
def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


# Dataset with image paths
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index) # 
        path, _ = self.samples[index]
        return original_tuple + (path,)




def parse_args():
    parser = argparse.ArgumentParser(description="Run Adversarial Patch Attack")
    parser.add_argument('--cle_data_path', type=str, required=True, help='Path to clean/original images')
    parser.add_argument('--tgt_data_path', type=str, required=True, help='Path to target images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save adversarial images')
    parser.add_argument('--txt_path', type=str, required=True, help='Path to coordinate txt file')
    parser.add_argument('--input_height', type=int, default=900)
    parser.add_argument('--input_width', type=int, default=1600)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--crop_scale_min', type=float, default=0.5)
    parser.add_argument('--crop_scale_max', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=16)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--num_iters', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--K', type=int, default=10, help='SVD component count')
    return parser.parse_args()

def main():
    args = parse_args()
    set_environment(42)
    device = args.device
    input_res = [args.input_height, args.input_width]

    crop_scale = [args.crop_scale_min, args.crop_scale_max]

    ensemble_extractor, models = get_models_ours(
        backbone_name=["B16", "B32", "Laion"], k=args.K
    )
    ensemble_loss = EnsembleFeatureLoss_ours_auto(models, k=args.K)

    transform_fn = transforms.Compose([
        transforms.Resize(input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(lambda img: to_tensor(img)),
    ])

    clean_data = ImageFolderWithPaths(args.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(args.tgt_data_path, transform=transform_fn)
    data_loader_clean = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False)
    data_loader_target = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False)

    with open(args.txt_path, 'r') as f:
        coords = [list(map(float, line.strip().split(','))) for line in f.readlines()]
    original_centers = torch.tensor(coords)
    center_batches = [original_centers[i:i + args.batch_size] for i in range(0, len(original_centers), args.batch_size)]

    for i, ((image_org, _, path_org), (image_tgt, _, path_tgt), center) in enumerate(
        zip(data_loader_clean, data_loader_target, center_batches)
    ):
        if args.batch_size * (i + 1) > args.num_samples:
            break

        source_crop = RandomPointConstrainedCrop(input_res, scale=crop_scale, norm_coord=center)
        target_crop = transforms.RandomResizedCrop(input_res, scale=crop_scale)

        print(f"\nProcessing image {i + 1}/{args.num_samples // args.batch_size}")
        image_org = image_org.to(device)
        image_tgt = image_tgt.to(device)

        final_image, final_patch = pgd(
            image_org, image_tgt, ensemble_extractor, ensemble_loss,
            source_crop, target_crop, center,
            num_iters=args.num_iters, epsilon=args.epsilon, alpha=args.alpha, device=device
        )

        for path_idx in range(len(path_org)):
            folder, name = path_org[path_idx].split("/")[-2], path_org[path_idx].split("/")[-1]
            folder_to_save = os.path.join(args.output_dir, folder)
            os.makedirs(folder_to_save, exist_ok=True)

            base_name = name.rsplit(".", 1)[0] + ".png"
            save_path = os.path.join(folder_to_save, base_name)
            torchvision.utils.save_image(final_image[path_idx], save_path)

if __name__ == "__main__":
    main()
