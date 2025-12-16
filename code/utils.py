import os
import json
import yaml
import hashlib
import base64
from typing import Dict, Any, List, Union
from omegaconf import OmegaConf
import wandb
from config_schema import MainConfig


def load_api_keys() -> Dict[str, str]:
    """Load API keys from the api_keys file.
    
    Returns:
        Dict[str, str]: Dictionary containing API keys for different models
        
    Raises:
        FileNotFoundError: If no api_keys file is found
    """
    for ext in ['yaml', 'yml', 'json']:
        file_path = f'api_keys.{ext}'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                if ext in ['yaml', 'yml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
    
    raise FileNotFoundError(
        "API keys file not found. Please create api_keys.yaml, api_keys.yml, or api_keys.json "
        "in the root directory with your API keys."
    )


def get_api_key(model_name: str) -> str:
    """Get API key for specified model.
    
    Args:
        model_name: Name of the model to get API key for
        
    Returns:
        str: API key for the specified model
        
    Raises:
        KeyError: If API key for model is not found
    """
    api_keys = load_api_keys()
    if model_name not in api_keys:
        raise KeyError(
            f"API key for {model_name} not found in api_keys file. "
            f"Available models: {list(api_keys.keys())}"
        )
    return api_keys[model_name]


def hash_training_config(cfg: MainConfig) -> str:
    """Create a deterministic hash of training-relevant config parameters.
    
    Args:
        cfg: Configuration object containing model settings
        
    Returns:
        str: MD5 hash of the config parameters
    """
    # Convert backbone list to plain Python list
    if isinstance(cfg.model.backbone, (list, tuple)):
        backbone = list(cfg.model.backbone)
    else:
        backbone = OmegaConf.to_container(cfg.model.backbone)
        
    # Create config dict with converted values
    train_config = {
        "data": {
            "batch_size": int(cfg.data.batch_size),
            "num_samples": int(cfg.data.num_samples),
            "cle_data_path": str(cfg.data.cle_data_path),
            "tgt_data_path": str(cfg.data.tgt_data_path),
        },
        "optim": {
            "alpha": float(cfg.optim.alpha),
            "epsilon": int(cfg.optim.epsilon),
            "steps": int(cfg.optim.steps),
        },
        "model": {
            "input_res": int(cfg.model.input_res),
            "use_source_crop": bool(cfg.model.use_source_crop),
            "use_target_crop": bool(cfg.model.use_target_crop),
            "crop_scale": tuple(float(x) for x in cfg.model.crop_scale),
            "ensemble": bool(cfg.model.ensemble),
            "backbone": backbone,
        },
        "attack": cfg.attack,
    }
    
    # Convert to JSON string with sorted keys
    json_str = json.dumps(train_config, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


def setup_wandb(cfg: MainConfig, tags=None) -> None:
    """Initialize Weights & Biases logging.
    
    Args:
        cfg: Configuration object containing wandb settings
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,
        config=config_dict,
        tags=tags,
    )


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)


def get_output_paths(cfg: MainConfig, config_hash: str) -> Dict[str, str]:
    """Get dictionary of output paths based on config.
    
    Args:
        cfg: Configuration object
        config_hash: Hash of training config
        
    Returns:
        Dict[str, str]: Dictionary containing output paths
    """
    return {
        'output_dir': os.path.join(cfg.data.output, "img", config_hash),
        'desc_output_dir': os.path.join(cfg.data.output, "description", config_hash)
    } 


import torch

def apply_patch(image, patch):
    img = image.clone()
    _, _, H, W = img.shape
    ph, pw = patch.shape[2:]
    top = (H - ph) // 2
    left = (W - pw) // 2
    img[:, :, top:top+ph, left:left+pw] = patch
    return img

def apply_patch_with_center(image, patch, center):
    """
    Apply a patch to an image at a specified normalized center location.

    Args:
        image (Tensor): Input image tensor of shape (N, C, H, W).
        patch (Tensor): Patch tensor of shape (N, C, ph, pw).
        center (Tensor): Normalized coordinates tensor of shape (N, 2), 
                        each coordinate in range [-1, 1], format [x, y].

    Returns:
        Tensor: Image with the patch applied.
    """
    img = image.clone()
    N, C, H, W = img.shape
    ph, pw = patch.shape[2:]

    assert img.shape[0] == patch.shape[0] == center.shape[0], "Batch sizes must match"
    assert center.shape[1] == 2, "Center should be a tensor of shape (N, 2)"
    
    for i in range(N):
        cx, cy = center[i]  
    
        cx_pixel = ((cx + 1) / 2) * W
        cy_pixel = ((cy + 1) / 2) * H
        
        left = int(torch.round(cx_pixel - pw / 2))
        top = int(torch.round(cy_pixel - ph / 2))
        
        if left < 0:
            left = 0
        if top < 0:
            top = 0
        if left + pw > W:
            left = W - pw
        if top + ph > H:
            top = H - ph
        img[i, :, top:top+ph, left:left+pw] = patch[i]
    
    return img

from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())  # 例如 float32

def load_patch(path, size, device='cuda'):
    patch = Image.open(path).convert("RGB")
    patch = to_tensor(patch).to(device) 
    patch = patch.unsqueeze(0)  # [1, 3, H, W]
    patch = F.interpolate(patch, size=(size, size), mode='bilinear', align_corners=False)
    return patch



import torch
import torchvision.transforms.functional as TF
import random
import math
from typing import Tuple, List, Optional
import torch
import random
from PIL import Image

class RandomPointConstrainedCrop:
    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.5, 0.9),
        ratio: Tuple[float, float] = (0.5, 2.0),
        norm_coord: Tuple[float, float] = (0.0, 0.0),
        attempts: int = 10,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.norm_coord = norm_coord
        self.attempts = attempts

    # -------- private helpers ------------------------------------------------
    @staticmethod
    def _clamp(val: int, low: int, high: int) -> int:
        return max(low, min(val, high))

    def _point_to_pixel(self, H: int, W: int) -> Tuple[int, int]:
        x = (self.norm_coord[0] + 1) * 0.5 * W
        y = (self.norm_coord[1] + 1) * 0.5 * H
        return int(round(x)), int(round(y))

    # -------------------------------------------------------------------------
    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        if img.dim() == 4:
            batch_mode = True
            _, _, H, W = img.shape
        elif img.dim() == 3:
            batch_mode = False
            _, H, W = img.shape
        else:
            raise ValueError("Unsupported tensor shape; expect 3D or 4D.")

        x_p, y_p = self._point_to_pixel(H, W)
        total_area = H * W

        for _ in range(self.attempts):
            q = random.uniform(*self.scale)
            target_area = q * total_area

            log_ar = random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1]))
            a = math.exp(log_ar)

            h = int(round(math.sqrt(target_area / a)))
            w = int(round(a * h))

            if h > H or w > W or h == 0 or w == 0:
                continue

            x_left_min = max(0, x_p - w)
            x_left_max = min(x_p, W - w)
            y_top_min = max(0, y_p - h)
            y_top_max = min(y_p, H - h)

            if x_left_max < x_left_min or y_top_max < y_top_min:
                continue

            x_left = random.randint(x_left_min, x_left_max)
            y_top = random.randint(y_top_min, y_top_max)

            return TF.resized_crop(img, y_top, x_left, h, w, self.size)

        side = min(H, W)
        return TF.resize(TF.center_crop(img, side), self.size)

class RandomResizedCropWithCoord:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3/4, 4/3), norm_coord=(0.0, 0.0)):
        self.size = size  # e.g. (900, 1600)
        self.scale = scale
        self.ratio = ratio
        self.norm_coord = norm_coord[0]  # (x_norm, y_norm), in [-1, 1]

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of shape (1, C, H, W)
        Returns:
            Tensor: Transformed image
        """

        _, _, H, W = img.shape

        # Convert normalized coords [-1, 1] to absolute pixel coords
        x_abs = int((self.norm_coord[0] + 1) / 2 * W) + random.randint(-50, 50)
        y_abs = int((self.norm_coord[1] + 1) / 2 * H) + random.randint(-50, 50)

        for _ in range(10):  # Try 10 times to find a valid crop
            target_area = random.uniform(*self.scale) * H * W
            aspect_ratio = random.uniform(*self.ratio)

            crop_w = int(round((target_area * aspect_ratio) ** 0.5))
            crop_h = int(round((target_area / aspect_ratio) ** 0.5))

            if crop_w <= W and crop_h <= H:
                left_min = max(0, x_abs - crop_w + 1)
                left_max = min(x_abs, W - crop_w)
                top_min = max(0, y_abs - crop_h + 1)
                top_max = min(y_abs, H - crop_h)

                if left_max >= left_min and top_max >= top_min:
                    left = random.randint(left_min, left_max)
                    top = random.randint(top_min, top_max)

                    # crop and resize
                    cropped = TF.resized_crop(img, top, left, crop_h, crop_w, self.size)
                    return cropped

        # Fallback: center crop and resize
        fallback = TF.resize(TF.center_crop(img, min(H, W)), self.size)
        return fallback


class RandomResizedCropWithPatch(torch.nn.Module):
    
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3/4, 4/3), 
                 interpolation=TF.InterpolationMode.BILINEAR, max_attempts=10):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
            
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.max_attempts = max_attempts
    
    def _normalized_to_pixel_coords(self, norm_coords: torch.Tensor, height: int, width: int) -> torch.Tensor:

        y_pixels = ((norm_coords[:, 0] + 1) / 2) * (height - 1)
        x_pixels = ((norm_coords[:, 1] + 1) / 2) * (width - 1)
        
        return torch.stack([y_pixels, x_pixels], dim=1).round().long()
    
    def get_patch_coords(self, img_shape: Tuple[int, int, int, int], 
                         patch_shape: Tuple[int, int, int, int], 
                         patch_center: torch.Tensor) -> List[Tuple[int, int, int, int]]:

        batch_size, _, img_height, img_width = img_shape
        _, _, patch_height, patch_width = patch_shape
        
        pixel_centers = self._normalized_to_pixel_coords(patch_center, img_height, img_width)
        
        patch_coords_list = []
        for b in range(batch_size):
            center_y, center_x = pixel_centers[b]
            
            top = max(0, min(center_y - patch_height // 2, img_height - patch_height))
            left = max(0, min(center_x - patch_width // 2, img_width - patch_width))
            
            patch_coords_list.append((top.item(), left.item(), patch_height, patch_width))
            
        return patch_coords_list
    
    def get_params(self, img: torch.Tensor, patch: torch.Tensor, patch_coords: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:

        _, _, img_height, img_width = img.shape
        patch_top, patch_left, patch_height, patch_width = patch_coords
        
        patch_bottom = patch_top + patch_height
        patch_right = patch_left + patch_width
        
        area = img_height * img_width
        
        for _ in range(self.max_attempts):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if 0 < w <= img_width and 0 < h <= img_height:
                i = random.randint(0, img_height - h)
                j = random.randint(0, img_width - w)
                
                intersect_top = max(i, patch_top)
                intersect_left = max(j, patch_left)
                intersect_bottom = min(i + h, patch_bottom)
                intersect_right = min(j + w, patch_right)
                
                if intersect_top < intersect_bottom and intersect_left < intersect_right:
                    return i, j, h, w
        
        center_y = patch_top + patch_height // 2
        center_x = patch_left + patch_width // 2
        
        for scale_factor in [0.5, 0.8, 1.0, 1.2, 1.5]:
            h = min(int(img_height * scale_factor), img_height)
            w = min(int(img_width * scale_factor), img_width)
            
            i = max(0, min(center_y - h // 2, img_height - h))
            j = max(0, min(center_x - w // 2, img_width - w))
            
            intersect_top = max(i, patch_top)
            intersect_left = max(j, patch_left)
            intersect_bottom = min(i + h, patch_bottom)
            intersect_right = min(j + w, patch_right)
            
            if intersect_top < intersect_bottom and intersect_left < intersect_right:
                return i, j, h, w
        
        return patch_top, patch_left, patch_height, patch_width
    
    def forward(self, img: torch.Tensor, patch: torch.Tensor, patch_center: torch.Tensor):

        if not torch.is_tensor(img) or not torch.is_tensor(patch) or not torch.is_tensor(patch_center):
            raise TypeError("img, patch, and patch_center should be Tensor")
        
        if patch_center.shape[0] != img.shape[0] or patch_center.shape[1] != 2:
            raise ValueError(f"Expected patch_center shape [B, 2], got {patch_center.shape}")
        
        if torch.any(patch_center < -1) or torch.any(patch_center > 1):
            raise ValueError("patch_center coordinates must be in range [-1, 1]")
        
        batch_size = img.shape[0]
        
        patch_coords_list = self.get_patch_coords(img.shape, patch.shape, patch_center)
        
        result = []
        
        for b in range(batch_size):
            i, j, h, w = self.get_params(img[b:b+1], patch[b:b+1], patch_coords_list[b])
            crop = TF.resized_crop(img[b], i, j, h, w, self.size, self.interpolation)
            result.append(crop)
        
        return torch.stack(result)


