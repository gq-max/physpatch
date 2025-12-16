import cv2
import kornia
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF
from scipy import ndimage
import torchvision.transforms as transforms
from constants import Const, N
import kornia.morphology as morph

import torch
import json
import numpy as np
from scipy import ndimage

def remove_small_regions(mask_np, min_area=500):
    labeled_mask, num_features = ndimage.label(mask_np)
    sizes = ndimage.sum(mask_np, labeled_mask, range(1, num_features + 1))
    output_mask = np.zeros_like(mask_np)

    for i, size in enumerate(sizes):
        if size >= min_area:
            output_mask[labeled_mask == (i + 1)] = 1

    return output_mask


def save_tensor_to_json(tensor: torch.Tensor, filename: str) -> None:
    tensor_list = tensor.detach().cpu().tolist()
    
    with open(filename, 'w') as f:
        json.dump(tensor_list, f)
import torch
from torchvision.transforms.functional import to_pil_image

def save_tensor_image(tensor: torch.Tensor, filename: str) -> None:
    if tensor.is_cuda:
        tensor = tensor.cpu()  

    if tensor.dim() == 4:
        tensor = tensor[0]

    image = to_pil_image(tensor)

    image.save(filename, format='JPEG')


class DynamicPatchGenerator(nn.Module):
    def __init__(self, image_size=(1600, 900), initial_threshold=0.7, learning_rate=0.15, device="cuda"):

        super(DynamicPatchGenerator, self).__init__()
        
        self.width, self.height = image_size
        self.initial_threshold = initial_threshold
        self.learning_rate = learning_rate
        
        self.potential_field = None
        self.device = device
        
        
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, self.height),
            torch.linspace(-1, 1, self.width),
            indexing='ij'
        )
        self.register_buffer('coord_grid', torch.stack([x_grid, y_grid], dim=0).unsqueeze(0))
    
    def initialize_gaussian_field(self, coord, sigma=0.2):
        batch_size = coord.shape[0]
        
        coord = coord.view(batch_size, 2, 1, 1)
        
        distance = torch.sum((self.coord_grid - coord) ** 2, dim=1, keepdim=True)
        
        gaussian_field = torch.exp(-distance / (2 * sigma ** 2))
        
        self.potential_field = gaussian_field.mean(dim=0, keepdim=True).to(self.device)
        
        return gaussian_field
    
    def get_mask(self, threshold=None):
        if threshold is None:
            threshold = self.initial_threshold
            
        soft_mask = torch.sigmoid(self.potential_field)
        t_m = (soft_mask[0] > threshold).float().cpu().numpy().astype(np.uint8)
        t_m = ndimage.binary_fill_holes(t_m).astype(np.uint8)
        t_m = torch.tensor(t_m, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        struct_elem = torch.ones((11, 11), device=self.device)
        t_m = morph.closing(t_m[:, 0, :, :], struct_elem)
        t_m = t_m.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        t_m = ndimage.gaussian_filter(t_m, sigma=3)
        t_m = (t_m > threshold).astype(np.float32)
        t_m = remove_small_regions(t_m, min_area=800)
        t_m = ndimage.binary_fill_holes(t_m).astype(np.uint8)
        t_m = ndimage.gaussian_filter(t_m, sigma=3)
        t_m = torch.tensor(t_m, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        return t_m

    
    
    def update_potential_field(self, mask_grads, iter):

        abs_grads = torch.abs(mask_grads) * self.mask_expanded
        smoothed_grads = TF.gaussian_blur(abs_grads, kernel_size=3, sigma=1)

        if smoothed_grads.max() > 0:
            norm_grads = (smoothed_grads - smoothed_grads.mean()) / (smoothed_grads.max() + 1e-8)
        else:
            norm_grads = smoothed_grads
        norm_grads = torch.maximum(norm_grads, torch.tensor(0.0).to(self.device))

        self.potential_field = self.potential_field + self.learning_rate * norm_grads
        self.potential_field = TF.gaussian_blur(self.potential_field, kernel_size=3, sigma=1)
    
    def forward(self, coord=(0.5, 0.5), threshold=None):
        if self.potential_field is None:
            self.initialize_gaussian_field(coord)

        binary_mask = self.get_mask(threshold)
        self.mask_expanded = binary_mask.clone().requires_grad_(True)
        return self.mask_expanded
    
