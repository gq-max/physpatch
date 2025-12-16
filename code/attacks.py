import torch
import torch.nn.functional as F
import clip
from utils import *
from dyn_mask import *
import numpy as np
import torch_dct as dct
import scipy.stats as st
device = "cuda" if torch.cuda.is_available() else "cpu"
from torch.autograd import Variable
import os

def set_environment(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tv_loss(patch: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Total Variation Loss
    """
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)

    dy = patch[:, :, 1:, :] - patch[:, :, :-1, :]
    dx = patch[:, :, :, 1:] - patch[:, :, :, :-1]

    if reduction == "mean":
        return (dy.abs().mean() + dx.abs().mean())
    elif reduction == "sum":
        return (dy.abs().sum() + dx.abs().sum())
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

palette = torch.tensor([
    [0,   0,   0  ],
    [255, 255, 255],
    [255, 0,   0  ],
    [0,   255, 0  ],
    [0,   0,   255],
    [255, 255, 0  ],
    [255, 0,   255],
    [0,   255, 255],
    [128, 128, 128], # Grey
    [220, 130, 0  ], # Orangish
    [160, 105, 55 ], # Brownish
    [200, 175, 30 ], # Goldish
    [220, 210, 50 ]  # Yellowish
], dtype=torch.float32)  

# === NPS Loss ===
def nps_loss(patch: torch.Tensor, palette: torch.Tensor = palette) -> torch.Tensor:
    """
    Non-Printability Score (NPS) Loss
    """
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)  # (1, 3, H, W)
    B, C, H, W = patch.shape


    patch_flat = patch.permute(0, 2, 3, 1).reshape(B, -1, 3)


    palette = palette.to(patch.device).unsqueeze(0).expand(B, -1, -1)  # (B, K, 3)


    dist = torch.cdist(patch_flat, palette)

    return dist.min(dim=-1).values.mean()


def pgd(
    image_tensor, tgt_tensor, ensemble_extractor, ensemble_loss,
    source_crop, target_crop, center=None,
    num_iters=10, epsilon=8, alpha=1.0, decay=1.0, device='cuda'
):
    set_environment(42)
    print("patch:", torch.min(init_patch), torch.max(init_patch), init_patch.shape)
    print("ori:", torch.min(image_tensor), torch.max(image_tensor), image_tensor.shape)
    print("tgt:", torch.min(tgt_tensor), torch.max(tgt_tensor), tgt_tensor.shape)
    init_patch = image_tensor.clone().detach().to(device)

    delta = torch.zeros_like(init_patch, requires_grad=True).to(device) # 优化

    patch_generator = DynamicPatchGenerator()
    threshold_list = [round(0.6 + i * 0.02, 6) for i in range(num_iters)]
    mask_expanded = torch.ones([1, 1, 900, 1600]).to(device)
    trans = My_T(num_copies=1)
    th = 120 * 120
    for iter in range(num_iters):
        with torch.no_grad():
            tgt = target_crop(tgt_tensor)
            ensemble_loss.set_ground_truth(tgt)
            ensemble_loss.set_full_feature(tgt_tensor)

            
        adv_patch = init_patch + delta
        if center is not None:
        
            if mask_expanded.sum()  > th:
                mask_expanded = patch_generator(center, threshold_list[iter])
            adv_image = image_tensor * (1 - mask_expanded) + adv_patch * mask_expanded
        else:
            adv_image = apply_patch(image=image_tensor, patch=adv_patch)

        local = torch.cat([trans.transform(source_crop(adv_image)) for _ in range(1)])
        features, features_local = ensemble_extractor(local)
        full = ensemble_extractor.encode_img(adv_image)
        loss_a = ensemble_loss(features, features_local, full)
        loss += loss_a

            
        if mask_expanded.sum()  > th:
            grads = torch.autograd.grad(loss, [delta, mask_expanded], create_graph=False)
            grad = grads[0]
            grad_mask = grads[1]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy(), "mask_grads:", grad_mask.mean().detach().cpu().numpy())
        else:
            grads = torch.autograd.grad(loss, [delta], create_graph=False)
            grad = grads[0]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy())
        
        delta.data = delta + alpha * torch.sign(grad)
        delta.data = torch.clamp(delta.data, min=-epsilon, max=epsilon)
        if mask_expanded.sum()  > th:
            with torch.no_grad():
                patch_generator.update_potential_field(grad_mask, iter)
        
    final_patch = init_patch + delta
    final_image = image_tensor * (1 - mask_expanded) + final_patch * mask_expanded
    final_image = torch.clamp(final_image/255.0, 0.0, 1.0)
    return final_image.detach(), final_patch

def mifgsm(
    image_tensor, tgt_tensor, ensemble_extractor, ensemble_loss,
    source_crop, target_crop, center=None,
    num_iters=10, epsilon=8, alpha=1.0, decay=1.0, device='cuda'
):
    set_environment(42)
    print("patch:", torch.min(init_patch), torch.max(init_patch), init_patch.shape)
    print("ori:", torch.min(image_tensor), torch.max(image_tensor), image_tensor.shape)
    print("tgt:", torch.min(tgt_tensor), torch.max(tgt_tensor), tgt_tensor.shape)
    init_patch = image_tensor.clone().detach().to(device)
    momentum = torch.zeros_like(init_patch).to(device)
    delta = torch.zeros_like(init_patch, requires_grad=True).to(device) # 优化
    # adv_patch = init_patch.clone().detach().requires_grad_(True)
    patch_generator = DynamicPatchGenerator()
    threshold_list = [round(0.6 + i * 0.02, 6) for i in range(num_iters)]
    mask_expanded = torch.ones([1, 1, 900, 1600]).to(device)
    trans = My_T(num_copies=1)
    th = 120 * 120
    for iter in range(num_iters):
        with torch.no_grad():
            tgt = target_crop(tgt_tensor)
            ensemble_loss.set_ground_truth(tgt)
            ensemble_loss.set_full_feature(tgt_tensor)

            
        adv_patch = init_patch + delta
        if center is not None:
        
            if mask_expanded.sum()  > th:
                mask_expanded = patch_generator(center, threshold_list[iter])
            adv_image = image_tensor * (1 - mask_expanded) + adv_patch * mask_expanded
        else:
            adv_image = apply_patch(image=image_tensor, patch=adv_patch)

        local = torch.cat([trans.transform(source_crop(adv_image)) for _ in range(1)])
        features, features_local = ensemble_extractor(local)
        full = ensemble_extractor.encode_img(adv_image)
        loss_a = ensemble_loss(features, features_local, full)
        loss += loss_a

            
        if mask_expanded.sum()  > th:
            grads = torch.autograd.grad(loss, [delta, mask_expanded], create_graph=False)
            grad = grads[0]
            grad_mask = grads[1]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy(), "mask_grads:", grad_mask.mean().detach().cpu().numpy())
        else:
            grads = torch.autograd.grad(loss, [delta], create_graph=False)
            grad = grads[0]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy())
        
        momentum = decay * momentum + grad
        delta.data = torch.clamp(
            delta + alpha * torch.sign(momentum),
            min=-epsilon,
            max=epsilon,
        )
        if mask_expanded.sum()  > th:
            with torch.no_grad():
                patch_generator.update_potential_field(grad_mask, iter)
        
    final_patch = init_patch + delta
    final_image = image_tensor * (1 - mask_expanded) + final_patch * mask_expanded
    final_image = torch.clamp(final_image/255.0, 0.0, 1.0)
    return final_image.detach(), final_patch


class My_T:
    def __init__(self, num_copies=5, **kwargs):
        self.num_copies = num_copies
        self.kernel = self.gkern()
        # self.op = [self.resize, self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180, self.scale, self.add_noise,self.dct,self.drop_out]
        self.op = [self.resize, self.vertical_shift, self.horizontal_shift, self.vertical_flip, 
           self.horizontal_flip, self.rotate180, self.scale, self.add_noise, self.dct, 
           self.drop_out, self.adjust_brightness, self.adjust_contrast]

    
    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w // 10, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low = 0, high=h // 10, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2,3))
    
    def scale(self, x):
        return torch.rand(1)[0] * x
    
    def resize(self, x):
        """
        Resize the input
        """
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor)+1
        new_w = int(w * scale_factor)+1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 255)
        return x
    
    def dct(self, x):
        """
        Discrete Fourier Transform
        """
        dctx = dct.dct_2d(x) #torch.fft.fft2(x, dim=(-2, -1))
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # dctx[:, :, -low_w:, -low_h:] = 0
        dctx[:, :, -low_w:,:] = 0
        dctx[:, :, :, -low_h:] = 0
        dctx = dctx # * self.mask.reshape(1, 1, w, h)
        idctx = dct.idct_2d(dctx)
        return idctx
    
    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16,16), 0, 255)

    def gkern(self, kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(device)
    
    def adjust_brightness(self, x, factor=None):
        """
        Brightness adjustment by scaling all pixels.
        factor > 1: brighter, factor < 1: darker
        """
        if factor is None:
            factor = np.random.uniform(0.9, 1.1)
        return (x * factor).clamp(0, 255)

    def adjust_contrast(self, x, factor=None):
        """
        Contrast adjustment by changing the difference from mean.
        factor > 1: more contrast, factor < 1: less contrast
        """
        if factor is None:
            factor = np.random.uniform(0.8, 1.2)
        mean = x.mean(dim=(2, 3), keepdim=True)
        return ((x - mean) * factor + mean).clamp(0, 255)

    def drop_out(self, x):
        
        return F.dropout2d(x, p=0.1, training=True)
    
    def blocktransform(self, x):
        i = torch.randint(0, len(self.op), [1]).item()
        trans = self.op[i]
        return trans(x)
    
    def transform(self, x, **kwargs):
        """
        Scale the input for BlockShuffle
        """
        
        return torch.cat([self.blocktransform(x) for _ in range(self.num_copies)])
