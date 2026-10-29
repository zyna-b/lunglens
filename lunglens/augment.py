"""Data augmentation done on the GPU, a whole batch at once, different random values per image.

Kept deliberately mild for X-rays: small rotations, shifts and zooms (patients are never perfectly
centred), exposure and contrast changes (machines differ), a little noise and blur.
No horizontal flips: a flipped X-ray puts the heart on the wrong side, which never happens in real life.
"""

import math

import torch
import torch.nn.functional as F


def _u(n, lo, hi, device):
    return torch.empty(n, device=device).uniform_(lo, hi)


def random_affine(x, out_size, rot=7.0, zoom=(0.92, 1.12), aspect=(0.93, 1.07), shift=0.06):
    b, dev = x.shape[0], x.device
    a = _u(b, -rot, rot, dev) * math.pi / 180
    z = torch.exp(_u(b, math.log(zoom[0]), math.log(zoom[1]), dev))
    r = torch.exp(_u(b, math.log(aspect[0]), math.log(aspect[1]), dev))
    sx, sy = 1 / (z * r.sqrt()), r.sqrt() / z
    cos, sin = a.cos(), a.sin()
    theta = torch.stack([
        torch.stack([cos * sx, -sin * sy, _u(b, -shift, shift, dev)], 1),
        torch.stack([sin * sx, cos * sy, _u(b, -shift, shift, dev)], 1),
    ], 1)
    grid = F.affine_grid(theta, (b, x.shape[1], out_size, out_size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def photometric(x, brightness=0.15, contrast=0.2, gamma=0.25, noise=0.02, blur_p=0.2):
    b, dev = x.shape[0], x.device
    s = lambda v: (1 + _u(b, -v, v, dev)).view(b, 1, 1, 1)
    x = x.clamp(1e-4, 1) ** s(gamma)
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * s(contrast) + m) * s(brightness)
    if noise:
        x = x + torch.randn_like(x) * _u(b, 0, noise, dev).view(b, 1, 1, 1)
    if blur_p:
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=dev)
        k = (k[:, None] * k[None, :] / 256).expand(x.shape[1], 1, 5, 5)
        blurred = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), k, groups=x.shape[1])
        pick = (torch.rand(b, device=dev) < blur_p).view(b, 1, 1, 1)
        x = torch.where(pick, blurred, x)
    return x.clamp(0, 1)


def train_batch(u8, out_size):
    """u8: uint8 [B, H, W] on the GPU -> augmented float [B, 1, out, out] in [0, 1]."""
    x = u8.unsqueeze(1).float() / 255
    return photometric(random_affine(x, out_size))


def eval_batch(u8, out_size):
    x = u8.unsqueeze(1).float() / 255
    if x.shape[-1] != out_size:
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", antialias=True, align_corners=False)
    return x
