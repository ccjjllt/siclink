"""Differentiable, unregistered approximation to SiClink's contour objective."""

from functools import lru_cache

import torch
from torch.nn import functional as F


@lru_cache(maxsize=16)
def gaussian_kernels(sigma, device):
    radius = int(4 * sigma + 0.5)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    g = torch.exp(-0.5 * (coords / sigma).square())
    g /= g.sum()
    first = -coords * g / sigma**2
    second = ((coords / sigma).square() - 1) * g / sigma**2
    return tuple(kernel.reshape(1, 1, 1, -1) for kernel in (g, first, second))


def convolve_horizontal(images, kernel):
    return F.conv2d(images, kernel, padding=(0, kernel.shape[-1] // 2))


def convolve_vertical(images, kernel):
    return F.conv2d(images, kernel.transpose(-1, -2), padding=(kernel.shape[-1] // 2, 0))


def directional_fields(images, sigma):
    gaussian, first, second = gaussian_kernels(sigma, images.device)
    dxx = convolve_vertical(convolve_horizontal(images, second), gaussian)
    dyy = convolve_vertical(convolve_horizontal(images, gaussian), second)
    dxy = convolve_vertical(convolve_horizontal(images, first), first)
    middle = (dxx + dyy) * 0.5
    return torch.cat((dxx, middle + dxy, dyy, middle - dxy), dim=1)


def local_pool(images):
    gaussian = gaussian_kernels(2, images.device)[0]
    return convolve_vertical(convolve_horizontal(images, gaussian), gaussian)


def contour_loss(probability, target, *, sigmas=(1, 2, 4), hard_fraction=0.0):
    """Use soft or straight-through binary glyphs and match local directional energy.

    Targets and predictions have shape [N, 1, H, W]. Validation still uses the
    official scorer, which additionally registers and locally deforms the images.
    """
    probability = probability.float()
    target = target.float()
    with torch.autocast(device_type=probability.device.type, enabled=False):
        if hard_fraction:
            hard = (probability >= 0.5).to(probability.dtype)
            probability = probability + hard_fraction * (hard - probability).detach()
        shared_scales, total_scales = [], []
        for sigma in sigmas:
            pred = directional_fields(probability, sigma)
            truth = directional_fields(target, sigma)
            pred_energy = local_pool(pred.square().sum(dim=1, keepdim=True))
            true_energy = local_pool(truth.square().sum(dim=1, keepdim=True))
            correlation = local_pool((pred * truth).sum(dim=1, keepdim=True)).clamp_min(0)
            agreement = (correlation.square() / (pred_energy * true_energy).clamp_min(1e-10)).clamp_max(1)
            shared = torch.minimum(pred_energy, true_energy) * agreement
            union = pred_energy + true_energy - shared
            shared_scales.append(shared.flatten(1).sum(1))
            total_scales.append(union.flatten(1).sum(1))
        matched = torch.stack(shared_scales).sum(0)
        total = torch.stack(total_scales).sum(0).clamp_min(1e-8)
    return 1 - (matched / total).mean()
