"""Fit one bounded rendering correction shared by all public glyphs."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from ocf_training import contour_loss


@dataclass(frozen=True)
class FontStyle:
    scale_x: float
    scale_y: float
    shift_x_pixels: float
    shift_y_pixels: float
    rotation_degrees: float
    stroke_adjustment: float

    def apply(self, alpha):
        """Apply a shared style to [N,1,64,64] antialiased masks in [0,1]."""
        parameters = alpha.new_tensor((self.scale_x, self.scale_y, self.shift_x_pixels, self.shift_y_pixels,
                                       self.rotation_degrees, self.stroke_adjustment))
        return _apply(alpha, parameters)


def _apply(alpha, parameters):
    sx, sy, tx, ty, angle, stroke = parameters.unbind()
    tx, ty = tx * 2 / alpha.shape[-1], ty * 2 / alpha.shape[-2]
    radians = angle * math.pi / 180
    cosine, sine = radians.cos(), radians.sin()
    # The affine rasterizer expects the inverse geometry, in canvas coordinates.
    inverse = torch.stack((cosine / sx, sine / sx, -(cosine * tx + sine * ty) / sx,
                           -sine / sy, cosine / sy, (sine * tx - cosine * ty) / sy)).reshape(1, 2, 3)
    grid = F.affine_grid(inverse, (1, *alpha.shape[1:]), align_corners=False)
    aligned = F.grid_sample(alpha, grid.expand(len(alpha), -1, -1, -1), align_corners=False)

    # Interpolate with a one-pixel widening or thinning so the learned amount
    # can move antialiased edges by less than one pixel.
    dilated = F.max_pool2d(aligned, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-aligned, 3, stride=1, padding=1)
    return aligned + stroke.clamp_min(0) * (dilated - aligned) + (-stroke).clamp_min(0) * (eroded - aligned)


def _dice(source, target):
    overlap = (source * target).flatten(1).sum(dim=1)
    mass = (source + target).flatten(1).sum(dim=1)
    return (2 * overlap + 1) / (mass + 1)


def fit_font_style(source, reference, confidence, *, steps=100, batch_size=32, seed=211):
    """Fit six shared parameters from automatically matched fit-image pairs.

    ``source`` is [N,1,64,64] public antialiased alpha, ``reference`` is
    the paired binary training masks, and ``confidence`` weights fit pairs.
    """
    if (source.ndim != 4 or source.shape[1:] != (1, 64, 64) or source.shape != reference.shape
            or confidence.shape != (len(source),) or not len(source) or min(steps, batch_size) < 1):
        raise ValueError("Provide matching fit glyphs, pair confidence and positive optimization settings.")
    source = source.float().detach()
    reference = reference.to(source).detach()
    confidence = confidence.to(source).detach().clamp(0, 1)
    device = source.device
    rng = torch.Generator(device=device).manual_seed(seed)
    origin = source.new_tensor((1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    limits = source.new_tensor((0.09, 0.09, 2.5, 2.5, 3.0, 0.85))
    adjustment = torch.nn.Parameter(torch.zeros_like(limits))
    optimizer = torch.optim.Adam([adjustment], lr=0.035)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.005)
    n_pairs = min(batch_size, len(source))
    for _ in range(steps):
        selected = torch.randperm(len(source), generator=rng, device=device)[:n_pairs]
        optimizer.zero_grad(set_to_none=True)
        unit_adjustment = adjustment.tanh()
        predicted = _apply(source[selected], origin + limits * unit_adjustment)
        targets = reference[selected]
        pixel = (confidence[selected] * (1 - _dice(predicted, targets))).mean()
        contour = contour_loss(predicted, targets, sigmas=(1, 2), hard_fraction=0.2)
        loss = pixel + 0.18 * contour + 0.018 * unit_adjustment.square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        values = origin + limits * adjustment.tanh()
        style = FontStyle(*values.tolist())
        improved = style.apply(source)
        before = _dice((source >= 0.5).float(), reference)
        after = _dice((improved >= 0.5).float(), reference)
        fit_summary = {"pairs": len(source), "mean_confidence": confidence.mean().item(),
                       "before_fit_dice": before.mean().item(), "after_fit_dice": after.mean().item()}
    return style, fit_summary


def fit_aligned_font_style(public_images, training_images, *, seed=211, pair_count=512):
    """Use a fixed subset of already aligned training pairs for output calibration."""
    import numpy as np

    rows = np.random.default_rng(seed + 2).choice(len(training_images), min(pair_count, len(training_images)),
                                                  replace=False)
    selected = torch.as_tensor(rows, device=training_images.device)
    source = public_images[selected, None].float() / 255
    reference = (training_images[selected, None] >= 128).float()
    confidence = ((_dice(source >= 0.5, reference) - 0.55) / 0.075).sigmoid()
    return fit_font_style(source, reference, confidence, seed=seed)
