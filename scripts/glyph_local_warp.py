"""Invert frozen spike encoding within a small smooth glyph deformation."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


KNOTS = 6
MAX_SHIFT_PIXELS = 1.25
MAX_INK_FRACTION = 0.02
MAX_CENTROID_SHIFT = 0.5
GRID_SIZE = 64
EPOCHS = 24
LEARNING_RATE = 0.16
BATCH_SIZE = 32
RENDER_SCALES = (1.0, 0.75, 0.50, 0.25, 0.0)


@dataclass
class LocalWarp:
    images: torch.Tensor
    displacement: torch.Tensor
    retained_scale: torch.Tensor


def encode_evoked(encoder, pixels, session):
    drive = encoder.visual_drive(encoder.image_features(pixels))
    rates = encoder._expected(drive, session)
    return rates.unflatten(-1, (encoder.n_units, encoder.n_time_bins))[..., 2:].flatten(1)


def observed_evoked(counts):
    return counts.mean(dim=1).unflatten(-1, (-1, 10))[..., 2:].flatten(1)


def poisson_log_likelihood(rates, observed):
    return (rates.log() * observed - rates).sum(dim=1)


@torch.no_grad()
def poisson_scores(encoder, masks, counts, session):
    observed = observed_evoked(counts)
    return torch.cat([poisson_log_likelihood(encode_evoked(encoder, batch[:, None].float(), session), response)
                      for batch, response in zip(masks.split(BATCH_SIZE), observed.split(BATCH_SIZE))])


def landmarks(images, xy):
    mass = images.sum(dim=(-1, -2))
    centroid = (images[:, :, :, None] * xy).sum(dim=(1, 2)) / mass[:, None].clamp_min(1)
    return mass, centroid


def flow_field(raw):
    # Remove constant translation and fix the outer knots to the original image frame.
    knots = raw.tanh()
    knots = F.pad(knots - knots.mean(dim=(2, 3), keepdim=True), (1, 1, 1, 1))
    dense = F.interpolate(knots, size=(GRID_SIZE, GRID_SIZE), mode="bilinear", align_corners=True)
    magnitude = dense.norm(dim=1, keepdim=True).amax(dim=(2, 3), keepdim=True).clamp_min(1)
    return MAX_SHIFT_PIXELS * dense / magnitude


def warp(batch, displacement, grid):
    offsets = displacement.permute(0, 2, 3, 1) * (2.0 / (GRID_SIZE - 1))
    return F.grid_sample(batch, grid[None] + offsets, mode="bilinear", padding_mode="zeros", align_corners=True)


def regularize_limits(images, xy, ink, center):
    mass, centroid = landmarks(images, xy)
    ink_distance = (mass - ink).abs() / (MAX_INK_FRACTION * ink).clamp_min(1)
    center_distance_squared = (centroid - center).square().sum(dim=1) / MAX_CENTROID_SHIFT**2
    return (ink_distance - 1).clamp_min(0).square() + (center_distance_squared - 1).clamp_min(0).square()


@torch.no_grad()
def constrained_binary(batch, field, xy, grid, ink, center):
    """Choose the first geometrically admissible binary warp from a fixed scale schedule."""
    result = batch[:, 0].bool()
    selected = field.new_zeros(len(field))
    accepted = torch.zeros(len(field), dtype=torch.bool, device=field.device)
    for scale in RENDER_SCALES:
        warped = warp(batch, scale * field, grid)[:, 0] >= 0.5
        mass, centroid = landmarks(warped.float(), xy)
        feasible = ((mass - ink).abs() <= (MAX_INK_FRACTION * ink).clamp_min(1)
                    ) & ((centroid - center).norm(dim=1) <= MAX_CENTROID_SHIFT)
        use = feasible & ~accepted
        result = torch.where(use[:, None, None], warped, result)
        selected[use] = scale
        accepted |= use
    return LocalWarp(result, selected[:, None, None, None] * field, selected)


def invert_local(encoder, masks, counts, session, *, score=poisson_log_likelihood, render=constrained_binary):
    if masks.ndim != 3 or masks.shape[-2:] != (GRID_SIZE, GRID_SIZE) or len(masks) != len(counts):
        raise ValueError("Provide paired binary 64 by 64 templates and grouped responses.")
    if encoder.n_time_bins != 10:
        raise ValueError("This inversion requires the existing 25 ms count encoder.")
    coordinate = torch.arange(GRID_SIZE, device=masks.device, dtype=torch.float32)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    xy = torch.stack((x, y), dim=-1)
    grid = (2.0 / (GRID_SIZE - 1)) * xy - 1
    observed = observed_evoked(counts)
    reports = []
    for mask, response in zip(masks.split(BATCH_SIZE), observed.split(BATCH_SIZE)):
        batch = mask[:, None].float()
        ink, center = landmarks(batch[:, 0], xy)
        raw = batch.new_zeros((len(batch), 2, KNOTS - 2, KNOTS - 2), requires_grad=True)
        optimizer = torch.optim.Adam((raw,), lr=LEARNING_RATE)
        for _ in range(EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            field = flow_field(raw)
            soft = warp(batch, field, grid)
            likelihood = score(encode_evoked(encoder, soft, session), response)
            bound = regularize_limits(soft[:, 0], xy, ink, center)
            change = field.square().mean(dim=(1, 2, 3))
            roughness = raw.diff(dim=2).square().mean() + raw.diff(dim=3).square().mean()
            loss = (-likelihood + 0.35 * change + 1.5 * bound).mean() + 0.30 * roughness
            raw.grad, = torch.autograd.grad(loss, raw)
            optimizer.step()
        with torch.no_grad():
            reports.append(render(batch, flow_field(raw), xy, grid, ink, center))
    return LocalWarp(torch.cat([item.images for item in reports]), torch.cat([item.displacement for item in reports]),
                     torch.cat([item.retained_scale for item in reports]))


@torch.no_grad()
def warp_statistics(initial, result):
    changed = (initial != result.images).flatten(1).sum(1).float()
    coordinate = torch.arange(GRID_SIZE, device=initial.device, dtype=torch.float32)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    xy = torch.stack((x, y), dim=-1)
    ink, center = landmarks(initial.float(), xy)
    final_ink, final_center = landmarks(result.images.float(), xy)
    position = (final_center - center).norm(dim=1)
    amount = (final_ink - ink).abs() / ink.clamp_min(1)
    displacement = result.displacement.norm(dim=1)
    return {"changed_images": int((changed > 0).sum()), "mean_changed_pixels": float(changed.mean()),
            "mean_absolute_ink_fraction_change": float(amount.mean()), "max_absolute_ink_fraction_change": float(amount.max()),
            "mean_centroid_shift_px": float(position.mean()), "max_centroid_shift_px": float(position.max()),
            "mean_peak_applied_displacement_px": float(displacement.flatten(1).amax(1).mean()),
            "max_applied_displacement_px": float(displacement.max()), "mean_retained_flow_scale": float(result.retained_scale.mean())}
