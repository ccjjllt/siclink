"""Correct the boundaries of public binary glyphs with one shared small model."""

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F

from ocf_training import contour_loss


MAX_DISPLACEMENT = 1.20
MAX_EDIT_FRACTION = 0.08
TEMPERATURE = 0.35


def edge_geometry(masks):
    """Use positive half-pixel signed distance on the ink side of an edge."""
    if masks.ndim != 3 or masks.shape[1:] != (64, 64) or masks.dtype != torch.bool:
        raise ValueError("Expected binary [glyphs, 64, 64] masks.")
    distance = np.empty(tuple(masks.shape), dtype=np.float32)
    for index, mask in enumerate(masks.detach().cpu().numpy()):
        field = distance_transform_edt(mask) - distance_transform_edt(~mask)
        distance[index] = np.clip(field - 0.5 * np.sign(field), -3, 3)
    signed = torch.from_numpy(distance).to(masks.device)[:, None]
    initial = masks.float()[:, None]
    density = F.avg_pool2d(initial, 3, stride=1, padding=1)
    dx = F.pad(density[..., 2:] - density[..., :-2], (1, 1, 0, 0))
    dy = F.pad(density[..., 2:, :] - density[..., :-2, :], (0, 0, 1, 1))
    return signed, torch.cat((initial, signed / 3, density, dx, dy), dim=1)


class LocalFontStyle(nn.Module):
    """Translation-equivariant normal displacement limited to the original edge."""

    def __init__(self):
        super().__init__()
        self.mapping = nn.Sequential(nn.Conv2d(5, 12, 5, padding=2), nn.SiLU(),
                                     nn.Conv2d(12, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 1, 1))
        nn.init.zeros_(self.mapping[-1].weight)
        nn.init.zeros_(self.mapping[-1].bias)

    def forward(self, signed, features):
        edge = signed.abs() <= MAX_DISPLACEMENT
        correction = MAX_DISPLACEMENT * self.mapping(features).tanh() * edge
        return signed + correction, correction


def fit_local_font_style(source, target, quality, *, seed=211, epochs=8):
    """Fit only from paired original FontStyle outputs and labeled fit masks."""
    if source.shape != target.shape or source.dtype != torch.bool or target.dtype != torch.bool:
        raise ValueError("Expected matching binary public and labeled fit masks.")
    if not len(source) or quality.shape != (len(source),) or epochs < 1:
        raise ValueError("Provide image-pair confidence scores and positive epochs.")
    torch.manual_seed(seed)
    generator = torch.Generator(device=source.device).manual_seed(seed + 9)
    model = LocalFontStyle().to(source.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0003)
    signed, features = edge_geometry(source)
    truth = target.float()[:, None]
    quality = (0.6 + 0.4 * ((quality - 0.55) / 0.35).clamp(0, 1))[:, None, None, None]
    reports = []
    for _ in range(epochs):
        running, seen = 0.0, 0
        for rows in torch.randperm(len(source), device=source.device, generator=generator).split(24):
            field, displacement = model(signed[rows], features[rows])
            logits = field / TEMPERATURE
            edge = signed[rows].abs() <= MAX_DISPLACEMENT
            selected = edge * quality[rows]
            pixel_error = F.binary_cross_entropy_with_logits(logits, truth[rows], reduction="none")
            pixel = (selected * pixel_error).sum() / selected.sum().clamp_min(1)
            energy = (displacement.square() * edge).sum() / edge.sum().clamp_min(1)
            smooth = (displacement.diff(dim=-1).abs().mean() + displacement.diff(dim=-2).abs().mean())
            contour = contour_loss(logits.sigmoid(), truth[rows], sigmas=(1, 2), hard_fraction=0.25)
            loss = 0.65 * pixel + 0.30 * contour + 0.04 * energy + 0.06 * smooth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(rows)
            seen += len(rows)
        schedule.step()
        reports.append(running / seen)
    return model.eval(), {"pairs": len(source), "epochs": epochs, "parameters": sum(p.numel() for p in model.parameters()),
                          "fit_objective_by_epoch": reports}


def _bounded_pixels(priority, allowed, budgets):
    available = priority.flatten(1).masked_fill(~allowed.flatten(1), -torch.inf)
    order = available.argsort(dim=1, descending=True)
    chosen = torch.arange(available.shape[1], device=available.device)[None] < budgets[:, None]
    flat = torch.zeros_like(available, dtype=torch.bool).scatter_(1, order, chosen)
    return flat.reshape_as(allowed)


@torch.inference_mode()
def render_local_style(model, images, *, forced_fraction=None):
    """Return binary local rendering and a control with identical edit counts.

    Both arms use the same edge band. The control receives the local model's
    numbers of additions and removals, but uses only distance and isotropic
    neighborhood density to choose where those edits occur.
    """
    masks = images if images.dtype == torch.bool else images >= 128
    signed, features = edge_geometry(masks)
    proposal, _ = model(signed, features)
    field = proposal[:, 0]
    edge = signed[:, 0].abs() <= MAX_DISPLACEMENT
    if forced_fraction is None:
        potential = ((field >= 0) != masks) & edge
        priority = field.abs()
        fraction = MAX_EDIT_FRACTION
    else:
        if not 0 <= forced_fraction <= MAX_EDIT_FRACTION:
            raise ValueError("Forced edits must stay within the same render budget.")
        direction = 1 - 2 * masks.float()
        potential = (direction * (field - signed[:, 0]) > 0) & edge
        priority = direction * field
        fraction = forced_fraction
    budget = (masks.sum(dim=(1, 2)) * fraction).long()
    count = torch.minimum(budget, potential.sum(dim=(1, 2)))
    changed = _bounded_pixels(priority, potential, count)
    plus = (changed & ~masks).sum(dim=(1, 2))
    minus = (changed & masks).sum(dim=(1, 2))
    density = features[:, 2]
    proximity = -signed[:, 0].abs()
    fill = _bounded_pixels(proximity + 0.25 * density, edge & ~masks, plus)
    clear = _bounded_pixels(proximity + 0.25 * (1 - density), edge & masks, minus)
    return {"global_font_style": masks, "local_font_style": masks ^ changed,
            "same_count_morphology": masks ^ (fill | clear)}
