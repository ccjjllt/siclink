"""Fit an image-pair utility to exact OCF labels in the known glyph gallery."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GalleryGeometry:
    parts: tuple[torch.Tensor, ...]
    scalars: torch.Tensor

    def features(self, action, reference, *, crossed=False):
        left, right = self.scalars[action], self.scalars[reference]
        if crossed:
            left, right = left[:, None], right[None]
        difference = left - right
        numbers = [left[..., 0] + right[..., 0], *difference.abs().unbind(-1), difference[..., 0]]
        for part in self.parts:
            a, b = part[action], part[reference]
            numbers.append(a @ b.T if crossed else (a * b).sum(dim=1))
        return torch.stack(numbers, dim=-1)


@torch.no_grad()
def gallery_geometry(masks, signatures, energy):
    pixels = masks.float()
    axis = torch.linspace(-1.0, 1.0, pixels.shape[1], device=pixels.device)
    mass = pixels.sum(dim=(1, 2)).clamp_min(1)
    horizontal = pixels.sum(dim=1) @ axis / mass
    vertical = pixels.sum(dim=2) @ axis / mass
    scales = signatures.unflatten(1, (3, 1664))
    parts = tuple(part.contiguous() for scale in scales.unbind(1) for part in scale.split((1024, 640), dim=1))
    silhouette = F.normalize(F.avg_pool2d(pixels[:, None], 4).flatten(1))
    scalars = torch.stack((energy.log(), (mass / 4096).log(), horizontal, vertical), dim=1)
    return GalleryGeometry((*parts, silhouette), scalars)


def pair_features(geometry, action, reference):
    return torch.cat([geometry.features(a, r) for a, r in zip(action.split(768), reference.split(768))])


def select_pairs(pool, proxy, size, generator):
    """Cover the whole score range with both random and visually close pairs."""
    device = pool.device
    random = size // 2
    position = torch.randint(len(pool), (size,), device=device, generator=generator)
    reference = pool[position]
    delta = torch.randint(1, len(pool), (random,), device=device, generator=generator)
    unrelated = pool[(position[:random] + delta).remainder(len(pool))]
    related = reference[random:]
    similarities = proxy[related][:, pool].clone()
    similarities.scatter_(1, position[random:, None], -torch.inf)
    shortlist = similarities.topk(min(24, len(pool) - 1), dim=1).indices
    draw = torch.randint(shortlist.shape[1], (len(related), 1), device=device, generator=generator)
    nearby = pool[shortlist.gather(1, draw)[:, 0]]
    return torch.cat((unrelated, nearby)), reference


@dataclass(frozen=True)
class ExactUtility:
    network: nn.Module
    center: torch.Tensor
    scale: torch.Tensor

    def predict(self, features):
        inputs = ((features - self.center) / self.scale).clamp(-8, 8)
        return self.network(inputs).squeeze(-1).sigmoid() * 100


def fit_exact_utility(geometry, action, reference, exact, *, seed):
    values = pair_features(geometry, action, reference)
    center = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(0.03)
    torch.manual_seed(seed)
    network = nn.Sequential(nn.Linear(values.shape[-1], 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(),
                            nn.Linear(32, 1)).to(values.device)
    model = ExactUtility(network, center, scale)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0025, weight_decay=0.015)
    rng = torch.Generator(device=values.device).manual_seed(seed)
    for _ in range(40):
        for batch in torch.randperm(len(values), device=values.device, generator=rng).split(256):
            loss = F.smooth_l1_loss(model.predict(values[batch]) / 100, exact[batch] / 100, beta=0.06)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    network.eval()
    return model


@torch.no_grad()
def gallery_utility(model, geometry):
    """Return [action, reference] utility, preserving the exact image diagonal."""
    n = len(geometry.scalars)
    pool = torch.arange(n, device=geometry.scalars.device)
    result = geometry.scalars.new_empty(n, n)
    for rows in pool.split(96):
        result[rows] = model.predict(geometry.features(rows, pool, crossed=True))
    return result.fill_diagonal_(100)
