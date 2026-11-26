"""Fit matched visual-cortex encoders with supervised near-glyph response differences."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from glyph_encoding import GlyphEncoder, _image_tensor, _initialize_encoder
from kaggle_gallery_validation import image_descriptors


ARCHITECTURES = {"mixed_real": (False, False), "bipartite_real": (True, False), "bipartite_shuffled": (True, True)}
PAIR_WEIGHT = 0.12
INK_TOLERANCE = 0.04


class BipartiteGlyphEncoder(GlyphEncoder):
    """Read signed coarse structure and phase-invariant local energy in two separate fields."""

    def __init__(self, sessions, units, *, structured):
        super().__init__(sessions, units, 10)
        channels = self.readout.shape[2] // 2
        self.core = nn.ModuleList(nn.Sequential(nn.Conv2d(9, channels, 3, padding=1), nn.GroupNorm(3, channels),
                                               nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
                                               nn.GroupNorm(3, channels), nn.SiLU()) for _ in range(2))
        self.readout = nn.Parameter(torch.zeros(units, 2, channels, 10))
        self.structured = structured
        self.register_buffer("visual_mean", torch.zeros(2, 9, 1, 1))
        self.register_buffer("visual_scale", torch.ones(2, 9, 1, 1))

    def image_features(self, images):
        half = F.avg_pool2d(images, 2)
        quarter = F.avg_pool2d(half, 2)
        coarse_signed = F.conv2d(quarter, self.orientation_bank, padding=3)
        energy_half = self._oriented_features(half)[:, 8:]
        energy_full = self._oriented_features(images)[:, 8:]
        local_contrast = (F.avg_pool2d(half.square(), 2) - quarter.square()).clamp_min(0).add(1e-6).sqrt()
        coarse = torch.cat((quarter, coarse_signed), dim=1)
        fine = torch.cat((local_contrast, F.avg_pool2d(energy_half, 2), F.avg_pool2d(energy_full, 4)), dim=1)
        return torch.stack((coarse, fine), dim=1)

    def visual_drive(self, features):
        lanes = (features - self.visual_mean) / self.visual_scale
        first, second = lanes.unbind(dim=1)
        if not self.structured:
            first, second = (first + second) * (0.5 ** 0.5), (first - second) * (0.5 ** 0.5)
        maps = torch.stack((self.core[0](first), self.core[1](second)), dim=1).flatten(3)
        centers = 0.95 * self.field_centers.tanh()
        # Same narrow/broad spatial supports in the two architectures.
        minimum = centers.new_tensor((0.065, 0.10))[None, :, None]
        span = centers.new_tensor((0.18, 0.33))[None, :, None]
        widths = minimum + span * self.field_widths.sigmoid()
        distance = (self.spatial_grid - centers[:, :, None]) / widths[:, :, None]
        fields = (-0.5 * distance.square().sum(dim=-1)).softmax(dim=-1)
        local = torch.einsum("brcp,urp->burc", maps, fields)
        return torch.einsum("burc,urct->but", local, self.readout)


def timing_view(rates):
    """Variance-stabilized total plus early, middle and late visually evoked counts."""
    windows = (rates[..., 2:].sum(-1), rates[..., 2:4].sum(-1), rates[..., 4:7].sum(-1), rates[..., 7:].sum(-1))
    return (torch.stack(windows, dim=-1) + 0.375).sqrt().flatten(-2)


@torch.no_grad()
def repeat_noise(training):
    scales = []
    for counts in training.values():
        features = timing_view(counts.unflatten(-1, (-1, 10)))
        variance = features.var(dim=1, unbiased=True).mean(dim=0)
        floor = variance.median().clamp_min(0.015)
        scales.append((0.8 * variance + 0.2 * floor).clamp_min(0.010).sqrt())
    return torch.stack(scales)


@torch.no_grad()
def neighbors(images, depth=8):
    binary = (images >= 128).to(torch.uint8) * 255
    descriptors = F.normalize(image_descriptors(binary), dim=1)
    ink = (binary > 0).float().mean(dim=(1, 2))
    distance = (ink[:, None] - ink[None]).abs()
    similar = descriptors @ descriptors.T
    similar.fill_diagonal_(-torch.inf)
    close = similar.masked_fill(distance > INK_TOLERANCE, -torch.inf)
    # Use closest ink matches only for rare rows lacking enough eligible neighbors.
    close = torch.where(torch.isfinite(close).sum(dim=1, keepdim=True) >= depth, close, similar - distance * 8)
    return close.topk(depth, dim=1).indices, ink


@dataclass(frozen=True)
class NearTraining:
    sessions: tuple[str, ...]
    counts_mean: torch.Tensor
    observed: torch.Tensor
    noise: torch.Tensor
    nearby: torch.Tensor
    features: torch.Tensor
    normalization: tuple[torch.Tensor, torch.Tensor]


@torch.no_grad()
def training_inputs(training, images):
    first = next(iter(training.values()))
    sessions = tuple(training)
    device = first.device
    units = first.shape[-1] // 10
    target = torch.stack([counts.mean(1) for counts in training.values()]).unflatten(-1, (units, 10))
    noise = repeat_noise(training)
    observed = timing_view(target) / noise[:, None]
    nearby, _ = neighbors(images)
    frame = BipartiteGlyphEncoder(sessions, units, structured=True).to(device)
    pictures = _image_tensor(images, device)
    features = torch.cat([frame.image_features(batch).to(torch.float16) for batch in pictures.split(64)])
    fp32 = features.float()
    center = fp32.mean(dim=(0, 3, 4), keepdim=False)[..., None, None]
    scale = fp32.std(dim=(0, 3, 4), unbiased=False)[..., None, None].clamp_min_(0.01)
    return NearTraining(sessions, target, observed, noise, nearby, features, (center, scale))


def fit_near_encoder(inputs, arm, *, epochs, seed, batch_size=72):
    structured, shuffled = ARCHITECTURES[arm]
    device = inputs.counts_mean.device
    model = BipartiteGlyphEncoder(inputs.sessions, inputs.counts_mean.shape[-2], structured=structured).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    _initialize_encoder(model, inputs.counts_mean, generator)
    with torch.no_grad():
        model.visual_mean.copy_(inputs.normalization[0])
        model.visual_scale.copy_(inputs.normalization[1])
        model.field_centers[:, 0].uniform_(-0.65, 0.65, generator=generator)
        from_first = torch.randn(model.field_centers[:, 0].shape, device=device, generator=generator)
        model.field_centers[:, 1].copy_((model.field_centers[:, 0] + 0.30 * F.normalize(from_first, dim=1)).clamp(-0.8, 0.8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0002)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    glyphs, depth = inputs.nearby.shape
    shuffle_rng = torch.Generator(device=device).manual_seed(seed + 47)
    signs = 2 * torch.randint(2, (glyphs, depth), device=device, generator=shuffle_rng) - 1
    loss_report = None
    for _ in range(epochs):
        order = torch.randperm(glyphs, device=device, generator=generator)
        logged = []
        for left in order.split(batch_size):
            slot = torch.randint(depth, (len(left),), device=device, generator=generator)
            right = inputs.nearby[left, slot]
            selected, inverse = torch.unique(torch.cat((left, right)), sorted=True, return_inverse=True)
            a, b = inverse.split(len(left))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                drive = model.visual_drive(inputs.features[selected])
            expected = F.softplus(model._shared_and_sessions(drive.float())).add(1e-6)
            likelihood = F.poisson_nll_loss(expected, inputs.counts_mean[:, selected], log_input=False)
            predicted = timing_view(expected) / inputs.noise[:, None]
            actual_diff = inputs.observed[:, left] - inputs.observed[:, right]
            if shuffled:
                actual_diff = actual_diff * signs[left, slot][None, :, None]
            pair = (predicted[:, a] - predicted[:, b] - actual_diff).square().mean()
            fields = 0.95 * model.field_centers.tanh()
            separation = (fields[:, 0] - fields[:, 1]).norm(dim=-1)
            geometry = ((0.12 - separation).clamp_min(0).square() + (separation - 0.55).clamp_min(0).square()).mean()
            regularization = (model.gains.square().mean() + model.offsets.square().mean()
                              + model.readout.diff(dim=-1).square().mean())
            loss = likelihood + PAIR_WEIGHT * pair + 0.025 * geometry + 0.001 * regularization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            logged.append(torch.stack((likelihood.detach(), pair.detach(), geometry.detach())))
        schedule.step()
        loss_report = torch.stack(logged).mean(dim=0).tolist()
    return model.eval(), {"parameters": sum(parameter.numel() for parameter in model.parameters()), "epochs": epochs,
                          "final_poisson": loss_report[0], "final_delta_loss": loss_report[1],
                          "final_geometry_penalty": loss_report[2], "shuffled_negative_label_fraction":
                          float((signs < 0).float().mean()) if shuffled else 0.0}
