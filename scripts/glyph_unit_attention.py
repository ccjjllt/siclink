"""Unit-token attention adapted from https://github.com/SiClink/neural-to-image-retrieval."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class GlyphTarget:
    training: torch.Tensor
    coefficient_scale: torch.Tensor
    gallery_coefficients: torch.Tensor
    mean_agreement: torch.Tensor
    energy: torch.Tensor
    log_mean: torch.Tensor
    log_scale: torch.Tensor
    log_limits: tuple[torch.Tensor, torch.Tensor]


@torch.no_grad()
def fit_glyph_target(signatures, energy, *, seed=211, dimensions=96):
    """Create a deterministic low-rank basis for the training glyph signatures."""
    center = signatures.mean(dim=0)
    values = signatures - center
    rank = min(dimensions, min(values.shape) - 1)
    width = min(rank + 16, min(values.shape))
    rng = torch.Generator(device=values.device).manual_seed(seed)
    sketch = values @ torch.randn((values.shape[1], width), generator=rng, device=values.device)
    axes, _ = torch.linalg.qr(sketch, mode="reduced")
    axes, _ = torch.linalg.qr(values @ (values.T @ axes), mode="reduced")
    _, _, vectors = torch.linalg.svd(axes.T @ values, full_matrices=False)
    basis = vectors[:rank].T.contiguous()
    coefficients = values @ basis
    scale = coefficients.std(dim=0, unbiased=False).clamp_min(0.03)
    log_energy = energy.log()
    log_mean = log_energy.mean()
    log_scale = log_energy.std(unbiased=False).clamp_min(0.05)
    targets = torch.cat((coefficients / scale, ((log_energy - log_mean) / log_scale)[:, None]), dim=1)
    return GlyphTarget(targets, scale, signatures @ basis, center @ signatures.T, energy,
                       log_mean, log_scale, log_energy.aminmax())


def _unit_features(counts):
    if counts.ndim != 3 or counts.shape[-1] % 10:
        raise ValueError("Expected grouped spikes with ten adjacent 25 ms bins per unit")
    delayed = counts.float().unflatten(-1, (-1, 10))[..., 2:]
    early = delayed[..., :2].sum(dim=-1, keepdim=True)
    sustained = delayed[..., 2:].sum(dim=-1, keepdim=True)
    return torch.cat((delayed, early, sustained, early + sustained), dim=-1).sqrt_()


@torch.no_grad()
def prepare_units(training, images, *, functional_footprints=False):
    """Normalize each unit and, optionally, estimate its training glyph tuning."""
    normalized, moments = {}, {}
    for session, counts in training.items():
        features = _unit_features(counts)
        singles = features.flatten(0, 1)
        mean = singles.mean(dim=0)
        scale = (singles.var(dim=0, unbiased=False) + 0.05).sqrt()
        moments[session] = (mean, scale)
        normalized[session] = ((features - mean) / scale).clamp(-8, 8)

    footprint = None
    if functional_footprints:
        glyphs = F.adaptive_avg_pool2d((images[:, None] >= 128).float(), (6, 6)).flatten(1)
        glyphs -= glyphs.mean(dim=0)
        responses = torch.stack([values[..., 8:10].mean(dim=1) for values in normalized.values()]).mean(dim=0)
        gram = glyphs.T @ glyphs / len(glyphs)
        gram.diagonal().add_(0.10)
        weights = torch.cholesky_solve(glyphs.T @ responses.flatten(1) / len(glyphs), torch.linalg.cholesky(gram))
        footprint = weights.T.reshape(responses.shape[1], -1)
        footprint -= footprint.mean(dim=0)
        footprint = (footprint / (footprint.var(dim=0, unbiased=False) + 0.01).sqrt()).clamp(-4, 4)
    return normalized, moments, footprint


class UnitAttention(nn.Module):
    def __init__(self, n_units, input_width, target_width, *, identity, footprint=None):
        super().__init__()
        width = 32
        self.spikes = nn.Sequential(nn.Linear(input_width, width), nn.GELU(), nn.Linear(width, width))
        self.identity = nn.Embedding(n_units, width) if identity else None
        self.register_buffer("footprint", footprint)
        self.functional = nn.Linear(footprint.shape[1], width, bias=False) if footprint is not None else None
        self.unit_norm = nn.LayerNorm(width)
        self.queries = nn.Parameter(torch.empty(8, width))
        self.pool = nn.MultiheadAttention(width, 4, batch_first=True)
        self.mix = nn.MultiheadAttention(width, 4, batch_first=True)
        self.query_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.LayerNorm(8 * width), nn.Linear(8 * width, 160), nn.GELU(),
                                  nn.Linear(160, target_width))

    def forward(self, features):
        units = self.spikes(features)
        if self.identity is not None:
            identity = self.identity.weight
            units = units * (1 + 0.25 * identity.tanh()) + identity
        if self.functional is not None:
            units = units + 0.5 * self.functional(self.footprint)
        units = self.unit_norm(units)
        queries = self.queries[None].expand(len(units), -1, -1)
        pooled, _ = self.pool(queries, units, units, need_weights=False)
        pooled = self.query_norm(pooled + queries)
        mixed, _ = self.mix(pooled, pooled, pooled, need_weights=False)
        return self.head((pooled + mixed).flatten(1))


def _initialize(network, generator):
    with torch.no_grad():
        network.queries.normal_(std=0.15, generator=generator)
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5**0.5, generator=generator)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight, generator=generator)
                module.in_proj_bias.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.normal_(std=0.20, generator=generator)


def fit_unit_attention(normalized, target, *, identity, footprint=None, epochs=12, seed=211, batch_size=192):
    features = torch.cat([values.flatten(0, 1) for values in normalized.values()])
    glyphs = torch.arange(len(target.training), device=features.device)
    labels = torch.cat([glyphs.repeat_interleave(values.shape[1]) for values in normalized.values()])
    device = features.device
    net = UnitAttention(features.shape[1], features.shape[2], target.training.shape[1],
                        identity=identity, footprint=footprint).to(device)
    _initialize(net, torch.Generator(device=device).manual_seed(seed))
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00012)
    mixed = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=mixed)
    last_loss = 0.0
    for epoch in range(epochs):
        order = torch.randperm(len(features), device=device, generator=generator)
        for rows in order.split(batch_size):
            inputs = features[rows]
            retained = torch.rand(inputs.shape[:2], device=device, generator=generator) >= 0.08
            target_embedding = target.training[labels[rows]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=mixed):
                estimated = net(inputs * retained[..., None])
            estimated = estimated.float()
            coefficients = F.mse_loss(estimated[:, :-1], target_embedding[:, :-1])
            energy = F.mse_loss(estimated[:, -1], target_embedding[:, -1])
            loss = coefficients + 0.15 * energy
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if epoch in (0, epochs - 1) or (epoch + 1) % 4 == 0:
            last_loss = float(loss.detach())
            print(f"{device} unit attention id={identity} footprint={footprint is not None} "
                  f"{epoch + 1}/{epochs}: {last_loss:.4f}", flush=True)
    return net.eval(), last_loss


@torch.inference_mode()
def attention_glyph_scores(net, counts, moments, target, *, batch_size=256):
    features = ((_unit_features(counts) - moments[0]) / moments[1]).clamp(-8, 8).mean(dim=1)
    estimates = torch.cat([net(batch) for batch in features.split(batch_size)])
    coefficients = estimates[:, :-1] * target.coefficient_scale
    agreement = (coefficients @ target.gallery_coefficients.T + target.mean_agreement).clamp(0, 1)
    energy = (estimates[:, -1] * target.log_scale + target.log_mean).clamp(*target.log_limits).exp()
    ratio = torch.minimum(energy[:, None], target.energy) / torch.maximum(energy[:, None], target.energy)
    shared = agreement * ratio
    return shared / (1 + ratio - shared)
