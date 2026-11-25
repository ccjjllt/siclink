"""Conditional flow matching for reconstructing unseen SiClink glyphs.

The flow starts near a cross-fitted direct reconstruction and predicts a
binary glyph. It adapts Lipman et al.'s straight conditional probability
path and Sensorium-Viz's response conditioning to a compact convolutional
model trained on paired spike recordings from one mouse.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from glyph_reconstruction import RidgeReadout, _fit_ridge
from ocf_training import contour_loss


class ConditionedResidual(nn.Module):
    def __init__(self, channels, context_size):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(context_size, channels * 2)

    def forward(self, features, context):
        hidden = self.conv1(F.silu(self.norm1(features)))
        scale, shift = self.condition(context)[:, :, None, None].chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale) + shift
        return features + self.conv2(F.silu(hidden))


class GlyphVelocity(nn.Module):
    def __init__(self, n_features, n_sessions):
        super().__init__()
        self.session_embedding = nn.Embedding(n_sessions + 1, 16)
        self.neural_context = nn.Sequential(nn.Linear(n_features + 17, 160), nn.SiLU(),
                                           nn.Linear(160, 96), nn.LayerNorm(96))
        self.time_context = nn.Sequential(nn.Linear(13, 96), nn.SiLU(), nn.Linear(96, 96))
        self.neural_map = nn.Linear(96, 8 * 8 * 8)
        self.input = nn.Conv2d(5, 24, 3, padding=1)
        self.fine = ConditionedResidual(24, 96)
        self.down1 = nn.Conv2d(24, 40, 4, stride=2, padding=1)
        self.middle = ConditionedResidual(40, 96)
        self.down2 = nn.Conv2d(40, 64, 4, stride=2, padding=1)
        self.merge_neural = nn.Conv2d(72, 64, 1)
        self.coarse = ConditionedResidual(64, 96)
        self.merge_middle = nn.Conv2d(104, 40, 1)
        self.middle_up = ConditionedResidual(40, 96)
        self.merge_fine = nn.Conv2d(64, 24, 1)
        self.fine_up = ConditionedResidual(24, 96)
        self.output = nn.Conv2d(24, 4, 3, padding=1)
        self.register_buffer("frequencies", 2 * math.pi * torch.tensor([1, 2, 4, 8, 16, 32]))

    def conditioning(self, features, sessions, reliability):
        values = torch.cat((features, self.session_embedding(sessions), reliability[:, None]), dim=1)
        hidden = self.neural_context(values)
        return hidden, self.neural_map(hidden).reshape(-1, 8, 8, 8)

    def forward(self, state, time, anchor, neural):
        hidden, spatial = neural
        angles = time[:, None] * self.frequencies
        context = hidden + self.time_context(torch.cat((time[:, None], angles.sin(), angles.cos()), dim=1))
        fine = self.fine(self.input(torch.cat((F.pixel_unshuffle(state, 2), anchor), dim=1)), context)
        middle = self.middle(self.down1(fine), context)
        coarse = self.merge_neural(torch.cat((self.down2(middle), spatial), dim=1))
        coarse = self.coarse(coarse, context)
        coarse = F.interpolate(coarse, scale_factor=2, mode="nearest")
        middle = self.middle_up(self.merge_middle(torch.cat((coarse, middle), dim=1)), context)
        middle = F.interpolate(middle, scale_factor=2, mode="nearest")
        fine = self.fine_up(self.merge_fine(torch.cat((middle, fine), dim=1)), context)
        return F.pixel_shuffle(self.output(fine), 2)


@dataclass
class FlowAnchors:
    pooled: RidgeReadout
    local: dict[str, RidgeReadout]

    def predict(self, session, features):
        pooled = self.pooled.predict(features)
        if session in self.local:
            pooled = 0.70 * self.local[session].predict(features) + 0.30 * pooled
        return pooled.reshape(*features.shape[:-1], 32, 32)


def _fit_anchors(trials, pixels, sessions):
    means = trials.mean(dim=2)
    pooled = _fit_ridge(means.mean(dim=0), pixels, ridge=0.16)
    local = {session: _fit_ridge(values, pixels, ridge=0.16)
             for session, values in zip(sessions, means)}
    return FlowAnchors(pooled, local)


def _anchor_image(anchor):
    return F.interpolate(anchor, size=(64, 64), mode="bilinear", align_corners=False)


@dataclass
class GlyphFlow:
    velocity: GlyphVelocity
    anchors: FlowAnchors
    sessions: tuple[str, ...]
    train_repeats: int
    noise_scale: float
    foreground: float

    def _prepare(self, session, trials):
        device = self.velocity.frequencies.device
        trials = trials.to(device=device, dtype=torch.float32)
        dimension = self.anchors.pooled.weights.shape[0]
        if trials.ndim not in (2, 3) or not len(trials) or trials.shape[-1] != dimension:
            raise ValueError("Provide preprocessed response vectors [N,D] or trials [N,R,D].")
        if trials.ndim == 3 and not trials.shape[1]:
            raise ValueError("Provide at least one trial per prediction.")
        n_repeats = trials.shape[1] if trials.ndim == 3 else self.train_repeats
        features = trials.mean(1) if trials.ndim == 3 else trials
        reliability = features.new_full((len(features),), min(n_repeats / self.train_repeats, 1) ** 0.5)
        session_number = self.sessions.index(session) if session in self.sessions else len(self.sessions)
        sessions = torch.full((len(features),), session_number, dtype=torch.long, device=device)
        anchors = self.anchors.predict(session, features)[:, None].clamp(0, 1)
        return features, anchors, reliability, sessions

    @torch.no_grad()
    def sample(self, session, trials, *, n_samples=4, steps=6, batch_size=64, seed=211,
               heun=False):
        """Draw soft glyph hypotheses [N, samples, 64, 64] from the conditional flow."""
        features, anchors, reliability, sessions = self._prepare(session, trials)
        if min(n_samples, steps, batch_size) < 1:
            raise ValueError("Samples, ODE steps and batch size must be positive.")
        device = features.device
        rng = torch.Generator(device=device).manual_seed(seed)
        parts = torch.arange(len(features), device=device).split(max(1, batch_size // n_samples))
        predictions = []
        for batch in parts:
            ordered = batch.repeat_interleave(n_samples)
            base = anchors[ordered]
            center = 2 * _anchor_image(base) - 1
            noise = torch.randn(center.shape, device=device, generator=rng)
            state = center + self.noise_scale * noise
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                neural = self.velocity.conditioning(features[ordered], sessions[ordered], reliability[ordered])
                for step in range(steps):
                    time = state.new_full((len(ordered),), step / steps)
                    update = self.velocity(state, time, base, neural).float() / steps
                    if heun:
                        future = self.velocity(state + update, time + 1 / steps, base, neural).float() / steps
                        update = 0.5 * (update + future)
                    state = state + update
            pixels = ((state + 1) * 0.5).clamp(0, 1)
            predictions.append(pixels.reshape(len(batch), n_samples, 64, 64))
        return torch.cat(predictions)

    @torch.no_grad()
    def predict_variants(self, session, trials, **options):
        """Return uint8 anchors and consensus or sampled glyphs for exact OCF."""
        _, anchors, _, _ = self._prepare(session, trials)
        reference = _anchor_image(anchors)[:, 0]
        candidates = self.sample(session, trials, **options)
        reference_flat = reference.flatten(1)
        area_level = torch.quantile(reference_flat, 1 - self.foreground, dim=1, keepdim=True)
        threshold = 0.75 * area_level + 0.25 * 0.45
        anchor = (reference_flat >= threshold).reshape_as(reference)
        masks = candidates >= 0.5
        average = 0.90 * candidates.mean(dim=1) + 0.10 * reference
        consensus = average >= 0.5
        flow_level = torch.quantile(average.flatten(1), 1 - self.foreground, dim=1)
        ink_adjusted = average >= (0.60 * flow_level + 0.40 * 0.5)[:, None, None]
        flattened = masks.flatten(2).float()
        intersection = flattened @ flattened.transpose(1, 2)
        masses = flattened.sum(dim=-1)
        union = masses[:, :, None] + masses[:, None, :] - intersection
        neighbor_overlap = (intersection / union.clamp_min(1)).mean(dim=-1)
        prior_intersection = (flattened * anchor.flatten(1)[:, None]).sum(dim=-1)
        prior_union = masses + anchor.sum(dim=(1, 2))[:, None] - prior_intersection
        center = (neighbor_overlap + 0.10 * (prior_intersection / prior_union.clamp_min(1))).argmax(dim=1)
        medoid = masks[torch.arange(len(masks), device=masks.device), center]
        return {"flow_anchor": anchor.to(torch.uint8) * 255,
                "flow_consensus": consensus.to(torch.uint8) * 255,
                "flow_ink": ink_adjusted.to(torch.uint8) * 255,
                "flow_medoid": medoid.to(torch.uint8) * 255}


def _initialize(network, rng):
    with torch.no_grad():
        for layer in network.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=rng)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Embedding):
                layer.weight.normal_(mean=0, std=0.1, generator=rng)
        nn.init.zeros_(network.output.weight)


def fit_glyph_flow(
    session_features: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    noise_scale: float = 0.30,
    contour_weight: float = 0.0,
    seed: int = 211,
    verbose: bool = False,
) -> GlyphFlow:
    """Fit on aligned training glyphs and preprocessed [N,R,D] trial features.

    Use one fit per mouse. All sessions must share the same number of
    available training repeats. Remove every outer held glyph from both inputs.
    During training, each minibatch draws a random subset of real repeats.
    The spatial conditioning maps for a glyph come from readouts fit without it.
    """
    if not session_features or images.ndim != 3 or images.shape[1:] != (64, 64) or len(images) < 2:
        raise ValueError("Provide sessions with aligned features for at least two 64 × 64 training glyphs.")
    first = next(iter(session_features.values()))
    if first.ndim != 3 or first.shape[0] != len(images) or min(first.shape[1:]) < 1 or any(
        features.shape != first.shape for features in session_features.values()
    ):
        raise ValueError("Training feature blocks must share shape [glyph, repeat, feature].")
    if min(epochs, batch_size) < 1 or noise_scale <= 0 or contour_weight < 0:
        raise ValueError("Use positive training settings and a nonnegative contour weight.")

    device = first.device
    rng = torch.Generator(device=device).manual_seed(seed)
    sessions = tuple(session_features)
    trials = torch.stack([features.to(device=device, dtype=torch.float32) for features in session_features.values()])
    n_sessions, n_glyphs, n_repeats, dimension = trials.shape
    images = images.to(device)
    threshold = 0.5 if images.max().item() <= 1 else 128
    masks = (images >= threshold).to(torch.float32)
    targets = masks[:, None] * 2 - 1
    flattened = F.avg_pool2d(masks[:, None], 2).flatten(1)

    with torch.no_grad():
        storage_dtype = torch.float16 if device.type == "cuda" else torch.float32
        spatial = torch.empty((n_sessions, n_glyphs, n_repeats, 32, 32), device=device, dtype=storage_dtype)
        folds = torch.randperm(n_glyphs, device=device, generator=rng).tensor_split(min(n_glyphs, 4))
        for held in folds:
            fit = torch.ones(n_glyphs, dtype=torch.bool, device=device)
            fit[held] = False
            fold = _fit_anchors(trials[:, fit], flattened[fit], sessions)
            for index, session in enumerate(sessions):
                spatial[index, held] = fold.predict(session, trials[index, held]).to(storage_dtype)
            del fold
        anchors = _fit_anchors(trials, flattened, sessions)

    network = GlyphVelocity(dimension, n_sessions).to(device)
    _initialize(network, rng)
    ema = deepcopy(network).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    for epoch in range(epochs):
        order = torch.randperm(n_sessions * n_glyphs, device=device, generator=rng)
        total_loss = masks.new_zeros(())
        for batch in order.split(batch_size):
            recording = batch // n_glyphs
            glyph = batch % n_glyphs
            count = torch.randint(1, n_repeats + 1, (len(batch),), device=device, generator=rng)
            ranks = torch.rand((len(batch), n_repeats), device=device, generator=rng).argsort(dim=1)
            weights = (ranks < count[:, None]).to(torch.float32) / count[:, None]
            response = (trials[recording, glyph] * weights[:, :, None]).sum(dim=1)
            base = (spatial[recording, glyph] * weights[:, :, None, None]).sum(dim=1)
            base = base[:, None].clamp(0, 1)
            reliability = (count.to(torch.float32) / n_repeats).sqrt()
            hide_session = torch.rand(len(batch), device=device, generator=rng) < 0.10
            recording = recording.masked_fill(hide_session, n_sessions)

            final = targets[glyph]
            noise = torch.randn(final.shape, device=device, generator=rng)
            initial = 2 * _anchor_image(base) - 1 + noise_scale * noise
            time = torch.rand(len(batch), device=device, generator=rng)
            fraction = time[:, None, None, None]
            point = initial.lerp(final, fraction)
            target_velocity = final - initial
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                neural = network.conditioning(response, recording, reliability)
                velocity = network(point, time, base, neural)
            velocity = velocity.float()
            flow_loss = F.mse_loss(velocity, target_velocity)
            end = ((point + (1 - fraction) * velocity + 1) * 0.5).clamp(0, 1)
            true = masks[glyph][:, None]
            overlap = (end * true).sum(dim=(1, 2, 3))
            total_ink = (end + true).sum(dim=(1, 2, 3))
            stroke_loss = 1 - ((2 * overlap + 1) / (total_ink + 1)).mean()
            loss = flow_loss + 0.20 * stroke_loss
            if contour_weight:
                loss = loss + contour_weight * contour_loss(end, true, sigmas=(1, 2), hard_fraction=0.25)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for average, latest in zip(ema.parameters(), network.parameters()):
                    average.lerp_(latest, 0.02)
                total_loss += loss.detach() * len(batch)
        scheduler.step()
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            print(f"Flow epoch {epoch + 1}/{epochs}: {(total_loss / len(order)).item():.4f}", flush=True)
    return GlyphFlow(ema, anchors, sessions, n_repeats, noise_scale, masks.mean().item())
