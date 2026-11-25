"""Predict mouse visual responses with a compact receptive-field model."""

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _image_tensor(images, device):
    if images.ndim != 3 or images.shape[1:] != (64, 64) or not len(images):
        raise ValueError("Expected nonempty images of shape [glyphs, 64, 64].")
    images = images.to(device=device, dtype=torch.float32)
    if images.max().item() > 1:
        images = images / 255.0
    return images[:, None]


def _gabor_filters():
    coordinates = torch.arange(-3, 4, dtype=torch.float32)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    bank = []
    for angle in (0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        along = x * math.cos(angle) + y * math.sin(angle)
        across = -x * math.sin(angle) + y * math.cos(angle)
        envelope = torch.exp(-0.5 * ((along / 1.3).square() + (across / 1.7).square()))
        for phase in (0, math.pi / 2):
            kernel = envelope * torch.cos(2 * math.pi * along / 4 + phase)
            kernel = kernel - kernel.mean()
            bank.append(kernel / torch.linalg.vector_norm(kernel))
    return torch.stack(bank)[:, None]


def _inverse_softplus(rate):
    return rate + (-torch.expm1(-rate)).log()


class GlyphEncoder(nn.Module):
    """Shared image features with per-unit spatial readouts and session calibration."""

    def __init__(self, sessions, n_units, n_time_bins, fields_per_unit=2):
        super().__init__()
        self.sessions = tuple(sessions)
        self.session_index = {session: index for index, session in enumerate(self.sessions)}
        self.n_units = n_units
        self.n_time_bins = n_time_bins
        n_channels = 24
        self.core = nn.Sequential(
            nn.Conv2d(25, n_channels, 3, padding=1),
            nn.GroupNorm(4, n_channels),
            nn.SiLU(),
            nn.Conv2d(n_channels, n_channels, 3, dilation=2, padding=2),
            nn.GroupNorm(4, n_channels),
            nn.SiLU(),
        )
        self.field_centers = nn.Parameter(torch.zeros(n_units, fields_per_unit, 2))
        self.field_widths = nn.Parameter(torch.full((n_units, fields_per_unit, 2), -0.8))
        self.readout = nn.Parameter(torch.zeros(n_units, fields_per_unit, n_channels, n_time_bins))
        self.baseline = nn.Parameter(torch.zeros(n_units, n_time_bins))
        self.offsets = nn.Parameter(torch.zeros(len(sessions), n_units, n_time_bins))
        self.gains = nn.Parameter(torch.zeros(len(sessions), n_units, 1))

        coordinates = torch.linspace(-1, 1, 16)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("spatial_grid", torch.stack((x, y), dim=-1).reshape(1, 1, 256, 2))
        self.register_buffer("orientation_bank", _gabor_filters())
        self.register_buffer("training_rates", torch.zeros(len(sessions), n_units * n_time_bins))

    def image_features(self, images):
        half = F.avg_pool2d(images, 2)
        quarter = F.avg_pool2d(half, 2)
        fine = self._oriented_features(half)
        coarse = self._oriented_features(quarter)
        return torch.cat((quarter, F.avg_pool2d(fine, 2), coarse), dim=1)

    def _oriented_features(self, images):
        signed = F.conv2d(images, self.orientation_bank, padding=3)
        energy = signed.unflatten(1, (4, 2)).square().sum(dim=2).add(1e-6).sqrt()
        return torch.cat((signed, energy), dim=1)

    def visual_drive(self, features):
        maps = self.core(features).flatten(2)
        centers = 0.95 * self.field_centers.tanh()
        widths = 0.08 + 0.55 * self.field_widths.sigmoid()
        distance = (self.spatial_grid - centers[:, :, None]) / widths[:, :, None]
        fields = (-0.5 * distance.square().sum(dim=-1)).softmax(dim=-1)
        local_features = torch.einsum("bcp,urp->burc", maps, fields)
        return torch.einsum("burc,urct->but", local_features, self.readout)

    def _shared_and_sessions(self, drive):
        offsets = self.offsets - self.offsets.mean(dim=0)
        gains = self.gains - self.gains.mean(dim=0)
        return self.baseline + offsets[:, None] + gains.clamp(-1, 1).exp()[:, None] * drive

    def _expected(self, drive, session):
        if session is None:
            logits = self.baseline + drive
        else:
            logits = self._shared_and_sessions(drive)[self.session_index[session]]
        return F.softplus(logits).flatten(1).add(1e-6)

    @torch.no_grad()
    def expected_counts(self, images, session=None, *, batch_size=128):
        """Return per-trial expected counts [glyphs, units × time bins].

        Use ``session=None`` for the shared response model on a new day.
        """
        if batch_size < 1:
            raise ValueError("Use a positive prediction batch size.")
        images = _image_tensor(images, self.baseline.device)
        return torch.cat([self._expected(self.visual_drive(self.image_features(batch)), session)
                          for batch in images.split(batch_size)])

    @torch.inference_mode(False)
    @torch.enable_grad()
    def invert(self, observed_counts, initial_images, session=None, *, steps=24,
               learning_rate=0.20, prior_weight=0.20, smoothness=0.01, batch_size=64):
        """Refine initial images with spike likelihood and a smooth spatial correction.

        Counts are [glyphs, repeats, features] or averaged [glyphs, features].
        The returned soft images are [glyphs, 64, 64] float32 in [0, 1].
        """
        device = self.baseline.device
        initial = _image_tensor(initial_images, device).detach().clone()
        observed = observed_counts.to(device=device, dtype=torch.float32).detach().clone()
        if observed.ndim == 3:
            observed = observed.mean(dim=1)
        if observed.shape != (len(initial), self.n_units * self.n_time_bins):
            raise ValueError("Observed counts must align with the initial images and fitted unit/time dimensions.")
        if steps < 0 or learning_rate <= 0 or prior_weight < 0 or smoothness < 0 or batch_size < 1:
            raise ValueError("Invalid inversion settings.")
        if steps == 0:
            return initial[:, 0]

        result = []
        for batch, responses in zip(initial.split(batch_size), observed.split(batch_size)):
            base_logits = batch.clamp(0.04, 0.96).logit()
            correction = torch.zeros((len(batch), 1, 16, 16), device=device, requires_grad=True)
            optimizer = torch.optim.Adam([correction], lr=learning_rate)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                upsampled = F.interpolate(correction, size=(64, 64), mode="bilinear", align_corners=False)
                estimate = (base_logits + upsampled).sigmoid()
                expected = self._expected(self.visual_drive(self.image_features(estimate)), session)
                likelihood = F.poisson_nll_loss(expected, responses, log_input=False)
                prior = F.mse_loss(estimate, batch)
                roughness = correction.diff(dim=-1).square().mean() + correction.diff(dim=-2).square().mean()
                objective = likelihood + prior_weight * prior + smoothness * roughness
                correction.grad, = torch.autograd.grad(objective, correction)
                optimizer.step()
            with torch.no_grad():
                upsampled = F.interpolate(correction, size=(64, 64), mode="bilinear", align_corners=False)
                result.append((base_logits + upsampled).sigmoid()[:, 0])
        return torch.cat(result)


def _initialize_encoder(model, targets, rng):
    with torch.no_grad():
        for layer in model.core.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=rng)
                nn.init.zeros_(layer.bias)
        model.field_centers.uniform_(-0.75, 0.75, generator=rng)
        model.readout.normal_(mean=0, std=0.04, generator=rng)
        observed_means = targets.mean(dim=1).flatten(1)
        model.training_rates.copy_(observed_means)
        session_logits = _inverse_softplus(observed_means.clamp_min(0.001)).reshape_as(model.offsets)
        baseline = session_logits.mean(dim=0)
        model.baseline.copy_(baseline)
        model.offsets.copy_(session_logits - baseline)


def _mean_features(counts, n_images, n_features, device):
    if counts.ndim != 3 or counts.shape[0] != n_images or not counts.shape[1] or counts.shape[2] != n_features:
        raise ValueError("Each recording needs raw count features [glyphs, repeats, units × time bins].")
    return counts.to(device=device, dtype=torch.float32).mean(dim=1)


def fit_glyph_encoder(
    session_counts: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    *,
    n_time_bins: int = 10,
    fields_per_unit: int = 2,
    epochs: int = 32,
    batch_size: int = 64,
    seed: int = 42,
) -> GlyphEncoder:
    """Fit response means for aligned training glyphs from one mouse.

    Inputs are grouped raw spike counts, as returned by ``grouped_blocks``.
    Remove all held-out glyph rows from both inputs before fitting.
    """
    if not session_counts or min(n_time_bins, fields_per_unit, epochs, batch_size) < 1:
        raise ValueError("Provide training blocks and positive encoder settings.")
    first = next(iter(session_counts.values()))
    if first.ndim != 3 or not first.shape[-1] or first.shape[-1] % n_time_bins:
        raise ValueError("Expected grouped responses with whole units of time-bin features.")
    sessions = tuple(session_counts)
    device = first.device
    n_features = first.shape[-1]
    n_units = n_features // n_time_bins
    images = _image_tensor(images, device)
    targets = torch.stack([_mean_features(counts, len(images), n_features, device)
                           for counts in session_counts.values()]).unflatten(-1, (n_units, n_time_bins))

    model = GlyphEncoder(sessions, n_units, n_time_bins, fields_per_unit).to(device)
    rng = torch.Generator(device=device).manual_seed(seed)
    _initialize_encoder(model, targets, rng)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0002)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    with torch.no_grad():
        image_features = torch.cat([model.image_features(batch) for batch in images.split(batch_size)])
    for _ in range(epochs):
        indices = torch.randperm(len(images), generator=rng, device=device)
        for batch in indices.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                drive = model.visual_drive(image_features[batch])
            logits = model._shared_and_sessions(drive.float())
            expected = F.softplus(logits).add(1e-6)
            loss = F.poisson_nll_loss(expected, targets[:, batch], log_input=False)
            loss = loss + 0.001 * (model.gains.square().mean() + model.offsets.square().mean())
            if n_time_bins > 1:
                loss = loss + 0.001 * model.readout.diff(dim=-1).square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
    return model.eval()


def _deviance(observed, expected):
    expected = expected.clamp_min(1e-6)
    ratios = observed.clamp_min(1e-6) / expected
    return 2 * (expected - observed + torch.xlogy(observed, ratios))


@torch.no_grad()
def evaluate_glyph_encoder(encoder, images, session_counts, *, batch_size=128):
    """Measure response prediction on held glyphs in seen sessions or a new day.

    The null predictor is each feature's training rate, pooled for a new day.
    Scores are computed on repeat means with the same order as ``images``.
    """
    details = {}
    for session, counts in session_counts.items():
        index = encoder.session_index.get(session)
        prediction = encoder.expected_counts(images, session if index is not None else None,
                                            batch_size=batch_size)
        observed = _mean_features(counts, len(images), prediction.shape[1], prediction.device)
        if index is None:
            baseline = encoder.training_rates.mean(dim=0)
        else:
            baseline = encoder.training_rates[index]
        deviance = _deviance(observed, prediction).mean()
        null_deviance = _deviance(observed, baseline).mean()
        pred = prediction - prediction.mean(dim=0)
        actual = observed - observed.mean(dim=0)
        denominator = pred.norm(dim=0) * actual.norm(dim=0)
        valid = denominator > 1e-8
        correlations = (pred * actual).sum(dim=0) / denominator.clamp_min(1e-8)
        details[session] = {"n_glyphs": len(images), "deviance": deviance.item(),
                            "null_deviance": null_deviance.item(),
                            "deviance_explained": (1 - deviance / null_deviance.clamp_min(1e-8)).item(),
                            "response_correlation": (correlations[valid].mean().item() if valid.any() else None)}
    return details


def poisson_logits(observed, expected):
    """Score a class by the Poisson likelihood of its predicted spike rates."""
    expected = expected.clamp_min(1e-5)
    return observed.mean(1) @ expected.log().T - expected.sum(1)


def _evoked_counts(counts, view):
    delayed = counts.unflatten(-1, (-1, 10))[..., 2:]
    return delayed.sum(-1) if view == "rate" else delayed.flatten(-2)


def count_template_logits(observed, rates, view):
    counts = _evoked_counts(observed, view)
    expected = _evoked_counts(rates, view)
    expected = 0.98 * expected + 0.02 * expected.mean(dim=0)
    return poisson_logits(counts, expected)


@dataclass
class CountTemplateBank:
    shared: torch.Tensor
    session_rates: dict[str, torch.Tensor]
    totals: dict[str, torch.Tensor]
    repetitions: dict[str, int]

    @torch.no_grad()
    def for_recording(self, session, observed):
        """Apply label-free recording calibration to the learned gallery rates."""
        reference = self.session_rates.get(session, self.shared)
        reference_mean = reference.mean(0)
        pooled_count = torch.zeros_like(reference)
        for name, expected in self.session_rates.items():
            gain = ((reference_mean + 0.08) / (expected.mean(0) + 0.08)).clamp(0.5, 2)
            pooled_count += self.totals[name] * gain
        pooled = pooled_count / sum(self.repetitions.values())
        if session in self.repetitions:
            local = self.totals[session] / self.repetitions[session]
            pooled = 0.60 * pooled + 0.40 * local
        else:
            target_rate = observed.mean((0, 1))
            shift = ((target_rate + 0.08) / (reference_mean + 0.08)).clamp(0.5, 2)
            reference, pooled = reference * shift, pooled * shift
        return {"forward": reference, "measured": pooled, "shrinkage": 0.55 * reference + 0.45 * pooled}


@torch.no_grad()
def make_count_templates(encoder, images, training):
    return CountTemplateBank(encoder.expected_counts(images),
                             {name: encoder.expected_counts(images, name) for name in training},
                             {name: counts.sum(1) for name, counts in training.items()},
                             {name: counts.shape[1] for name, counts in training.items()})
