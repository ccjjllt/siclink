"""Open-set reconstruction from aligned per-session training glyph prototypes."""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RidgeReadout:
    weights: torch.Tensor
    bias: torch.Tensor

    def predict(self, features):
        return features @ self.weights + self.bias


def _fit_ridge(features, targets, ridge):
    center = features.mean(dim=0)
    features = features - center
    target_mean = targets.mean(dim=0)
    gram = features.T @ features / len(features)
    gram.diagonal().add_(ridge)
    cross = features.T @ (targets - target_mean) / len(features)
    weights = torch.cholesky_solve(cross, torch.linalg.cholesky(gram))
    return RidgeReadout(weights, target_mean - center @ weights)


def _stable_dimensions(session_features, n_components):
    """Find reproducible response directions from paired training glyphs."""
    pooled = session_features.mean(dim=0)
    if len(session_features) > 1:
        deviations = (session_features - pooled).flatten(0, 1)
    else:
        deviations = pooled
    covariance = deviations.T @ deviations / len(deviations)
    covariance.diagonal().add_(0.35 * covariance.diagonal().mean().clamp_min(1e-4))
    lower = torch.linalg.cholesky(covariance)

    whitened = torch.linalg.solve_triangular(lower, pooled.T.contiguous(), upper=False)
    spectrum = whitened @ whitened.T / len(pooled)
    eigenvalues, basis = torch.linalg.eigh(spectrum)
    dimension = min(n_components, len(eigenvalues), len(pooled))
    eigenvalues = eigenvalues[-dimension:].clamp_min(0)
    basis = basis[:, -dimension:]
    reliability = (eigenvalues / (eigenvalues + 0.25)).sqrt()
    projection = torch.linalg.solve_triangular(lower.T, basis, upper=True)
    return projection * reliability


def _kernel(left, right, scale):
    inner = left @ right.T
    distances = (left.square().sum(1, keepdim=True) + right.square().sum(1) - 2 * inner)
    distances = distances.clamp_min_(0).div_(scale)
    return 0.5 * inner / scale + 0.5 * torch.exp(-distances) + 0.5 * torch.exp(-0.25 * distances)


@dataclass
class KernelRidge:
    anchors: torch.Tensor
    scale: torch.Tensor
    column_mean: torch.Tensor
    grand_mean: torch.Tensor
    weights: torch.Tensor
    pixel_mean: torch.Tensor

    def predict(self, features):
        gram = _kernel(features, self.anchors, self.scale)
        gram = gram - gram.mean(1, keepdim=True) - self.column_mean + self.grand_mean
        return gram @ self.weights + self.pixel_mean


def _fit_kernel_ridge(anchors, pixels, ridge):
    # Twice the average squared norm is the mean pairwise squared distance
    # for centered points, up to the finite-sample correction.
    scale = (2 * anchors.square().sum(dim=1).mean()).clamp_min(1e-6)
    gram = _kernel(anchors, anchors, scale)
    column_mean = gram.mean(dim=0)
    grand_mean = column_mean.mean()
    gram.sub_(column_mean).sub_(column_mean[:, None]).add_(grand_mean)
    gram.diagonal().add_(ridge)
    pixel_mean = pixels.mean(dim=0)
    weights = torch.cholesky_solve(pixels - pixel_mean, torch.linalg.cholesky(gram))
    return KernelRidge(anchors, scale, column_mean, grand_mean, weights, pixel_mean)


@dataclass
class ReconstructionReadout:
    local_linear: dict[str, RidgeReadout]
    pooled_linear: RidgeReadout
    centers: dict[str, torch.Tensor]
    projection: torch.Tensor
    nonlinear: dict[str, KernelRidge]

    def predict(self, session, features):
        local = self.local_linear[session].predict(features)
        pooled = self.pooled_linear.predict(features)
        linear = 0.70 * local + 0.30 * pooled
        projected = (features - self.centers[session]) @ self.projection
        nonlinear = self.nonlinear[session].predict(projected)
        nonlinear = 0.80 * nonlinear + 0.20 * pooled
        return linear.reshape(-1, 64, 64), nonlinear.reshape(-1, 64, 64)


def _fit_readout(session_features, images, n_components, linear_ridge, kernel_ridge):
    sessions = tuple(session_features)
    pixels = images.flatten(1)
    stacked = torch.stack(tuple(session_features.values()))
    centers = stacked.mean(dim=1)
    projection = _stable_dimensions(stacked - centers[:, None], n_components)

    pooled = stacked.mean(dim=0)
    projected = (stacked - centers[:, None]) @ projection
    projected_pool = projected.mean(dim=0)
    local = {}
    nonlinear = {}
    for index, session in enumerate(sessions):
        local[session] = _fit_ridge(stacked[index], pixels, linear_ridge)
        anchors = 0.85 * projected[index] + 0.15 * projected_pool
        nonlinear[session] = _fit_kernel_ridge(anchors, pixels, kernel_ridge)
    return ReconstructionReadout(local, _fit_ridge(pooled, pixels, linear_ridge),
                                 dict(zip(sessions, centers)), projection, nonlinear)


def _conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         nn.GroupNorm(4, out_channels), nn.SiLU())


class GlyphRefiner(nn.Module):
    """Learn a spatial correction to linear and kernel response readouts."""

    def __init__(self):
        super().__init__()
        self.fine = _conv_block(3, 16)
        self.middle = nn.Sequential(_conv_block(16, 24), _conv_block(24, 24))
        self.coarse = nn.Sequential(_conv_block(24, 32), _conv_block(32, 32))
        self.middle_up = _conv_block(56, 24)
        self.fine_up = _conv_block(40, 16)
        self.output = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, estimates):
        fine = self.fine(estimates)
        middle = self.middle(F.avg_pool2d(fine, 2))
        coarse = self.coarse(F.avg_pool2d(middle, 2))
        coarse = F.interpolate(coarse, scale_factor=2, mode="bilinear", align_corners=False)
        middle = self.middle_up(torch.cat((middle, coarse), dim=1))
        middle = F.interpolate(middle, scale_factor=2, mode="bilinear", align_corners=False)
        residual = self.output(self.fine_up(torch.cat((fine, middle), dim=1)))[:, 0]
        return 6.0 * (estimates[:, 1] - 0.40) + residual


def _orientation_filters(device):
    return torch.tensor([[[0,  0, 0], [ 1, -2, 1], [0, 0, 0]],
                         [[0,  1, 0], [ 0, -2, 0], [0, 1, 0]],
                         [[1,  0, 0], [ 0, -2, 0], [0, 0, 1]],
                         [[0,  0, 1], [ 0, -2, 0], [1, 0, 0]]],
                        dtype=torch.float32, device=device)[:, None]


def _refinement_loss(logits, target, filters):
    probabilities = logits.sigmoid()[:, None]
    target = target[:, None]
    pixel = F.binary_cross_entropy_with_logits(logits[:, None], target)
    overlap = (probabilities * target).sum(dim=(1, 2, 3))
    mass = (probabilities + target).sum(dim=(1, 2, 3))
    dice = (2 * overlap + 1) / (mass + 1)

    # Differentiable contour objective. The official OCF ranks the resulting
    # PNGs separately and also performs geometric alignment.
    edges = F.conv2d(F.avg_pool2d(probabilities, 3, stride=1, padding=1), filters, padding=1)
    target_edges = F.conv2d(F.avg_pool2d(target, 3, stride=1, padding=1), filters, padding=1)
    intersection = (edges * target_edges).sum(dim=(1, 2, 3)).clamp_min(0)
    union = (edges.square() + target_edges.square()).sum(dim=(1, 2, 3)) - intersection
    contour = (intersection + 1e-6) / (union + 1e-6)
    return pixel + 0.50 * (1 - dice.mean()) + 0.15 * (1 - contour.mean())


def _estimates(linear, nonlinear, spatial_prior):
    return torch.stack((linear.clamp(0, 1), nonlinear.clamp(0, 1),
                        spatial_prior.expand_as(nonlinear)), dim=1)


def _fit_refiner(readout_options, session_features, images, seed, epochs, batch_size, contour_epochs):
    device = images.device
    n_glyphs = len(images)
    sessions = tuple(session_features)
    rng = torch.Generator(device=device).manual_seed(seed)
    order = torch.randperm(n_glyphs, device=device, generator=rng)
    estimates = images.new_empty((len(sessions), n_glyphs, 3, 64, 64))
    with torch.no_grad():
        for heldout in order.tensor_split(min(4, n_glyphs)):
            fit_mask = torch.ones(n_glyphs, dtype=torch.bool, device=device)
            fit_mask[heldout] = False
            fit_images = images[fit_mask]
            fit_features = {session: values[fit_mask] for session, values in session_features.items()}
            readout = _fit_readout(fit_features, fit_images, *readout_options)
            prior = fit_images.mean(dim=0)
            for index, session in enumerate(sessions):
                linear, nonlinear = readout.predict(session, session_features[session][heldout])
                estimates[index, heldout] = _estimates(linear, nonlinear, prior)
            del readout

    estimates = estimates.flatten(0, 1)
    refiner = GlyphRefiner().to(device)
    with torch.no_grad():
        for layer in refiner.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=5 ** 0.5, generator=rng)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(refiner.output.weight)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    filters = _orientation_filters(device)
    if contour_epochs:
        from ocf_training import contour_loss
    for epoch in range(epochs):
        indices = torch.randperm(len(estimates), device=device, generator=rng)
        for batch in indices.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = refiner(estimates[batch])
            truth = images[batch.remainder(n_glyphs)]
            loss = _refinement_loss(logits.float(), truth, filters)
            if contour_epochs and epoch >= epochs - contour_epochs:
                probability = logits.float().sigmoid()[:, None]
                loss += 0.5 * contour_loss(probability, truth[:, None], sigmas=(1, 2), hard_fraction=0.75)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
    return refiner.eval()


def _png(scores, foreground, *, adapt=False):
    flattened = scores.flatten(1)
    if adapt:
        # Estimate the relative amount of ink without forcing every glyph to
        # cover the same area. Keep the estimate near the training-bank mean.
        ink = (0.6 * foreground + 0.4 * flattened.mean(dim=1)).clamp(0.04, 0.48)
        ranks = ((1 - ink) * (flattened.shape[1] - 1)).round().long()
        limit = flattened.sort(dim=1).values.gather(1, ranks[:, None])
    else:
        limit = torch.quantile(flattened, 1.0 - foreground, dim=1, keepdim=True)
    threshold = 0.75 * limit + 0.25 * 0.45
    return (flattened >= threshold).reshape_as(scores).to(torch.uint8) * 255


@dataclass
class CoarseInversion:
    encoder: RidgeReadout
    weights: torch.Tensor

    def predict(self, initial, responses):
        # Correct the 16 × 16 image to predict the observed neural response
        # while tethering the correction to the direct nonlinear estimate.
        coarse = F.avg_pool2d(initial[:, None].clamp(0, 1), 4)
        residuals = responses - self.encoder.predict(coarse.flatten(1))
        corrections = (residuals @ self.weights).reshape_as(coarse)
        corrections = F.interpolate(corrections, size=(64, 64), mode="bilinear", align_corners=False)[:, 0]
        return initial + corrections.clamp(-0.18, 0.18)


def _fit_inversion(readout, session_features, images):
    coarse = F.avg_pool2d(images[:, None], 4).flatten(1)
    encoders = {}
    for session, features in session_features.items():
        projected = (features - readout.centers[session]) @ readout.projection
        encoder = _fit_ridge(coarse, projected, ridge=0.005)
        gram = encoder.weights.T @ encoder.weights
        gram.diagonal().add_(0.8 * gram.diagonal().mean().clamp_min(1e-5))
        weights = torch.cholesky_solve(encoder.weights.T, torch.linalg.cholesky(gram))
        encoders[session] = CoarseInversion(encoder, weights)
    return encoders


@dataclass
class GlyphReconstructor:
    readout: ReconstructionReadout
    spatial_prior: torch.Tensor
    foreground: float
    refiner: GlyphRefiner | None
    inversions: dict[str, CoarseInversion]
    batch_size: int

    @torch.no_grad()
    def predict_variants(self, session: str, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return comparison candidates as [N, 64, 64] uint8 arrays on the fit device.

        ``linear`` is the local/pooled pixel ridge control. ``kernel`` applies
        the same threshold to the nonlinear score. ``kernel_ink`` allows
        per-glyph ink variation. When enabled, ``refined`` and ``refined_gray``
        are the hard and soft predictions from the residual image model.
        ``inverse`` corrects low-frequency nonlinear scores in response space.
        """
        features = features.to(device=self.spatial_prior.device, dtype=torch.float32)
        linear, nonlinear = self.readout.predict(session, features)
        variants = {"linear": _png(linear, self.foreground),
                    "kernel": _png(nonlinear, self.foreground),
                    "kernel_ink": _png(nonlinear, self.foreground, adapt=True)}
        if session in self.inversions:
            responses = (features - self.readout.centers[session]) @ self.readout.projection
            inverse = self.inversions[session].predict(nonlinear, responses)
            variants["inverse"] = _png(inverse, self.foreground, adapt=True)
        if self.refiner is not None:
            inputs = _estimates(linear, nonlinear, self.spatial_prior)
            probabilities = torch.cat([self.refiner(batch).sigmoid() for batch in inputs.split(self.batch_size)])
            variants["refined"] = _png(probabilities, self.foreground, adapt=True)
            variants["refined_gray"] = (probabilities * 255).round().to(torch.uint8)
        return variants

    def predict(self, session: str, features: torch.Tensor, *, variant: str = "kernel_ink") -> torch.Tensor:
        return self.predict_variants(session, features)[variant]


def fit_glyph_reconstructor(
    session_features: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    *,
    seed: int = 42,
    n_components: int = 160,
    linear_ridge: float = 0.16,
    kernel_ridge: float = 0.35,
    refine_epochs: int = 10,
    batch_size: int = 32,
    invert: bool = False,
    contour_epochs: int = 0,
) -> GlyphReconstructor:
    """Fit one mouse's readout using training glyphs in the same row order.

    Each value in ``session_features`` has shape [glyphs, features] and
    contains the per-glyph mean of preprocessed trial vectors for that
    session. Queries use the same preprocessing and repeat pooling.
    ``images`` contains only the corresponding training glyphs, as uint8
    0–255 or binary masks. The outer validation glyphs must be removed from
    both inputs by the caller. The refiner cross-fits these inputs again so
    that every example uses a pixel readout fit without that glyph.
    """
    if not session_features or images.ndim != 3 or images.shape[1:] != (64, 64) or not len(images):
        raise ValueError("Provide per-session features and at least one 64 × 64 training glyph.")
    shape = next(iter(session_features.values())).shape
    if len(shape) != 2 or shape[0] != len(images) or not shape[1] or any(
        features.shape != shape for features in session_features.values()
    ):
        raise ValueError("Feature matrices must agree on shape [training glyphs, features].")
    if (min(n_components, batch_size) < 1 or min(linear_ridge, kernel_ridge) <= 0
            or not 0 <= contour_epochs <= refine_epochs):
        raise ValueError("Invalid reconstruction settings.")

    device = next(iter(session_features.values())).device
    images = images.to(device)
    threshold = 0.5 if images.max().item() <= 1 else 128
    images = (images >= threshold).to(torch.float32)
    session_features = {session: features.to(device=device, dtype=torch.float32)
                        for session, features in session_features.items()}
    readout_options = (n_components, linear_ridge, kernel_ridge)
    refiner = None
    if refine_epochs and len(images) >= 2:
        refiner = _fit_refiner(readout_options, session_features, images, seed, refine_epochs, batch_size,
                               contour_epochs)
    with torch.no_grad():
        readout = _fit_readout(session_features, images, *readout_options)
        spatial_prior = images.mean(dim=0)
        inversions = _fit_inversion(readout, session_features, images) if invert else {}
    return GlyphReconstructor(readout, spatial_prior, spatial_prior.mean().item(), refiner, inversions, batch_size)
