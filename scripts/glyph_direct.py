"""Reconstruct seen glyphs directly from four-repeat or single-trial activity."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from glyph_reconstruction import RidgeReadout, _fit_ridge, _orientation_filters, _refinement_loss
from glyph_retrieval import _moments, _prototypes, _reliable_axes, _standardize, _variance_stabilize, _whiten
from ocf_training import contour_loss


@dataclass
class DirectDecoder:
    task: str
    normalization: dict[str, tuple[torch.Tensor, torch.Tensor]]
    projection: torch.Tensor
    ridge: RidgeReadout
    local_ridge: dict[str, RidgeReadout]
    network: nn.Module | None
    contour_network: nn.Module | None
    foreground: float


class SpikeImageDecoder(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.mapping = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 384), nn.GELU(),
                                     nn.Linear(384, 24 * 8 * 8))
        self.coarse = nn.Sequential(nn.Conv2d(24, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU())
        self.middle = nn.Sequential(nn.Conv2d(25, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU())
        self.fine = nn.Sequential(nn.Conv2d(17, 8, 3, padding=1), nn.GroupNorm(4, 8), nn.SiLU())
        self.output = nn.Conv2d(9, 1, 3, padding=1)

    def forward(self, features, ridge):
        ridge = ridge.float().clamp(0.015, 0.985)[:, None]
        canvas = self.mapping(features).reshape(-1, 24, 8, 8)
        canvas = self.coarse(F.interpolate(canvas, scale_factor=2, mode="bilinear", align_corners=False))
        low = F.avg_pool2d(ridge, 4)
        canvas = self.middle(F.interpolate(torch.cat((canvas, low), dim=1), scale_factor=2,
                                          mode="bilinear", align_corners=False))
        middle = F.avg_pool2d(ridge, 2)
        canvas = self.fine(F.interpolate(torch.cat((canvas, middle), dim=1), scale_factor=2,
                                        mode="bilinear", align_corners=False))
        update = self.output(torch.cat((canvas, ridge), dim=1))
        return ridge.logit()[:, 0] + update[:, 0]


def _training_features(groups):
    normalized, stats, covariances, centers = {}, {}, [], []
    for session, counts in groups.items():
        features = _variance_stabilize(counts, n_bins=10)
        stats[session] = _moments(features)
        features = _standardize(features, *stats[session])
        normalized[session] = features
        classes, repeats = features.shape[:2]
        labels = torch.arange(classes, device=features.device).repeat_interleave(repeats)
        center, _, covariance = _prototypes(features.flatten(0, 1), labels, classes)
        covariances.append(covariance)
        centers.append(center)

    covariance = torch.stack(covariances).mean(dim=0)
    covariance.diagonal().add_(0.25)
    lower = torch.linalg.cholesky(covariance)
    centers = torch.stack(centers)
    white = _whiten(centers.flatten(0, 1), lower).reshape(centers.shape)
    axes = _reliable_axes(white, torch.ones(centers.shape[1], dtype=torch.bool, device=centers.device))
    projection = torch.linalg.solve_triangular(lower.T, axes[:, -192:], upper=True)
    return {name: features @ projection for name, features in normalized.items()}, stats, projection


def _estimation_views(groups, task):
    if task == "task2":
        return {session: values.mean(dim=1) for session, values in groups.items()}
    return {session: values.flatten(0, 1) for session, values in groups.items()}


def _training_pixels(views, images):
    targets = {}
    for session, values in views.items():
        repeats = len(values) // len(images)
        targets[session] = images.repeat_interleave(repeats, dim=0)
    return targets


def _initialize_network(dimensions, device, generator):
    network = SpikeImageDecoder(dimensions).to(device=device, memory_format=torch.channels_last)
    with torch.no_grad():
        for layer in network.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight, a=5**0.5, generator=generator)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(network.output.weight)
    return network


def _optimize(network, features, estimates, targets, *, device, generator, epochs, batch_size,
              contour=False):
    network.train()
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.00025 if contour else 0.001, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00008)
    filters = _orientation_filters(device)
    for epoch in range(epochs):
        order = torch.randperm(len(features), device=device, generator=generator)
        if contour:
            order = order[: min(len(order), 4096)]
        for batch in order.split(batch_size):
            inputs = features[batch]
            mask = torch.rand(inputs.shape, device=device, generator=generator) >= 0.10
            inputs = inputs * mask + torch.randn(inputs.shape, device=device, generator=generator) * 0.03
            target = targets[batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = network(inputs, estimates[batch])
            logits = logits.float()
            loss = _refinement_loss(logits, target, filters)
            if contour:
                loss = loss + 0.65 * contour_loss(logits.sigmoid()[:, None], target[:, None], hard_fraction=0.25)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if epoch in (0, epochs - 1) or (epoch + 1) % 4 == 0:
            print(f"{device} {'contour' if contour else 'direct'} {epoch + 1}/{epochs} loss={loss.item():.4f}", flush=True)
    return network.eval()


def fit_direct(groups, images, task, *, seed=211, epochs=12, contour_epochs=1, batch_size=128):
    """Fit on [glyph, repeat, features] groups from the allowed training sessions."""
    projected, stats, projection = _training_features(groups)
    views = _estimation_views(projected, task)
    images = (images >= 128).float()
    image_targets = _training_pixels(views, images)
    features = torch.cat(tuple(views.values()))
    targets = torch.cat(tuple(image_targets.values()))
    ridge = _fit_ridge(features, targets.flatten(1), ridge=0.09)
    local = {}
    if task == "task3":
        local = {name: _fit_ridge(values, image_targets[name].flatten(1), ridge=0.14)
                 for name, values in views.items()}

    network = tuned = None
    if epochs > 0:
        device = features.device
        generator = torch.Generator(device=device).manual_seed(seed)
        # Out-of-fold ridge inputs prevent the convolutional head from learning
        # an unrealistically clean starting estimate for a memorized glyph.
        glyphs = torch.cat([torch.arange(len(images), device=device).repeat_interleave(len(values) // len(images))
                            for values in views.values()])
        assignment = torch.randperm(len(images), device=device, generator=generator).remainder(3)[glyphs]
        estimates = targets.new_empty(targets.shape)
        for fold in range(3):
            held = assignment == fold
            model = _fit_ridge(features[~held], targets[~held].flatten(1), ridge=0.09)
            estimates[held] = model.predict(features[held]).reshape(-1, 64, 64)

        network = _initialize_network(features.shape[1], device, generator)
        network = _optimize(network, features, estimates, targets, device=device, generator=generator,
                            epochs=epochs, batch_size=batch_size)
        if contour_epochs > 0:
            tuned = _optimize(deepcopy(network), features, estimates, targets, device=device, generator=generator,
                              epochs=contour_epochs, batch_size=batch_size, contour=True)
    return DirectDecoder(task, stats, projection, ridge, local, network, tuned, images.mean().item())


def _render(probabilities, foreground, *, fixed_threshold, mix):
    pixels = probabilities.flatten(1)
    ink = (0.65 * foreground + 0.35 * pixels.mean(dim=1, keepdim=True)).clamp(0.05, 0.50)
    rank = ((1 - ink) * (pixels.shape[1] - 1)).round().long()
    cutoff = pixels.sort(dim=1).values.gather(1, rank)
    cutoff = mix * fixed_threshold + (1 - mix) * cutoff
    image = ((pixels - cutoff) / 0.13 + 0.5).clamp(0, 1)
    return (image.reshape_as(probabilities) * 255).round().to(torch.uint8)


@torch.inference_mode()
def direct_images(model, counts, session, selection=None):
    """Return ridge, CNN, and optional contour-fine-tuned grayscale reconstructions."""
    features = _variance_stabilize(counts, n_bins=10)
    moments = model.normalization.get(session)
    if model.task == "task2" or moments is None:
        moments = _moments(features)
    if selection is not None:
        features = features[selection]
    features = _standardize(features, *moments).mean(dim=1) @ model.projection
    ridge = model.ridge.predict(features).reshape(-1, 64, 64)
    local = model.local_ridge.get(session)
    estimates = ridge if local is None else 0.80 * ridge + 0.20 * local.predict(features).reshape_as(ridge)
    predictions = {"ridge": _render(estimates, model.foreground, fixed_threshold=0.39, mix=0.30)}
    for name, network in (("cnn", model.network), ("contour", model.contour_network)):
        if network is None:
            continue
        probability = torch.cat([network(inputs, initial).sigmoid()
                                 for inputs, initial in zip(features.split(256), ridge.split(256))])
        predictions[name] = _render(probability, model.foreground, fixed_threshold=0.47, mix=0.40)
        if name == "cnn":
            predictions["cnn_gray"] = (probability * 255).round().to(torch.uint8)
    return predictions
