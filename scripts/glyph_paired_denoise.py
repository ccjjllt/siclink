"""Fit spike embeddings from independent repeats of the same labeled glyph."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from glyph_contour_retrieval import _project_training, _response_features
from glyph_reconstruction import RidgeReadout, _fit_ridge
from glyph_retrieval import _prototypes, _standardize


@dataclass
class PairedDenoiser:
    moments: dict
    projection: torch.Tensor
    linear: RidgeReadout
    residual: nn.Module
    galleries: dict[str, dict[str, RidgeReadout]]
    diagnostics: dict


def _repeat_targets(projected):
    means = {session: values.mean(dim=1) for session, values in projected.items()}
    total = torch.stack(list(means.values())).sum(dim=0)
    output = []
    for session, values in projected.items():
        local = (values.sum(dim=1, keepdim=True) - values) / (values.shape[1] - 1)
        cross_session = (total - means[session]) / (len(projected) - 1)
        output.append(0.60 * local + 0.40 * cross_session[:, None])
    return torch.cat([values.flatten(0, 1) for values in output])


def _fit_residual(features, targets, linear, epochs, seed):
    device = features.device
    generator = torch.Generator(device=device).manual_seed(seed)
    dimensions = features.shape[-1]
    residual = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 128), nn.GELU(),
                             nn.Linear(128, dimensions)).to(device)
    with torch.no_grad():
        nn.init.kaiming_uniform_(residual[1].weight, a=5**0.5, generator=generator)
        nn.init.zeros_(residual[1].bias)
        nn.init.zeros_(residual[3].weight)
        nn.init.zeros_(residual[3].bias)
        initial = linear.predict(features)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=0.001, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00012)
    for _ in range(epochs):
        order = torch.randperm(len(features), device=device, generator=generator)
        for rows in order.split(512):
            incoming = features[rows]
            keep = torch.rand(incoming.shape, device=device, generator=generator) >= 0.12
            change = residual(incoming * keep)
            loss = F.mse_loss(initial[rows] + change, targets[rows]) + 0.15 * change.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return residual.eval()


def _fit_galleries(groups):
    centers, covariances = {}, []
    for session, values in groups.items():
        classes, repeats = values.shape[:2]
        labels = torch.arange(classes, device=values.device).repeat_interleave(repeats)
        center, _, noise = _prototypes(values.flatten(0, 1), labels, classes)
        centers[session] = center
        covariances.append(noise)
    pooled = torch.stack(list(centers.values())).mean(dim=0)
    cov = torch.stack(covariances).mean(dim=0)
    diagonal = cov.diagonal()
    diagonal.add_(0.40 * diagonal.mean().clamp_min(1e-4))
    factor = torch.linalg.cholesky(cov)
    galleries = {}
    for session, local in centers.items():
        center = 0.55 * local + 0.45 * pooled
        weights = torch.cholesky_solve(center.T, factor)
        bias = -0.5 * (center * weights.T).sum(dim=1)
        galleries[session] = RidgeReadout(weights, bias)
    return galleries


def fit_paired_denoiser(training, *, epochs=10, seed=211):
    """Learn a regression, with no class head, from paired training repeats."""
    if len(training) < 2 or any(counts.shape[1] < 2 for counts in training.values()) or epochs < 1:
        raise ValueError("Paired denoising requires repeated training trials in multiple sessions")
    with torch.no_grad():
        projected, moments, projection = _project_training(training, "evoked-multiscale", 10)
        features = torch.cat([values.flatten(0, 1) for values in projected.values()])
        targets = _repeat_targets(projected)
        linear = _fit_ridge(features, targets, ridge=0.09)
    residual = _fit_residual(features, targets, linear, epochs, seed)
    with torch.no_grad():
        linear_views = {session: linear.predict(values) for session, values in projected.items()}
        mlp_views = {session: linear_views[session] + residual(values) for session, values in projected.items()}
        galleries = {"raw_latent": _fit_galleries(projected), "paired_linear": _fit_galleries(linear_views),
                     "paired_mlp": _fit_galleries(mlp_views)}
        initial = linear.predict(features)
        diagnostics = {"fit_pairs": len(features), "latent_dimensions": features.shape[1],
                       "fit_linear_mse": float(F.mse_loss(initial, targets)),
                       "fit_mlp_mse": float(F.mse_loss(initial + residual(features), targets))}
    return PairedDenoiser(moments, projection, linear, residual, galleries, diagnostics)


@torch.inference_mode()
def paired_glyph_scores(model, query, session):
    features = _response_features(query, "evoked-multiscale", 10)
    features = _standardize(features, *model.moments[session]).mean(dim=1) @ model.projection
    linear = model.linear.predict(features)
    views = {"raw_latent": features, "paired_linear": linear, "paired_mlp": linear + model.residual(features)}
    return {name: model.galleries[name][session].predict(values) for name, values in views.items()}
