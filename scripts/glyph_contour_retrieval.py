"""Regress from visual responses to the expected contour signature of a seen glyph."""

from dataclasses import dataclass

import torch

from glyph_mbr import contour_utility
from glyph_reconstruction import RidgeReadout
from glyph_retrieval import _moments, _prototypes, _reliable_axes, _standardize, _variance_stabilize, _whiten


@dataclass
class FourierReadout:
    frequency: torch.Tensor
    phase: torch.Tensor
    readout: RidgeReadout


@dataclass
class ContourReadout:
    task: str
    view: str
    n_bins: int
    moments: dict[str, tuple[torch.Tensor, torch.Tensor]]
    projection: torch.Tensor
    signature: torch.Tensor
    energy: torch.Tensor
    readout: RidgeReadout
    local: dict[str, RidgeReadout]
    nonlinear: FourierReadout | None


def _response_features(counts, view, n_bins):
    if n_bins != 10 or counts.shape[-1] % n_bins:
        raise ValueError("Contour regression needs 10 adjacent 25 ms bins per unit")
    if view == "stabilized":
        return _variance_stabilize(counts, n_bins=n_bins)
    if view != "evoked-multiscale":
        raise ValueError(f"Unknown contour response view: {view}")
    delayed = counts.float().unflatten(-1, (-1, n_bins))[..., 2:]
    grouped = delayed.unflatten(-1, (4, 2)).sum(dim=-1)
    totals = grouped.sum(dim=-1, keepdim=True)
    return torch.cat((delayed, grouped, totals), dim=-1).sqrt().flatten(-2)


def _project_training(groups, view, n_bins):
    normalized, moments, centers, covariances = {}, {}, [], []
    for session, counts in groups.items():
        features = _response_features(counts, view, n_bins)
        moments[session] = _moments(features)
        features = _standardize(features, *moments[session])
        normalized[session] = features
        n_glyphs, repeats = features.shape[:2]
        labels = torch.arange(n_glyphs, device=features.device).repeat_interleave(repeats)
        center, _, covariance = _prototypes(features.flatten(0, 1), labels, n_glyphs)
        centers.append(center)
        covariances.append(covariance)

    covariance = torch.stack(covariances).mean(dim=0)
    covariance.diagonal().add_(0.25)
    lower = torch.linalg.cholesky(covariance)
    centers = torch.stack(centers)
    white = _whiten(centers.flatten(0, 1), lower).reshape(centers.shape)
    available = torch.ones(centers.shape[1], dtype=torch.bool, device=centers.device)
    axes = _reliable_axes(white, available)[:, -192:]
    projection = torch.linalg.solve_triangular(lower.T, axes, upper=True)
    return {session: features @ projection for session, features in normalized.items()}, moments, projection


def _fit_readout(views, targets, ridge):
    # Class-aligned repeat groups let us compute the cross moment without a
    # separate copy of the 5,000-dimensional target for every trial.
    counts = sum(values.shape[0] * values.shape[1] for values in views)
    per_glyph = sum(values.sum(dim=1) for values in views)
    center = per_glyph.sum(dim=0) / counts
    mean_target = targets.mean(dim=0)
    gram = sum(values.flatten(0, 1).T @ values.flatten(0, 1) for values in views) / counts
    gram -= center[:, None] * center
    gram.diagonal().add_(ridge)
    cross = per_glyph.T @ (targets - mean_target) / counts
    weights = torch.cholesky_solve(cross, torch.linalg.cholesky(gram))
    return RidgeReadout(weights, mean_target - center @ weights)


def _fourier_features(features, frequency, phase):
    extra = 0.35 * (features @ frequency + phase).cos()
    return torch.cat((features, extra), dim=-1)


@torch.no_grad()
def fit_contour_retrieval(groups, signature, energy, *, task, view="evoked-multiscale", n_bins=10,
                          seed=211, nonlinear=False):
    """Fit contour predictions using grouped-four or single-trial training views."""
    if not groups or task not in ("task2", "task3"):
        raise ValueError("Provide grouped Task 2 or Task 3 training data")
    if any(len(values) != len(signature) for values in groups.values()):
        raise ValueError("Training groups must be in the same glyph order as the contour signatures")
    projected, moments, projection = _project_training(groups, view, n_bins)
    if task == "task2":
        projected = {name: values.mean(dim=1, keepdim=True) for name, values in projected.items()}

    targets = torch.cat((signature, energy.log()[:, None]), dim=1)
    views = tuple(projected.values())
    readout = _fit_readout(views, targets, ridge=0.09)
    local = {}
    if task == "task3":
        local = {name: _fit_readout((values,), targets, ridge=0.14) for name, values in projected.items()}

    kernel = None
    if nonlinear:
        device = signature.device
        generator = torch.Generator(device=device).manual_seed(seed)
        radius = torch.stack([values.square().sum(dim=-1).mean() for values in views]).mean().sqrt()
        frequency = torch.randn((projection.shape[1], 256), device=device, generator=generator)
        frequency /= radius.clamp_min(0.10)
        phase = torch.rand(256, device=device, generator=generator) * (2 * torch.pi)
        expanded = tuple(_fourier_features(values, frequency, phase) for values in views)
        kernel = FourierReadout(frequency, phase, _fit_readout(expanded, targets, ridge=0.10))
    return ContourReadout(task, view, n_bins, moments, projection, signature, energy, readout, local, kernel)


def _agreement(estimate, model):
    # The ridge output approximates the expected signature. Its norm is
    # smaller for uncertain responses, so avoid forcing it to unit length.
    low, high = model.energy.log().aminmax()
    energy = estimate[:, -1].clamp(low, high).exp()
    return contour_utility(estimate[:, :-1], energy, model.signature, model.energy)


@torch.inference_mode()
def contour_retrieval_scores(model, query, session, *, repeat_weights=None):
    """Score each hard gallery glyph by expected contour agreement with the query."""
    features = _response_features(query, model.view, model.n_bins)
    stats = model.moments.get(session)
    if stats is None:
        stats = _moments(features)
    features = _standardize(features, *stats)
    if repeat_weights is None:
        features = features.mean(dim=1)
    else:
        features = (features * repeat_weights[..., None]).sum(dim=1)
    features = features @ model.projection
    estimate = model.readout.predict(features)
    if session in model.local:
        estimate = 0.80 * estimate + 0.20 * model.local[session].predict(features)
    scores = {"contour_ridge": _agreement(estimate, model)}
    if model.nonlinear is not None:
        kernel = model.nonlinear
        expanded = _fourier_features(features, kernel.frequency, kernel.phase)
        scores["contour_fourier"] = _agreement(kernel.readout.predict(expanded), model)
    return scores


def blend_contour_poisson(contour, poisson):
    """Combine equally weighted shape and likelihood rankings for each sample."""
    def standardize(scores):
        centered = scores - scores.mean(1, keepdim=True)
        return centered / centered.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)

    return 0.5 * (standardize(contour) + standardize(poisson))
