"""Remove repeat-to-repeat response noise predictable from early post-onset spikes."""

from dataclasses import dataclass
from typing import Mapping

import torch

from glyph_retrieval import _template_logits, _whiten
from kaggle_diagnostics import standardize


CONDITIONERS = ("unconditioned", "early_population", "early_units")
EVOKED_WINDOWS = {"total": ((50, 250),), "two-windows": ((50, 150), (150, 250))}


def _spike_features(counts, early_ms, view):
    spike_bins = counts.float().unflatten(-1, (-1, 10))
    early_counts = spike_bins[..., :early_ms // 25].sum(dim=-1)
    early = torch.cat((early_counts.sqrt(), early_counts.sum(dim=-1, keepdim=True).sqrt()), dim=-1)
    evoked = torch.stack([spike_bins[..., start // 25:end // 25].sum(dim=-1)
                          for start, end in EVOKED_WINDOWS[view]], dim=-1)
    return early, evoked.sqrt().flatten(-2)


def _residual_moments(early, evoked):
    residual_early = (early - early.mean(dim=1, keepdim=True)).flatten(0, 1)
    residual_evoked = (evoked - evoked.mean(dim=1, keepdim=True)).flatten(0, 1)
    degrees = len(early) * (early.shape[1] - 1)
    return residual_early.T @ residual_early / degrees, residual_early.T @ residual_evoked / degrees


@dataclass
class EarlyNoiseConditioner:
    early_ms: int
    view: str
    calibration: dict
    coefficients: dict[str, dict[str, torch.Tensor]]

    def transform(self, counts, session, condition):
        early_stats, evoked_stats = self.calibration[session]
        early, evoked = _spike_features(counts, self.early_ms, self.view)
        evoked, _ = standardize(evoked, evoked_stats)
        if condition == "unconditioned":
            return evoked
        early, _ = standardize(early, early_stats)
        return evoked - early @ self.coefficients[session][condition]


@torch.no_grad()
def fit_early_conditioner(
    training: Mapping[str, torch.Tensor],
    *,
    early_ms: int = 25,
    view: str = "total",
    ridge: float = 0.45,
    shared_fraction: float = 0.35,
) -> EarlyNoiseConditioner:
    """Regress within-glyph evoked residuals on within-glyph early residuals.

    Inputs are one mouse's grouped training repeats [N,R,units × 10].
    All normalization and nuisance moments use only those input repeats.
    The first 25 or 50 ms is after stimulus onset.
    """
    if not training or early_ms not in (25, 50) or view not in EVOKED_WINDOWS or ridge <= 0:
        raise ValueError("Choose training blocks, an early window, a response view, and positive ridge regularization.")
    if not 0 <= shared_fraction <= 1:
        raise ValueError("The shared fraction must fall in [0,1].")
    calibration, moments = {}, {}
    for session, counts in training.items():
        if counts.ndim != 3 or not len(counts) or counts.shape[1] < 2 or not counts.shape[2] or counts.shape[2] % 10:
            raise ValueError("Training needs [glyphs, at least two repeats, units × 10] for each recording.")
        early, evoked = _spike_features(counts, early_ms, view)
        early, early_stats = standardize(early)
        evoked, evoked_stats = standardize(evoked)
        calibration[session] = (early_stats, evoked_stats)
        moments[session] = _residual_moments(early, evoked)

    pooled_gram = torch.stack([gram for gram, _ in moments.values()]).mean(dim=0)
    pooled_cross = torch.stack([cross for _, cross in moments.values()]).mean(dim=0)
    coefficients = {}
    for session, (gram, cross) in moments.items():
        gram = gram.lerp(pooled_gram, shared_fraction)
        cross = cross.lerp(pooled_cross, shared_fraction)
        population = torch.zeros_like(cross)
        population[-1] = cross[-1] / (gram[-1, -1] * (1 + ridge)).clamp_min(1e-5)
        gram.diagonal().add_(ridge * gram.diagonal().mean().clamp_min(0.02))
        units = torch.cholesky_solve(cross, torch.linalg.cholesky(gram))
        coefficients[session] = {"early_population": population, "early_units": units}
    return EarlyNoiseConditioner(early_ms, view, calibration, coefficients)


@dataclass
class EvokedTemplate:
    centers: torch.Tensor
    noise_factor: torch.Tensor

    def logits(self, query):
        return _template_logits(_whiten(query.mean(dim=1), self.noise_factor), self.centers)


@torch.no_grad()
def fit_evoked_templates(training, *, noise_ridge=0.35, local_fraction=0.75):
    """Fit the same local/pooled Gaussian template scorer for each response variant."""
    centers, covariance = {}, {}
    for session, features in training.items():
        centers[session] = features.mean(dim=1)
        residual = (features - centers[session][:, None]).flatten(0, 1)
        degrees = len(features) * (features.shape[1] - 1)
        covariance[session] = residual.T @ residual / degrees
    pooled_center = torch.stack(tuple(centers.values())).mean(dim=0)
    pooled_covariance = torch.stack(tuple(covariance.values())).mean(dim=0)
    result = {}
    for session, local in centers.items():
        center = pooled_center.lerp(local, local_fraction)
        noise = pooled_covariance.lerp(covariance[session], local_fraction)
        noise.diagonal().add_(noise_ridge)
        factor = torch.linalg.cholesky(noise)
        result[session] = EvokedTemplate(_whiten(center, factor), factor)
    return result


@torch.no_grad()
def split_repeat_variance(raw_fit, raw_query, corrected_fit, corrected_query):
    """Use labels only for this diagnostic of independent held-repeat residuals."""
    before = raw_query - raw_fit.mean(dim=1, keepdim=True)
    after = corrected_query - corrected_fit.mean(dim=1, keepdim=True)
    before_variance = before.square().mean()
    after_variance = after.square().mean()
    return {"held_repeat_variance_before": before_variance.item(),
            "held_repeat_variance_after": after_variance.item(),
            "held_repeat_variance_explained": (1 - after_variance / before_variance.clamp_min(1e-8)).item()}
