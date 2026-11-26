"""Cross-fit target-day glyph templates using only anonymous repeat groups."""

import numpy as np
import torch
from torch.nn import functional as F

from glyph_hierarchical_gallery import response_features
from glyph_retrieval import _template_logits, _whiten
from kaggle_diagnostics import standardize


FEATURE_RANK = 24
TEMPERATURE = 0.55
RIDGE = 0.80
MAX_TEMPLATE_SHIFT = 0.20
ROW_FOLDS = 4


@torch.no_grad()
def image_coordinates(masks):
    """Use only the public labeled gallery's morphology to parameterize drift."""
    pixels = masks.float()[:, None]
    dx = F.pad(pixels[..., 1:] - pixels[..., :-1], (0, 1, 0, 0))
    dy = F.pad(pixels[..., 1:, :] - pixels[..., :-1, :], (0, 0, 0, 1))
    views = (pixels, dx, dy, dx.abs(), dy.abs())
    value = torch.cat([F.avg_pool2d(view, 8).flatten(1) for view in views], dim=1)
    value = (value - value.mean(dim=0)) / value.std(dim=0, unbiased=False).clamp_min(0.02)
    _, vectors = torch.linalg.eigh(value.T @ value / len(value))
    reduced = value @ vectors[:, -FEATURE_RANK:]
    return reduced / reduced.std(dim=0, unbiased=False).clamp_min(1e-5)


@torch.no_grad()
def shuffled_coordinates(features, ink, energy, seed):
    """Shuffle detailed morphology among glyphs with comparable ink and energy."""
    def bucket(values):
        limits = torch.quantile(values, values.new_tensor((0.25, 0.50, 0.75)))
        return torch.bucketize(values.contiguous(), limits)

    group = 4 * bucket(ink) + bucket(energy)
    shuffled = torch.empty_like(features)
    rng = np.random.default_rng(seed)
    for label in group.unique():
        rows = (group == label).nonzero()[:, 0]
        order = rows[torch.as_tensor(rng.permutation(len(rows)), device=rows.device)]
        shuffled[order] = features[order.roll(1)]
    return shuffled


def covariance_factor(model, repeats):
    covariance = model.session_covariance + model.posterior_covariance + model.repeat_covariance / repeats
    covariance.diagonal().add_(model.predictive_floor)
    return torch.linalg.cholesky(0.5 * (covariance + covariance.T))


def score_scale(scores):
    return (scores - scores.mean(dim=1, keepdim=True)).square().mean().sqrt().clamp_min(1e-6)


def half_posterior(points, centers):
    scores = _template_logits(points, centers)
    scale = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (scores / (TEMPERATURE * scale)).softmax(dim=1)


def ridge_correction(design, residual, weight):
    w = weight / weight.sum().clamp_min(1e-8)
    weighted = design * w[:, None]
    gram = design.T @ weighted
    gram.diagonal().add_(RIDGE)
    return torch.cholesky_solve(weighted.T @ residual, torch.linalg.cholesky(gram))


@torch.no_grad()
def adapt(anonymous, source_model, features, shuffled, seed):
    """Fit a glyph-conditioned residual map off-row in both repeat directions."""
    if anonymous.ndim != 3 or anonymous.shape[1] != 4 or len(anonymous) != len(features):
        raise ValueError("Adaptation requires a complete quartet-repeat recording.")
    normalized, _ = standardize(response_features(anonymous, source_model.view), source_model.normalization)
    signal = _whiten(normalized.flatten(0, 1), source_model.noise_whitener).reshape_as(normalized)
    centers = source_model.posterior_templates
    half_factor = covariance_factor(source_model, 2)
    full_factor = covariance_factor(source_model, 4)
    full_centers = _whiten(centers, full_factor)
    full_query = _whiten(signal.mean(dim=1), full_factor)
    zero = _template_logits(full_query, full_centers)
    halves = signal[:, :2].mean(dim=1), signal[:, 2:].mean(dim=1)
    half_centers = _whiten(centers, half_factor)
    posteriors = tuple(half_posterior(_whiten(half, half_factor), half_centers) for half in halves)
    left, right = posteriors
    agreement = (left * right).sum(dim=1) / (left.norm(dim=1) * right.norm(dim=1)).clamp_min(1e-8)
    peak = (left.amax(dim=1) * right.amax(dim=1)).sqrt()
    weight = agreement.square() * (peak / (peak + 0.03))
    residue = (halves[1] - left @ centers, halves[0] - right @ centers)

    # The query's own rows participate in neither fit direction nor weight scaling.
    order = torch.as_tensor(np.random.default_rng(seed).permutation(len(anonymous)), device=anonymous.device)
    row_fold = torch.empty_like(order)
    row_fold[order] = torch.arange(len(order), device=order.device).remainder(ROW_FOLDS)
    scale = zero.new_empty(len(anonymous))
    for fold in range(ROW_FOLDS):
        held = row_fold == fold
        scale[held] = score_scale(zero[~held])
    max_norm = MAX_TEMPLATE_SHIFT * (centers - centers.mean(0)).norm(dim=1, keepdim=True)
    output, models = {}, {}
    for name, shape in (("morphology", features), ("shuffled", shuffled)):
        design = tuple(posterior @ shape for posterior in posteriors)
        corrections = torch.empty_like(zero)
        norms = []
        for fold in range(ROW_FOLDS):
            fitted = row_fold != fold
            held = ~fitted
            maps = [ridge_correction(x[fitted], y[fitted], weight[fitted]) for x, y in zip(design, residue)]
            shift = shape @ (0.5 * (maps[0] + maps[1]))
            shift *= (max_norm / shift.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
            update = _whiten(shift, full_factor)
            penalty = ((full_centers + 0.5 * update) * update).sum(dim=1)
            corrections[held] = full_query[held] @ update.T - penalty
            norms.append(float(shift.square().mean().sqrt()))
        output[name] = corrections
        models[name] = {"whitened_template_shift_rms_per_fold": norms,
                        "correction_rms_over_source_predictive_scale": float(
                            (corrections / scale[:, None]).square().mean().sqrt())}

    diagnostics = {"row_folds": ROW_FOLDS, "feature_rank": features.shape[1], "temperature": TEMPERATURE,
                   "agreement_weight_mean": float(weight.mean()), "mean_half_posterior_cosine": float(agreement.mean()),
                   "same_posterior_top1_fraction": float((left.argmax(1) == right.argmax(1)).float().mean()),
                   "mean_half_peak_probability": float(peak.mean()), "models": models}
    return zero, output, scale, diagnostics
