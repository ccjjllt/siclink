"""Estimate stable glyph responses and a new recording's predictive uncertainty."""

from dataclasses import dataclass
from typing import Mapping

import torch

from glyph_retrieval import _template_logits, _whiten
from kaggle_diagnostics import standardize


VIEWS = {"total": ((50, 250),), "two-windows": ((50, 150), (150, 250))}
EB_BLEND_WEIGHT = 0.35


def blend_hierarchical_logits(validated, hierarchical):
    """Combine two score matrices while retaining relative query contrast."""
    def rescale(matrix):
        centered = matrix - matrix.mean(dim=1, keepdim=True)
        return centered / centered.square().mean().sqrt().clamp_min(1e-6)

    return rescale(validated).lerp(rescale(hierarchical), EB_BLEND_WEIGHT)


def response_features(counts, view):
    units = counts.float().unflatten(-1, (-1, 10))
    windows = torch.stack([units[..., start // 25:end // 25].sum(dim=-1) for start, end in VIEWS[view]], dim=-1)
    return windows.sqrt().flatten(-2)


def _symmetric(matrix):
    return 0.5 * (matrix + matrix.T)


def _repeat_covariance(values):
    residual = (values - values.mean(dim=1, keepdim=True)).flatten(0, 1)
    return residual.T @ residual / (len(values) * (values.shape[1] - 1))


def _whiten_covariance(covariance, factor):
    left = torch.linalg.solve_triangular(factor, covariance, upper=False)
    return _symmetric(torch.linalg.solve_triangular(factor, left.T.contiguous(), upper=False).T)


def _positive_covariance(matrix, *, diagonal_fraction, floor):
    values, vectors = torch.linalg.eigh(_symmetric(matrix))
    values = values.clamp_min(0)
    spectrum = (1 - diagonal_fraction) * values + diagonal_fraction * values.mean() + floor
    return (vectors * spectrum) @ vectors.T


def _gaussian_logits(query, centers, covariance):
    factor = torch.linalg.cholesky(_symmetric(covariance))
    return _template_logits(_whiten(query, factor), _whiten(centers, factor))


@dataclass
class HierarchicalGallery:
    view: str
    normalization: tuple[torch.Tensor, torch.Tensor]
    noise_whitener: torch.Tensor
    repeat_covariance: torch.Tensor
    session_covariance: torch.Tensor
    posterior_covariance: torch.Tensor
    mean_templates: torch.Tensor
    posterior_templates: torch.Tensor
    predictive_floor: float
    diagnostics: dict

    @torch.no_grad()
    def logits(self, query, *, query_repeat_mix=0.0):
        """Score each query with no alignment or assigned labels from its block."""
        if (query.ndim != 3 or not len(query) or not query.shape[1]
                or not 0 <= query_repeat_mix <= 1):
            raise ValueError("Provide grouped query repeats and a repeat covariance weight in [0,1].")
        trial_count = query.shape[1]
        if query_repeat_mix and trial_count < 2:
            raise ValueError("Estimating query noise requires at least two repeats per sample.")

        features, _ = standardize(response_features(query, self.view), self.normalization)
        pooled = _whiten(features.mean(dim=1), self.noise_whitener)
        ideal = self.repeat_covariance / trial_count
        ideal.diagonal().add_(self.predictive_floor)
        day = ideal + self.session_covariance
        integrated = day + self.posterior_covariance
        scores = {"constant": _gaussian_logits(pooled, self.mean_templates, ideal),
                  "session_variance": _gaussian_logits(pooled, self.mean_templates, day),
                  "empirical_bayes": _gaussian_logits(pooled, self.posterior_templates, integrated)}
        if query_repeat_mix:
            query_noise = _whiten_covariance(_repeat_covariance(features), self.noise_whitener)
            measured = self.repeat_covariance.lerp(query_noise, query_repeat_mix) / trial_count
            measured.diagonal().add_(self.predictive_floor)
            measured = measured + self.session_covariance + self.posterior_covariance
            scores["empirical_bayes_repeat_noise"] = _gaussian_logits(pooled, self.posterior_templates, measured)
        return scores


@torch.no_grad()
def fit_hierarchical_gallery(
    training: Mapping[str, torch.Tensor],
    *,
    view: str = "total",
    predictive_floor: float = 0.05,
) -> HierarchicalGallery:
    """Fit y[g,s,r] = stable[g] + shift[g,s] + repeat[g,s,r].

    All recordings must have the same labeled glyph and unit row order.
    The model treats labeled sessions as exchangeable recording realizations.
    A future recording uses pooled training feature moments. Its optional
    noise-only update uses residuals within each grouped query's repeats.

    The stable and recording-specific responses have shared Gaussian
    covariances across glyphs. Repeat errors are independent conditional on
    the glyph and recording. The stable covariance comes from pairs of
    labeled recordings, while the recording-specific covariance subtracts
    finite-repeat noise from the variance of each glyph's recording means.
    """
    if len(training) < 2 or view not in VIEWS or predictive_floor <= 0:
        raise ValueError("Provide at least two labeled sessions and positive predictive regularization.")
    first = next(iter(training.values()))
    if (first.ndim != 3 or len(first) < 2 or not first.shape[2] or first.shape[2] % 10
            or any(block.ndim != 3 or len(block) != len(first) or block.shape[1] < 2
                   or block.shape[2] != first.shape[2] for block in training.values())):
        raise ValueError("Each recording needs aligned [glyphs, repeats, units × 10] counts.")

    calibrations, means, noise, repeats = [], [], [], []
    for block in training.values():
        normalized, calibration = standardize(response_features(block, view))
        calibrations.append(calibration)
        means.append(normalized.mean(dim=1))
        noise.append(_repeat_covariance(normalized))
        repeats.append(normalized.shape[1])
    query_normalization = tuple(torch.stack(items).mean(dim=0) for items in zip(*calibrations))

    pooled_noise = torch.stack(noise).mean(dim=0)
    regularized = pooled_noise.clone()
    regularized.diagonal().add_(0.20 * pooled_noise.diagonal().mean().clamp_min(0.02))
    whitener = torch.linalg.cholesky(regularized)
    noise = [_whiten_covariance(covariance, whitener) for covariance in noise]
    pooled_noise = torch.stack(noise).mean(dim=0)
    means = torch.stack([_whiten(center, whitener) for center in means])
    sessions, glyphs, dimension = means.shape
    grand_mean = means.mean(dim=(0, 1))
    centered = means - grand_mean

    total = centered.sum(dim=0)
    stacked = centered.flatten(0, 1)
    cross_session = (total.T @ total - stacked.T @ stacked) / (sessions * (sessions - 1) * glyphs)
    stable = _positive_covariance(cross_session, diagonal_fraction=0.06, floor=1e-4)
    difference = (centered - centered.mean(dim=0, keepdim=True)).flatten(0, 1)
    mean_variation = difference.T @ difference / ((sessions - 1) * glyphs)
    expected_repeat = torch.stack([covariance / count for covariance, count in zip(noise, repeats)]).mean(dim=0)
    session_deviation = _positive_covariance(mean_variation - expected_repeat, diagonal_fraction=0.18, floor=0)

    # Each recorded glyph mean observes the same latent response with known
    # session-level plus finite-repeat uncertainty. Integrate that uncertainty
    # before predicting a mean for an independent recording.
    precision = torch.cholesky_inverse(torch.linalg.cholesky(_symmetric(stable)))
    information = centered.new_zeros((glyphs, dimension))
    for observed, covariance, repeat_count in zip(centered, noise, repeats):
        variance = session_deviation + covariance / repeat_count
        variance.diagonal().add_(predictive_floor)
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(_symmetric(variance)))
        precision = precision + inverse
        information = information + observed @ inverse
    posterior = torch.cholesky_inverse(torch.linalg.cholesky(_symmetric(precision)))
    posterior_templates = information @ posterior + grand_mean
    unshrunk_templates = means.mean(dim=0)
    one_recording = stable + session_deviation + expected_repeat
    diagnostics = {"labeled_sessions": sessions, "glyphs": glyphs, "dimensions": dimension,
                   "training_repeats": repeats,
                   "stable_signal_trace": float(stable.trace()),
                   "session_glyph_shift_trace": float(session_deviation.trace()),
                   "repeat_mean_noise_trace": float(expected_repeat.trace()),
                   "posterior_mean_uncertainty_trace": float(posterior.trace()),
                   "stable_fraction_of_training_repeat_mean_variance": float(stable.trace() / one_recording.trace()),
                   "template_rms_ratio": float((posterior_templates - grand_mean).norm() /
                                               (unshrunk_templates - grand_mean).norm().clamp_min(1e-8))}
    return HierarchicalGallery(view, query_normalization, whitener, pooled_noise, session_deviation,
                               posterior, unshrunk_templates, posterior_templates, predictive_floor, diagnostics)
