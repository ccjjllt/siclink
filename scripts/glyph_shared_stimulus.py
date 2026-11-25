"""Learn shared glyph codes and source-constrained observation drift for Task 2."""

from dataclasses import dataclass

import torch

from glyph_hierarchical_gallery import _gaussian_logits, _positive_covariance, _repeat_covariance, response_features


@dataclass
class SharedStimulusGallery:
    view: str
    scale: torch.Tensor
    code: torch.Tensor
    loading: torch.Tensor
    drift: torch.Tensor
    residual_templates: torch.Tensor
    noise: torch.Tensor
    session_error: torch.Tensor
    diagnostics: dict

    @torch.no_grad()
    def logits(self, query, *, noise_mix=0.50, maximum_change=0.15):
        """Adapt only source-observed loading directions using unpaired moments."""
        values = response_features(query, self.view) / self.scale
        values -= values.mean(dim=(0, 1), keepdim=True)
        response = values.mean(1)
        repeats = values.shape[1]
        target_noise = _repeat_covariance(values)
        templates = self.code @ self.loading + self.residual_templates
        source_precision_cov = self.noise / repeats + self.session_error
        source_precision_cov.diagonal().add_(0.05)
        target_precision_cov = self.noise.lerp(target_noise, noise_mix) / repeats + self.session_error
        target_precision_cov.diagonal().add_(0.05)

        target_signal = response.T @ response / len(response) - target_noise / repeats
        anchor_signal = templates.T @ templates / len(templates)
        anchor_signal -= self.noise / (repeats * len(self.drift))
        tangent = self.loading.T @ self.drift
        tangent = tangent + tangent.transpose(1, 2)
        features = len(self.scale)
        off_diagonal = torch.triu_indices(features, features, offset=1, device=values.device)
        first, second = off_diagonal
        units = self.noise.diagonal().clamp_min(0.10).sqrt()
        scale = units[first] * units[second]
        jacobian = (tangent[:, first, second] / scale).T
        difference = (target_signal - anchor_signal)[first, second] / scale
        normal = jacobian.T @ jacobian
        normal.diagonal().add_(0.50 * normal.diagonal().mean().clamp_min(1e-5))
        coefficients = torch.linalg.solve(normal, jacobian.T @ difference)
        loading_delta = (coefficients[:, None, None] * self.drift).sum(0)
        delta = self.code @ loading_delta
        limit = maximum_change * templates.norm()
        strength = (limit / delta.norm().clamp_min(1e-8)).clamp_max(1)
        adapted = templates + strength * delta
        effective = strength * coefficients
        model_signal = adapted.T @ adapted / len(adapted) - self.noise / (repeats * len(self.drift))
        corrected = (target_signal - model_signal)[first, second] / scale
        diagnostics = {"coarse_windows": self.view, "target_repeats": repeats,
                       "loading_coefficients": effective.tolist(),
                       "relative_template_change": float((adapted - templates).norm() / templates.norm().clamp_min(1e-8)),
                       "offdiag_moment_rms_frozen": float(difference.square().mean().sqrt()),
                       "offdiag_moment_rms_adapted": float(corrected.square().mean().sqrt())}
        return {"frozen_shared": _gaussian_logits(response, templates, source_precision_cov),
                "target_noise_only": _gaussian_logits(response, templates, target_precision_cov),
                "adapted_loading": _gaussian_logits(response, adapted, target_precision_cov)}, diagnostics


@torch.no_grad()
def fit_shared_stimulus(training, *, rank=32, view="total"):
    """Fit aligned source glyph codes from coarse repeat means and neuron loadings."""
    source = [response_features(counts, view) for counts in training.values()]
    if len(source) < 2 or len(source[0]) < 3 or rank < 1:
        raise ValueError("Provide at least two aligned recordings and a positive latent rank.")
    sessions, glyphs, repeats, features = len(source), *source[0].shape
    scale = (torch.stack([item.flatten(0, 1).var(dim=0, unbiased=False) for item in source]).mean(0) + 0.05).sqrt()
    normalized = [(item - item.mean(dim=(0, 1), keepdim=True)) / scale for item in source]
    means = torch.stack([item.mean(dim=1) for item in normalized])
    covariances = torch.stack([_repeat_covariance(item) for item in normalized])
    noise = covariances.mean(0)
    trial_scale = (noise.diagonal() + 0.15).sqrt()

    # Remove finite-repeat noise only from within-day covariance blocks. The
    # between-day blocks use corresponding labeled glyphs and share no trials.
    concatenated = means.transpose(0, 1).reshape(glyphs, sessions * features) / trial_scale.repeat(sessions)
    covariance = concatenated.T @ concatenated / glyphs
    scales = trial_scale[:, None] * trial_scale[None, :]
    for index, trial_covariance in enumerate(covariances):
        columns = slice(index * features, (index + 1) * features)
        covariance[columns, columns] -= 0.75 * trial_covariance / (repeats * scales)
    eigenvalues, axes = torch.linalg.eigh(covariance)
    dimension = min(rank, len(axes), glyphs - 1, max(1, int((eigenvalues > 1e-5).sum())))
    code, _ = torch.linalg.qr(concatenated @ axes[:, -dimension:], mode="reduced")
    code *= glyphs**0.5
    loads = code.T @ means / glyphs
    loading = loads.mean(0)
    drift = 0.65 * (loads - loading)
    pooled = means.mean(0)
    residual = pooled - code @ loading
    errors = (means - (code @ (loading + drift) + residual)).flatten(0, 1)
    extra_cov = errors.T @ errors / len(errors) - (1 - 1 / sessions) * noise / repeats
    session_error = _positive_covariance(extra_cov, diagonal_fraction=0.30, floor=0)
    diagnostics = {"rank": dimension, "sessions": list(training), "glyphs": glyphs, "neuronal_features": features,
                   "pooled_shared_fraction": float((code @ loading).square().sum() / pooled.square().sum().clamp_min(1e-8)),
                   "noise_trace": float(noise.trace()), "session_error_trace": float(session_error.trace())}
    return SharedStimulusGallery(view, scale, code, loading, drift, residual, noise, session_error, diagnostics)
